#!/usr/bin/env python3
"""
Stage-2 feature caching for Phase 3.

For each of 1,118 DILIst drugs:
  - Query MODEL_DOSE (E-Hill scalar) at each of 10 LINCS cells, mean-pool -> feat_dose [1]
  - Query MODEL_GEX (perturbed GEX prediction) at each of 10 LINCS cells, compute DE,
    mean-pool -> feat_gex [num_gene]
  - Encode SMILES with frozen MolFormer: feat_embed [768]
  - Concatenate -> feature row [1 + num_gene + 768]
Emit data/processed/dili_features.parquet (1118 rows).

CELL COUNT DEVIATION: Uses 10 cells (A375, A549, BT20, HA1E, HELA, HT29, MCF7,
MDAMB231, PC3, VCAP) matching actual training distribution, NOT the 9 cited in the
design doc (which omits HA1E).

FEATURE DIM DEVIATION: feat_gex = num_gene from chkpt_gex.pt['model_params']['num_gene']
(919, NOT 978 as stated in design doc — only 919 landmark genes overlap between
gene_vector.csv and lincs_train_safe.parquet when reading gene_vector.csv with header=None).

MODEL_DOSE uses all 978 landmark genes (CCLE-based 978-dim cell baseline).
MODEL_GEX uses 919 landmark genes (PDG diseased-based 919-dim cell baseline).
"""
import sys, os, argparse

# ============================================================================
# SET CUDA_VISIBLE_DEVICES BEFORE IMPORTING TORCH (CLAUDE.md hard rule 4)
# ============================================================================
def _get_gpu_arg():
    for i, a in enumerate(sys.argv):
        if a == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

_gpu = _get_gpu_arg()
if _gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu
    print(f'[cache_dili_features] CUDA_VISIBLE_DEVICES={_gpu}')

# ----- Now safe to import torch -----
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Project root and MultiDCP paths
# ============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_MULTIDCP_MODELS = Path('/raid/home/joshua/projects/MultiDCP/MultiDCP/models')
_MULTIDCP_UTILS  = Path('/raid/home/joshua/projects/MultiDCP/MultiDCP/utils')
sys.path.insert(0, str(_MULTIDCP_MODELS))
sys.path.insert(0, str(_MULTIDCP_UTILS))

# Canonical data paths
GENE_VECTOR_FILE = Path('/raid/home/joshua/data/MultiDCP/data/gene_vector.csv')
DISEASED_FILE    = Path('/raid/home/joshua/projects/MultiDCP/MultiDCP/data/'
                        'pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv')
CCLE_FILE        = Path('/raid/home/joshua/data/MultiDCP/data/'
                        'adjusted_ccle_tcga_ad_tpm_log2.csv')

CELLS_10 = ['A375', 'A549', 'BT20', 'HA1E', 'HELA', 'HT29', 'MCF7', 'MDAMB231', 'PC3', 'VCAP']


# ============================================================================
# Argument parsing
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Stage-2 feature caching for Phase 3')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU index (sets CUDA_VISIBLE_DEVICES before torch import)')
    parser.add_argument('--dose-ckpt', type=str, required=True,
                        help='Path to chkpt_dose.pt (MODEL_DOSE checkpoint)')
    parser.add_argument('--gex-ckpt', type=str, required=True,
                        help='Path to chkpt_gex.pt (MODEL_GEX checkpoint)')
    parser.add_argument('--dili-csv', type=str, required=True,
                        help='Path to dili_canonical.csv (1118 drugs with SMILES + DILI labels)')
    parser.add_argument('--out', type=str, default='data/processed/dili_features.parquet',
                        help='Output parquet path')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for MolFormer SMILES encoding')
    parser.add_argument('--wandb-project', type=str, default='joshroll/MultiDCP_multihead_dili')
    parser.add_argument('--wandb-group',   type=str, default='multihead_dili')
    parser.add_argument('--wandb-name',    type=str, default='stage2_cache_P3')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    return parser.parse_args()


# ============================================================================
# Model loading helpers
# ============================================================================
def load_dose_model(ckpt_path: str, device: torch.device):
    """Load MODEL_DOSE (MultiDCPEhillPretraining) from checkpoint.

    Returns: (model, model_params dict)
    """
    import multidcp_balanceloss as multidcp

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_params = dict(ckpt['model_params'])
    print(f'  MODEL_DOSE model_params: num_gene={model_params.get("num_gene")}, '
          f'cell_id_input_dim={model_params.get("cell_id_input_dim")}, '
          f'pert_idose_input_dim={model_params.get("pert_idose_input_dim")}')

    model = multidcp.MultiDCPEhillPretraining(device=device,
                                               model_param_registry=model_params)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    model = model.double()
    for param in model.parameters():
        param.requires_grad = False
    return model, model_params


def load_gex_model(ckpt_path: str, device: torch.device):
    """Load MODEL_GEX (MultiDCP_AE) from checkpoint.

    Returns: (model, model_params dict)
    """
    import multidcp_balanceloss as multidcp

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_params = dict(ckpt['model_params'])
    print(f'  MODEL_GEX model_params: num_gene={model_params.get("num_gene")}, '
          f'cell_id_input_dim={model_params.get("cell_id_input_dim")}')

    model = multidcp.MultiDCP_AE(device=device, model_param_registry=model_params)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    model = model.double()
    for param in model.parameters():
        param.requires_grad = False
    return model, model_params


# ============================================================================
# Gene and cell baseline preparation
# ============================================================================
def build_gene_filter_and_baselines(dose_num_gene: int, gex_num_gene: int):
    """Build landmark gene sets and cell baseline tensors for both models.

    MODEL_DOSE: reads gene_vector.csv (default header), expects dose_num_gene from checkpoint
        (all landmark genes, typically 978).
    MODEL_GEX:  reads gene_vector.csv (header=None), overlaps with LINCS parquet,
                expects gex_num_gene=919.

    Returns:
        dose_gene_tensor:    torch.DoubleTensor [dose_num_gene, 128]
        gex_landmark_genes:  list of str (919 gene names in training order)
        gex_gene_tensor:     torch.DoubleTensor [gex_num_gene, 128]
        dose_baselines:      dict {cell: DoubleTensor [dose_num_gene]}
        gex_baselines:       dict {cell: DoubleTensor [gex_num_gene]}
    """
    # ---- Gene topology for MODEL_DOSE (978-gene, default csv read) ----
    gene_df_dose = pd.read_csv(GENE_VECTOR_FILE, index_col=0)  # 977 rows with default header
    dose_genes = gene_df_dose.index.tolist()
    dose_gene_tensor = torch.tensor(gene_df_dose.values, dtype=torch.float64)
    print(f'  DOSE gene topology: {len(dose_genes)} genes from gene_vector.csv')

    # ---- Gene topology for MODEL_GEX (919-gene overlap, header=None) ----
    # Training used header=None giving 978 rows; overlap with lincs parquet = 919
    gene_df_gex = pd.read_csv(GENE_VECTOR_FILE, index_col=0, header=None)
    all_landmark_gex = gene_df_gex.index.tolist()  # 978 genes

    # Get LINCS parquet columns to compute overlap (read schema only)
    import pyarrow.parquet as pq
    lincs_path = str(_PROJECT_ROOT / 'data/processed/lincs_train_safe.parquet')
    lincs_schema = pq.read_schema(lincs_path)
    lincs_cols = set(lincs_schema.names)

    gex_landmark_genes = [g for g in all_landmark_gex if g in lincs_cols]
    print(f'  GEX gene topology: {len(gex_landmark_genes)} genes (landmark/LINCS overlap)')

    if len(gex_landmark_genes) != gex_num_gene:
        print(f'  WARNING: overlap={len(gex_landmark_genes)} != checkpoint num_gene={gex_num_gene}')
        print(f'  Proceeding with {len(gex_landmark_genes)} overlap genes')
        gex_num_gene = len(gex_landmark_genes)

    gex_gene_tensor = torch.tensor(
        gene_df_gex.loc[gex_landmark_genes].values, dtype=torch.float64
    )

    # ---- Cell baselines for MODEL_DOSE (CCLE file, 978-dim) ----
    ccle_df = pd.read_csv(CCLE_FILE, index_col=0)
    ccle_cols = list(ccle_df.columns)
    # MODEL_DOSE was trained with basal_expression_file = CCLE file
    # cell_id feature is raw expression from CCLE at the landmark genes used in training
    # The transform function uses basal_csv.loc[cell_id, :] without additional filtering
    dose_baselines = {}
    for cell in CELLS_10:
        if cell in ccle_df.index:
            vec = ccle_df.loc[cell, :].values.astype(np.float64)
            # CCLE has 978 columns = same as dose_num_gene
            if len(vec) != dose_num_gene:
                print(f'  WARNING: CCLE cell {cell} has {len(vec)} cols, '
                      f'expected {dose_num_gene}. Truncating/padding.')
                if len(vec) > dose_num_gene:
                    vec = vec[:dose_num_gene]
                else:
                    vec = np.pad(vec, (0, dose_num_gene - len(vec)))
            dose_baselines[cell] = torch.tensor(vec, dtype=torch.float64)
        else:
            print(f'  WARNING: cell {cell} not in CCLE file — using zeros')
            dose_baselines[cell] = torch.zeros(dose_num_gene, dtype=torch.float64)

    # ---- Cell baselines for MODEL_GEX (PDG diseased file, gex_num_gene-dim) ----
    diseased_df = pd.read_csv(DISEASED_FILE, index_col=0)
    gex_baselines = {}
    for cell in CELLS_10:
        if cell in diseased_df.index:
            vec = diseased_df.loc[cell, gex_landmark_genes].values.astype(np.float64)
            gex_baselines[cell] = torch.tensor(vec, dtype=torch.float64)
        else:
            print(f'  WARNING: cell {cell} not in diseased baseline — using zeros')
            gex_baselines[cell] = torch.zeros(gex_num_gene, dtype=torch.float64)

    return (dose_gene_tensor, gex_landmark_genes, gex_gene_tensor,
            dose_baselines, gex_baselines)


# ============================================================================
# Per-drug inference helpers
# ============================================================================
@torch.no_grad()
def infer_dose_one_drug_all_cells(model_dose, drug_smiles: str,
                                   dose_gene_tensor: torch.Tensor,
                                   dose_baselines: dict,
                                   device: torch.device,
                                   pert_idose_input_dim: int) -> float:
    """Compute mean E-Hill prediction across all 10 cells for one drug.

    MODEL_DOSE (MultiDCPEhillPretraining) forward:
        (input_drug, input_gene, mask, input_cell_gex, input_pert_idose, job_id)
    Returns: float scalar (mean over cells)
    """
    from data_utils import convert_smile_to_feature, create_mask_feature

    # Convert SMILES to drug graph (batch of 1)
    drug_feat = convert_smile_to_feature([drug_smiles], device)
    mask = create_mask_feature(drug_feat, device)

    # Gene topology: [1, num_gene, 128] (repeat batch dim)
    gene_t = dose_gene_tensor.unsqueeze(0).to(device)  # [1, num_gene, 128]

    # pert_idose: use 6-dim one-hot with all zeros (dose-agnostic at inference)
    # Same convention as AE training: pass zeros since we don't have a specific dose
    pert_idose = torch.zeros(1, pert_idose_input_dim, dtype=torch.float64).to(device)

    dose_preds = []
    for cell in CELLS_10:
        cell_t = dose_baselines[cell].unsqueeze(0).to(device)  # [1, num_gene]

        out, _ = model_dose(
            drug_feat, gene_t, mask, cell_t, pert_idose,
            job_id='pretraining', epoch=0
        )
        # out shape: [1] or scalar
        val = float(out.squeeze().cpu().item())
        dose_preds.append(val)

    return float(np.mean(dose_preds))


@torch.no_grad()
def infer_gex_one_drug_all_cells(model_gex, drug_smiles: str,
                                   gex_gene_tensor: torch.Tensor,
                                   gex_baselines: dict,
                                   device: torch.device,
                                   gex_num_gene: int,
                                   pert_idose_input_dim: int) -> np.ndarray:
    """Compute mean DE GEX vector across all 10 cells for one drug.

    MODEL_GEX (MultiDCP_AE) forward for perturbed path:
        (input_cell_gex, input_drug, input_gene, mask, input_pert_idose, job_id='perturbed')
    DE rule: DE = predicted_perturbed - baseline
    Returns: np.array [gex_num_gene]
    """
    from data_utils import convert_smile_to_feature, create_mask_feature

    # Convert SMILES to drug graph (batch of 1)
    drug_feat = convert_smile_to_feature([drug_smiles], device)
    mask = create_mask_feature(drug_feat, device)

    # Gene topology for GEX model: [1, num_gene, 128]
    gene_t = gex_gene_tensor.unsqueeze(0).to(device)  # [1, gex_num_gene, 128]

    # pert_idose: 6-dim zeros (dose-agnostic)
    pert_idose = torch.zeros(1, pert_idose_input_dim, dtype=torch.float64).to(device)

    gex_de_preds = []
    for cell in CELLS_10:
        cell_t = gex_baselines[cell].unsqueeze(0).to(device)  # [1, gex_num_gene]

        out, _ = model_gex(
            cell_t,                # input_cell_gex
            input_drug=drug_feat,  # input_drug
            input_gene=gene_t,     # input_gene
            mask=mask,             # mask
            input_pert_idose=pert_idose,
            job_id='perturbed',
            epoch=0,
        )
        # out shape: [1, gex_num_gene] (perturbed GEX prediction)
        pred_gex = out.squeeze(0).cpu().numpy()  # [gex_num_gene]
        baseline_np = gex_baselines[cell].numpy()  # [gex_num_gene]

        # DE rule: DE = predicted_perturbed - diseased_baseline
        de = pred_gex - baseline_np
        gex_de_preds.append(de)

    return np.mean(np.stack(gex_de_preds), axis=0)  # [gex_num_gene]


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    start_time = datetime.now()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[cache_dili_features] device={device}')

    # WandB init
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        _wandb_project = args.wandb_project
        _wandb_entity = None
        if '/' in _wandb_project:
            _wandb_entity, _wandb_project = _wandb_project.split('/', 1)
        wandb.init(
            project=_wandb_project, entity=_wandb_entity,
            group=args.wandb_group, name=args.wandb_name,
            config=vars(args),
        )

    # ---- Load checkpoints ----
    print('Loading MODEL_DOSE...')
    model_dose, dose_params = load_dose_model(args.dose_ckpt, device)
    dose_num_gene = int(dose_params.get('num_gene', 978))
    dose_pert_idose_dim = int(dose_params.get('pert_idose_input_dim', 6))
    print(f'  MODEL_DOSE: num_gene={dose_num_gene}, halt_gate_1_pass=', end='')
    _dose_ckpt = torch.load(args.dose_ckpt, map_location='cpu', weights_only=False)
    print(_dose_ckpt.get('halt_gate_1_pass'))

    print('Loading MODEL_GEX...')
    model_gex, gex_params = load_gex_model(args.gex_ckpt, device)
    gex_num_gene = int(gex_params.get('num_gene', 919))
    gex_pert_idose_dim = int(gex_params.get('pert_idose_input_dim', 6))
    _gex_ckpt = torch.load(args.gex_ckpt, map_location='cpu', weights_only=False)
    print(f'  MODEL_GEX: num_gene={gex_num_gene}, halt_gate_2_pass={_gex_ckpt.get("halt_gate_2_pass")}')

    # Sanity gate: both halt gates must have passed
    assert _dose_ckpt.get('halt_gate_1_pass') is True, (
        'MODEL_DOSE halt gate 1 did not pass — abort Stage-2 caching.')
    assert _gex_ckpt.get('halt_gate_2_pass') is True, (
        'MODEL_GEX halt gate 2 did not pass — abort Stage-2 caching.')

    # ---- Build gene topology and cell baselines ----
    print('Building gene topology and cell baselines...')
    (dose_gene_tensor, gex_landmark_genes, gex_gene_tensor,
     dose_baselines, gex_baselines) = build_gene_filter_and_baselines(
        dose_num_gene, gex_num_gene
    )
    # Update gex_num_gene in case overlap differed from checkpoint
    gex_num_gene = len(gex_landmark_genes)
    print(f'  Confirmed: feat_gex dim = {gex_num_gene}')
    print(f'  Total feature dim = 1 (dose) + {gex_num_gene} (gex) + 768 (embed) '
          f'= {1 + gex_num_gene + 768}')

    # ---- Load MolFormer ----
    print('Loading MolFormer...')
    from src.embed.molformer_wrapper import load_molformer
    molformer = load_molformer(device=device)
    print('  MolFormer loaded OK')

    # ---- Load DILIst canonical ----
    dili_df = pd.read_csv(args.dili_csv)
    assert len(dili_df) == 1118, f'Expected 1118 drugs in dili_canonical.csv, got {len(dili_df)}'
    print(f'DILIst drugs: {len(dili_df)} | columns: {list(dili_df.columns)}')

    # ---- Pre-encode all MolFormer embeddings ----
    print(f'Encoding MolFormer embeddings for all {len(dili_df)} drugs...')
    smiles_list = dili_df['canonical_smiles'].tolist()
    molformer_embeddings = molformer.encode(smiles_list, batch_size=args.batch_size)
    # Shape: [1118, 768] on CPU
    assert molformer_embeddings.shape == (len(dili_df), 768), (
        f'MolFormer output shape mismatch: {molformer_embeddings.shape}')
    assert not torch.isnan(molformer_embeddings).any(), 'NaN in MolFormer embeddings'
    print(f'  MolFormer embeddings shape: {molformer_embeddings.shape}')

    # ---- Main feature extraction loop ----
    print(f'Running {len(dili_df)} × 10 cells × 2 models = '
          f'{len(dili_df) * 10 * 2:,} forward passes...')
    rows = []
    for drug_idx, drug_row in dili_df.iterrows():
        drug_name = drug_row['drug_name']
        smiles = drug_row['canonical_smiles']

        # MODEL_DOSE inference: mean E-Hill over 10 cells
        feat_dose = infer_dose_one_drug_all_cells(
            model_dose, smiles, dose_gene_tensor, dose_baselines, device,
            dose_pert_idose_dim,
        )

        # MODEL_GEX inference: mean DE vector over 10 cells
        feat_gex = infer_gex_one_drug_all_cells(
            model_gex, smiles, gex_gene_tensor, gex_baselines, device,
            gex_num_gene, gex_pert_idose_dim,
        )  # [gex_num_gene]

        # MolFormer embedding
        feat_embed = molformer_embeddings[drug_idx].numpy()  # [768]

        # Build output row
        out_row = {
            'drug_name':     drug_name,
            'pert_id':       drug_row['pert_id'],
            'dili_binary':   int(drug_row['dili_binary']),
            'dili_severity': drug_row['dili_severity'],
            'in_lincs':      drug_row['in_lincs'],
            'in_pdg':        drug_row['in_pdg'],
            'feat_dose':     float(feat_dose),
        }
        for j, val in enumerate(feat_gex):
            out_row[f'feat_gex_{j}'] = float(val)
        for j, val in enumerate(feat_embed):
            out_row[f'feat_embed_{j}'] = float(val)
        rows.append(out_row)

        if (drug_idx + 1) % 100 == 0 or drug_idx == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f'  Processed {drug_idx + 1}/{len(dili_df)} drugs '
                  f'({elapsed:.0f}s elapsed)')
            if use_wandb:
                import wandb
                wandb.log({'progress/drugs_done': drug_idx + 1,
                           'progress/elapsed_sec': elapsed})

    # ---- Write parquet ----
    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(str(out_path), index=False)
    print(f'Wrote {out_path}: shape {out_df.shape}')

    # ---- Final stats ----
    feat_cols   = [c for c in out_df.columns if c.startswith('feat_')]
    gex_cols    = [c for c in feat_cols if 'gex' in c]
    embed_cols  = [c for c in feat_cols if 'embed' in c]
    nan_count   = int(out_df.isnull().sum().sum())
    print(f'=== Output stats ===')
    print(f'  Rows: {len(out_df)}')
    print(f'  feat_dose:  1 col')
    print(f'  feat_gex:   {len(gex_cols)} cols')
    print(f'  feat_embed: {len(embed_cols)} cols')
    print(f'  Total feat: {len(feat_cols)} cols')
    print(f'  NaN count:  {nan_count}')
    print(f'  dili_binary: {dict(out_df["dili_binary"].value_counts())}')

    if nan_count > 0:
        print('FAIL: NaN values found in feature parquet — Stage-2 caching invalid.')

    elapsed_total = (datetime.now() - start_time).total_seconds()
    print(f'Total time: {elapsed_total:.1f}s')

    if use_wandb:
        import wandb
        wandb.log({
            'output/n_drugs':      len(out_df),
            'output/n_cols':       len(out_df.columns),
            'output/n_nan':        nan_count,
            'output/feat_dose_mean': float(out_df['feat_dose'].mean()),
            'output/feat_dose_std':  float(out_df['feat_dose'].std()),
            'output/elapsed_sec':    elapsed_total,
        })
        wandb.finish()

    print('Phase 3 Plan 1 — cache_dili_features.py complete.')
    return nan_count == 0


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
