#!/usr/bin/env python3
"""
MODEL_GEX training script — MultiDCP-AE fork for LINCS leakage-filtered data.

Phase 2 of multihead_dili. Trains on data/processed/lincs_train_safe.parquet.
Canonical DE-rule evaluator: src/eval/de_evaluator.py (vendored from train_bl_pdg_de.py).

MultiDCP SHA: 871b8de (/raid/home/joshua/projects/MultiDCP/)

Architecture: MultiDCP_AE trained in AE mode on LINCS landmark gene expression.
The LINCS parquet has 10,716 gene columns (full L1000 space) plus metadata;
we filter to 978 landmark genes via gene_vector.csv before DataLoader creation.
pert_idose values are all 'x' (dose-aggregated) — we use AE path only (no dose encoding).
"""
import os
import sys
import argparse

# =============================================================================
# SET CUDA_VISIBLE_DEVICES BEFORE IMPORTING TORCH (CLAUDE.md hard rule 5)
# =============================================================================
def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

_gpu = _get_gpu_arg()
if _gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu
    print(f"[train_model_gex] CUDA_VISIBLE_DEVICES={_gpu}")

# Path setup for upstream MultiDCP modules (SHA-pinned: 871b8de)
_MULTIDCP_ROOT = '/raid/home/joshua/projects/MultiDCP/MultiDCP'
sys.path.insert(0, os.path.join(_MULTIDCP_ROOT, 'models'))
sys.path.insert(0, os.path.join(_MULTIDCP_ROOT, 'utils'))

# NOW import torch and everything else
import pandas as pd
import numpy as np
import torch
import random
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import train_test_split
import wandb
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset, DataLoader

import multidcp_balanceloss as multidcp
import metric
from multidcp_ae_utils import initialize_model_registry, print_lr, validation_epoch_end, test_epoch_end, report_final_results

# Project eval module (DE rule)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
from src.eval.de_evaluator import compute_percell_pearson, compute_topk_by_differential_expression

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"[train_model_gex] device={device}")


# =============================================================================
# CUSTOM DATASET: reads pre-split landmark-gene-filtered numpy arrays
# =============================================================================
class LINCSGEXDataset(Dataset):
    """Dataset for LINCS dose-aggregated GEX (AE training path)."""

    def __init__(self, gene_vals: np.ndarray, cell_ids: list, device):
        self.device = device
        # gene_vals: [n, n_landmark_genes] float64
        self.data = torch.from_numpy(gene_vals.astype(np.float64)).to(device)
        self.cell_ids = cell_ids  # list of str cell IDs, length n

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns (feature, label, cell_id_str) — AE: feature == label
        return self.data[idx], self.data[idx], self.cell_ids[idx]


# =============================================================================
# TRAINING LOOP (AE-only, no drug/dose inputs)
# =============================================================================
def model_training_ae(args, model, train_loader, dev_loader, dev_cell_ids, metrics_summary):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    best_dev_pearson = float("-inf")

    for epoch in range(args.max_epoch):
        print(f"Iteration {epoch}:")
        print_lr(optimizer)
        model.train()
        epoch_loss = 0

        for i, (feature, label, _) in enumerate(train_loader):
            optimizer.zero_grad()
            predict, cell_hidden_ = model(input_cell_gex=feature, job_id='ae', epoch=epoch)
            loss_t = model.loss(label, predict)
            loss_t.backward()
            optimizer.step()
            epoch_loss += loss_t.item()

        print('AE Train loss:')
        print(epoch_loss / (i + 1))
        wandb.log({'AE Train loss': epoch_loss / (i + 1)}, step=epoch)

        # ---- Dev evaluation ----
        model.eval()
        epoch_loss_dev = 0
        lb_np = np.empty([0, args.num_landmark_genes])
        predict_np = np.empty([0, args.num_landmark_genes])
        cell_ids_dev = []

        with torch.no_grad():
            for i, (feature, label, cell_id_batch) in enumerate(dev_loader):
                predict, _ = model(input_cell_gex=feature, job_id='ae', epoch=epoch)
                loss = model.loss(label, predict)
                epoch_loss_dev += loss.item()
                lb_np = np.concatenate((lb_np, label.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
                cell_ids_dev.extend(list(cell_id_batch))

        print('AE Dev loss:', epoch_loss_dev / (i + 1))
        wandb.log({'AE Dev loss': epoch_loss_dev / (i + 1)}, step=epoch)

        # Standard metrics (Pearson, RMSE, etc.)
        import metric as _metric
        pearson, _ = _metric.correlation(lb_np, predict_np, 'pearson')
        rmse_val = _metric.rmse(lb_np, predict_np)
        metrics_summary['pearson_list_ae_dev'].append(pearson)
        metrics_summary['rmse_list_ae_dev'].append(rmse_val)
        print(f'AE Dev Pearson: {pearson:.4f} | RMSE: {rmse_val:.4f}')
        wandb.log({'AE Dev Pearson': pearson, 'AE Dev RMSE': rmse_val}, step=epoch)

        # ---- DE-rule per-cell Pearson (Halt Gate 2 metric) ----
        true_de_by_cell = {}
        pred_de_by_cell = {}
        for cell_id in set(cell_ids_dev):
            mask_c = np.array(cell_ids_dev) == cell_id
            true_de_by_cell[cell_id] = lb_np[mask_c]
            pred_de_by_cell[cell_id] = predict_np[mask_c]

        de_results = compute_percell_pearson(true_de_by_cell, pred_de_by_cell, k=args.top_k_de)
        wandb.log({
            'dev/perturbed_pearson_mean': de_results['mean_pearson'],
            'dev/halt_gate_2_pass': int(de_results['halt_gate_2_pass']),
            **{f"dev/pearson_{cid}": v for cid, v in de_results['per_cell'].items()},
        }, step=epoch)
        print(f"  [HG2] Mean DE Pearson (top-{args.top_k_de}): {de_results['mean_pearson']:.4f} "
              f"({'PASS' if de_results['halt_gate_2_pass'] else 'FAIL'} >= 0.2)")

        # Save best checkpoint based on DE Pearson (HG2 metric)
        if de_results['mean_pearson'] > best_dev_pearson:
            best_dev_pearson = de_results['mean_pearson']
            os.makedirs(os.path.dirname(os.path.abspath(args.saved_model_name)), exist_ok=True)
            torch.save(model.state_dict(), args.saved_model_name)
            print(f"  [checkpoint] Saved best model at epoch {epoch} (mean_pearson={best_dev_pearson:.4f})")

    print(f"\n[train_model_gex] Training complete. Best dev DE Pearson: {best_dev_pearson:.4f}")
    return best_dev_pearson


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='MODEL_GEX MultiDCP-AE training on LINCS')

    # Original MultiDCP args (kept for compatibility with datareader/utils)
    parser.add_argument('--drug_file', default='/raid/home/joshua/projects/MultiDCP/MultiDCP/data/all_drugs_l1000.csv')
    parser.add_argument('--gene_file', default='/raid/home/joshua/projects/MultiDCP/MultiDCP/data/gene_vector.csv')
    parser.add_argument('--train_file', default=None)
    parser.add_argument('--dev_file', default=None)
    parser.add_argument('--test_file', default=None)
    parser.add_argument('--ae_input_file', default='/raid/home/joshua/projects/MultiDCP/MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4')
    parser.add_argument('--ae_label_file', default='/raid/home/joshua/projects/MultiDCP/MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4')
    parser.add_argument('--cell_ge_file', default='/raid/home/joshua/projects/MultiDCP/MultiDCP/data/adjusted_ccle_tcga_ad_tpm_log2.csv')
    parser.add_argument('--all_cells', default='/raid/home/joshua/projects/MultiDCP/MultiDCP/data/ccle_tcga_ad_cells.p')
    parser.add_argument('--linear_encoder_flag', dest='linear_encoder_flag', action='store_true', default=False)
    parser.add_argument('--fusion_type', type=str, default='concat')
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Phase 2 additions
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device index to use (sets CUDA_VISIBLE_DEVICES before torch import)')
    parser.add_argument('--safe_parquet', type=str, default=None,
                        help='Path to leakage-filtered LINCS parquet (lincs_train_safe.parquet). '
                             'When set, loads data from parquet instead of CSV files.')
    parser.add_argument('--dev_frac', type=float, default=0.10,
                        help='Fraction of safe_parquet to hold out as dev set (default 0.10)')
    parser.add_argument('--test_frac', type=float, default=0.05,
                        help='Fraction of safe_parquet to hold out as test set (default 0.05)')
    parser.add_argument('--top_k_de', type=int, default=80,
                        help='Top-k DE genes for halt-gate Pearson evaluation (default 80)')
    parser.add_argument('--saved_model_name', type=str,
                        default='results/checkpoints/chkpt_gex.pt',
                        help='Path to save best checkpoint')
    parser.add_argument('--wandb_project', type=str, default='joshroll/MultiDCP_multihead_dili')
    parser.add_argument('--wandb_group', type=str, default='model_gex')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader num_workers (cap at 4-6, shared CPU box)')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=343)
    parser.add_argument('--dropout', type=float, default=0.3)

    args = parser.parse_args()

    # Reproducibility
    seed = args.seed
    np.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ---- Load parquet and derive landmark-filtered splits ----
    if args.safe_parquet:
        print(f"[train_model_gex] Loading safe parquet: {args.safe_parquet}")
        df_full = pd.read_parquet(args.safe_parquet)
        print(f"  Full parquet shape: {df_full.shape}")

        # LANDMARK GENE FILTER: parquet has 10,716 gene columns; MultiDCP expects 978
        _gene_vec = pd.read_csv(args.gene_file, index_col=0, header=None)
        landmark_genes = list(_gene_vec.index)  # 978 gene symbols
        meta_cols = ['sig_id', 'idx', 'pert_id', 'pert_type', 'cell_id', 'pert_idose']
        avail_genes = [g for g in landmark_genes if g in df_full.columns]
        print(f"  Landmark genes in parquet: {len(avail_genes)}/978")
        if len(avail_genes) < 900:
            raise ValueError(f"Only {len(avail_genes)} landmark genes found in parquet — check gene names")

        # TRACKING: Note HA1E as 10th cell (design doc said 9, plan confirms 10)
        cell_ids_all = sorted(df_full['cell_id'].unique().tolist())
        print(f"  Cell lines in parquet ({len(cell_ids_all)}): {cell_ids_all}")
        if 'HA1E' in cell_ids_all:
            print("  [TRACKING] HA1E present as 10th cell — Stage-2 inference (Phase 3) must account for this")

        # Split: 85% train, 10% dev, 5% test
        idx = np.arange(len(df_full))
        idx_train, idx_temp = train_test_split(idx, test_size=(args.dev_frac + args.test_frac),
                                               random_state=args.seed)
        dev_share = args.dev_frac / (args.dev_frac + args.test_frac)
        idx_dev, idx_test = train_test_split(idx_temp, test_size=(1 - dev_share),
                                             random_state=args.seed)

        df_train = df_full.iloc[idx_train].reset_index(drop=True)
        df_dev   = df_full.iloc[idx_dev].reset_index(drop=True)
        df_test  = df_full.iloc[idx_test].reset_index(drop=True)
        print(f"  Split: train={len(df_train)}, dev={len(df_dev)}, test={len(df_test)}")

        # Extract numpy arrays (landmark genes only)
        train_vals = df_train[avail_genes].values.astype(np.float64)
        dev_vals   = df_dev[avail_genes].values.astype(np.float64)
        test_vals  = df_test[avail_genes].values.astype(np.float64)
        train_cell_ids = df_train['cell_id'].tolist()
        dev_cell_ids   = df_dev['cell_id'].tolist()
        test_cell_ids  = df_test['cell_id'].tolist()

        args.num_landmark_genes = len(avail_genes)
        print(f"  num_landmark_genes: {args.num_landmark_genes}")

        # Build DataLoaders
        train_dataset = LINCSGEXDataset(train_vals, train_cell_ids, device)
        dev_dataset   = LINCSGEXDataset(dev_vals, dev_cell_ids, device)
        test_dataset  = LINCSGEXDataset(test_vals, test_cell_ids, device)

        # Custom collate to handle string cell_ids (can't stack strings with default collate)
        def _collate(batch):
            features = torch.stack([x[0] for x in batch])
            labels   = torch.stack([x[1] for x in batch])
            cell_ids = [x[2] for x in batch]
            return features, labels, cell_ids

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=_collate)
        dev_loader   = DataLoader(dev_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, collate_fn=_collate)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, collate_fn=_collate)

        print(f'#Train: {len(train_dataset)}')
        print(f'#Dev: {len(dev_dataset)}')
        print(f'#Test: {len(test_dataset)}')

    else:
        raise ValueError("--safe_parquet is required for MODEL_GEX training. "
                         "Please provide the path to lincs_train_safe.parquet.")

    # ---- Model initialization ----
    model_param_registry = initialize_model_registry()
    model_param_registry.update({
        'num_gene': args.num_landmark_genes,
        'cell_id_input_dim': args.num_landmark_genes,   # AE input dim = gene count
        'cell_decoder_dim': args.num_landmark_genes,     # AE output dim = gene count
        'pert_idose_input_dim': 6,  # kept at 6 for model compat (not used in AE path)
        'dropout': args.dropout,
        'linear_encoder_flag': args.linear_encoder_flag,
        'fusion_type': args.fusion_type,
    })

    print(f'--------------with linear encoder: {args.linear_encoder_flag!r}--------------')
    model = multidcp.MultiDCP_AE(device=device, model_param_registry=model_param_registry)
    model.init_weights(pretrained=args.pretrained_model)
    model.to(device)
    model = model.double()

    # ---- WandB init ----
    run_name = f"model_gex_seed{args.seed}_bs{args.batch_size}_ep{args.max_epoch}"
    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config=vars(args),
        reinit=True,
    )
    wandb.watch(model, log="all")
    print(f"WandB initialized: project={args.wandb_project}, group={args.wandb_group}, run={run_name}")

    # ---- Training ----
    metrics_summary = defaultdict(list)
    os.makedirs(os.path.dirname(os.path.abspath(args.saved_model_name)), exist_ok=True)

    best_pearson = model_training_ae(args, model, train_loader, dev_loader, dev_cell_ids, metrics_summary)

    # ---- Final checkpoint save ----
    if not os.path.exists(args.saved_model_name):
        # If best checkpoint was never saved (degenerate case), save final model state
        torch.save(model.state_dict(), args.saved_model_name)
        print(f"[train_model_gex] Saved final model state to {args.saved_model_name}")

    end_time = datetime.now()
    print(f"\n[train_model_gex] Total time: {end_time - start_time}")
    wandb.finish()
