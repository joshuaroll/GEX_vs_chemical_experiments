#!/usr/bin/env python3
"""
Final dev-set evaluation for MODEL_GEX (Phase 2).
Computes per-cell DE Pearson on top-80 DE genes for halt gate 2 check.

Uses the AE reconstruction path (job_id='ae') since the LINCS parquet has
dose-aggregated signatures (pert_idose='x') and no drug SMILES in the parquet.

Usage:
    conda run -n dili_v04_env python scripts/eval_model_gex.py \
        --checkpoint results/checkpoints/chkpt_gex.pt \
        --safe_parquet data/processed/lincs_train_safe.parquet \
        --gpu 1 \
        --seed 343 \
        --top_k_de 80

MultiDCP SHA: 871b8de (/raid/home/joshua/projects/MultiDCP/)
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

_MULTIDCP_ROOT = '/raid/home/joshua/projects/MultiDCP/MultiDCP'
sys.path.insert(0, os.path.join(_MULTIDCP_ROOT, 'models'))
sys.path.insert(0, os.path.join(_MULTIDCP_ROOT, 'utils'))

import torch
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from src.eval.de_evaluator import compute_percell_pearson

import multidcp_balanceloss as multidcp
from multidcp_ae_utils import initialize_model_registry

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"[eval_model_gex] device={device}")


def main(args):
    # Re-derive dev split with same seed as training
    print(f"[eval_model_gex] Loading parquet: {args.safe_parquet}")
    df_full = pd.read_parquet(args.safe_parquet)
    print(f"  Full parquet: {len(df_full)} rows")

    idx = np.arange(len(df_full))
    idx_train, idx_temp = train_test_split(idx, test_size=0.15, random_state=args.seed)
    dev_share = 0.10 / 0.15
    idx_dev, _ = train_test_split(idx_temp, test_size=(1 - dev_share), random_state=args.seed)
    df_dev = df_full.iloc[idx_dev].reset_index(drop=True)
    print(f'Dev set: {len(df_dev)} rows')

    # Identify landmark gene columns (919 present out of 978 from gene_vector.csv)
    _gene_vec = pd.read_csv(
        '/raid/home/joshua/projects/MultiDCP/MultiDCP/data/gene_vector.csv',
        index_col=0, header=None
    )
    landmark_genes = list(_gene_vec.index)
    meta_cols = ['sig_id', 'idx', 'pert_id', 'pert_type', 'cell_id', 'pert_idose']
    gene_cols = [g for g in landmark_genes if g in df_dev.columns]
    num_landmark_genes = len(gene_cols)
    print(f'Landmark genes available: {num_landmark_genes}/978')
    assert num_landmark_genes >= 900, f'Too few landmark genes: {num_landmark_genes}'

    # Load model with SAME architecture as training
    model_param_registry = initialize_model_registry()
    model_param_registry.update({
        'num_gene': num_landmark_genes,
        'cell_id_input_dim': num_landmark_genes,
        'cell_decoder_dim': num_landmark_genes,
        'pert_idose_input_dim': 6,
        'dropout': 0.0,             # eval mode — no dropout
        'linear_encoder_flag': False,
        'fusion_type': 'sparse_moe',  # must match training
    })
    model = multidcp.MultiDCP_AE(device=device, model_param_registry=model_param_registry)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model = model.double()
    print(f'Checkpoint loaded: {args.checkpoint} ({len(state_dict)} param tensors)')

    # Run inference on dev set grouped by cell
    # LINCS signatures are already differential expression — use directly as true_de
    true_de_by_cell = {}
    pred_de_by_cell = {}

    cell_ids_unique = sorted(df_dev['cell_id'].unique().tolist())
    print(f'Cell lines in dev set ({len(cell_ids_unique)}): {cell_ids_unique}')

    for cell_id in cell_ids_unique:
        df_cell = df_dev[df_dev['cell_id'] == cell_id]
        gene_vals = df_cell[gene_cols].values.astype(float)  # [n_cell, n_landmark_genes]
        true_de_by_cell[cell_id] = gene_vals

        # AE path inference: input GEX -> reconstructed GEX
        gene_tensor = torch.tensor(gene_vals, dtype=torch.float64).to(device)

        preds = []
        bs = 128
        with torch.no_grad():
            for start in range(0, len(gene_tensor), bs):
                batch = gene_tensor[start:start + bs]
                pred, _ = model(input_cell_gex=batch, job_id='ae')
                preds.append(pred.cpu().numpy())
        pred_de_by_cell[cell_id] = np.concatenate(preds, axis=0)
        print(f'  {cell_id}: {len(df_cell)} samples processed')

    # Compute halt gate 2 metric
    results = compute_percell_pearson(true_de_by_cell, pred_de_by_cell, k=args.top_k_de)

    print('\n' + '=' * 60)
    print('HALT GATE 2 (HG2) EVALUATION RESULTS')
    print('=' * 60)
    print(f'Per-cell Pearson (top-{args.top_k_de} DE genes):')
    for cell_id_key, pearson in sorted(results['per_cell'].items()):
        print(f'  {cell_id_key}: {pearson:.4f}')
    print(f'Mean Pearson (HG2 metric): {results["mean_pearson"]:.4f}')
    print(f'Threshold: 0.2')
    print(f'HG2 Status: {"PASS" if results["halt_gate_2_pass"] else "FAIL"}')
    print('=' * 60)

    # Write JSON results
    results_out = {
        'per_cell': results['per_cell'],
        'mean_pearson': results['mean_pearson'],
        'halt_gate_2_pass': results['halt_gate_2_pass'],
        'checkpoint': args.checkpoint,
        'top_k_de': args.top_k_de,
        'n_dev_samples': len(df_dev),
        'n_cells': len(cell_ids_unique),
        'cell_ids': cell_ids_unique,
        'num_landmark_genes': num_landmark_genes,
        'seed': args.seed,
    }
    json_path = 'results/tables/P2_hg2_results.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f'Results written to {json_path}')

    return results_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval_model_gex — MODEL_GEX HG2 evaluation')
    parser.add_argument('--checkpoint', default='results/checkpoints/chkpt_gex.pt')
    parser.add_argument('--safe_parquet', default='data/processed/lincs_train_safe.parquet')
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--seed', type=int, default=343)
    parser.add_argument('--top_k_de', type=int, default=80)
    args = parser.parse_args()
    main(args)
