"""
DE (Differential Expression) evaluation utilities for multihead_dili Phase 2.

PROVENANCE: Functions vendored from
  /raid/home/joshua/projects/PDGrapher_Baseline_Models/Biolord/pdgrapher_experiments/train_bl_pdg_de.py
  (commit as of 2026-05-20; do not modify logic without updating that canonical source)

DE RULE (LOAD-BEARING, project hard rule 2):
  All metrics computed on DIFFERENTIAL EXPRESSION = treated - baseline.
  Top-k genes selected by |true_de| per sample.
  Raw expression as a feature or metric is FORBIDDEN.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_topk_by_differential_expression(true_de, k=20):
    """
    Select top-k differentially expressed genes per sample.

    Args:
        true_de: True differential expression (treated - diseased), shape [n_samples, n_genes]
        k: Number of top genes to select

    Returns:
        top_indices: Shape [n_samples, k], indices of top-k DEGs per sample
    """
    # Sort by absolute differential expression, descending
    top_indices = np.argsort(np.abs(true_de), axis=1)[:, ::-1][:, :k]
    return top_indices


def compute_global_metrics(true_de, pred_de):
    """Compute metrics over ALL values (flattened across samples and genes)."""
    true_flat = true_de.ravel()
    pred_flat = pred_de.ravel()

    mask = np.isfinite(true_flat) & np.isfinite(pred_flat)
    true_masked = true_flat[mask]
    pred_masked = pred_flat[mask]

    if len(true_masked) < 2:
        return {'pearson': np.nan, 'spearman': np.nan, 'r2': np.nan,
                'r2_from_pearson': np.nan, 'rmse': np.nan, 'mae': np.nan, 'n_values': 0}

    pearson_r = pearsonr(true_masked, pred_masked).statistic
    spearman_r = spearmanr(true_masked, pred_masked).correlation

    ss_res = np.sum((true_masked - pred_masked) ** 2)
    ss_tot = np.sum((true_masked - np.mean(true_masked)) ** 2)
    true_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    rmse = np.sqrt(np.mean((true_masked - pred_masked) ** 2))
    mae = np.mean(np.abs(true_masked - pred_masked))

    return {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'r2': true_r2,
        'r2_from_pearson': pearson_r ** 2,
        'rmse': rmse,
        'mae': mae,
        'n_values': len(true_masked)
    }


def compute_persample_metrics(true_de, pred_de, gene_indices=None, show_progress=True):
    """
    Compute metrics per sample on DIFFERENTIAL EXPRESSION, then aggregate.

    Args:
        true_de: True differential expression (treated - diseased)
        pred_de: Predicted differential expression (pred_treated - diseased)
        gene_indices: If provided, compute metrics only on these genes per sample
        show_progress: Whether to print progress messages
    """
    n_samples = true_de.shape[0]

    if show_progress:
        print(f"    Processing {n_samples} samples...")

    # If using specific gene indices, extract them
    if gene_indices is not None:
        true_selected = np.array([true_de[i, gene_indices[i]] for i in range(n_samples)])
        pred_selected = np.array([pred_de[i, gene_indices[i]] for i in range(n_samples)])
    else:
        true_selected = true_de
        pred_selected = pred_de

    # Compute statistics per sample (vectorized)
    true_mean = np.mean(true_selected, axis=1, keepdims=True)
    pred_mean = np.mean(pred_selected, axis=1, keepdims=True)

    true_centered = true_selected - true_mean
    pred_centered = pred_selected - pred_mean

    true_std = np.std(true_selected, axis=1)
    pred_std = np.std(pred_selected, axis=1)

    var_true = true_std ** 2
    var_pred = pred_std ** 2

    n_genes_per_sample = true_selected.shape[1]

    # Avoid division by zero
    valid_mask = (true_std > 1e-10) & (pred_std > 1e-10)

    # Pearson correlation (vectorized)
    covariance = np.sum(true_centered * pred_centered, axis=1) / n_genes_per_sample

    pearson_vals = np.full(n_samples, np.nan)
    pearson_vals[valid_mask] = covariance[valid_mask] / (true_std[valid_mask] * pred_std[valid_mask])

    r2_pearson_vals = pearson_vals ** 2

    # True R² (coefficient of determination)
    ss_res = np.sum((true_selected - pred_selected) ** 2, axis=1)
    ss_tot = np.sum(true_centered ** 2, axis=1)

    r2_true_vals = np.full(n_samples, np.nan)
    valid_ss = ss_tot > 1e-10
    r2_true_vals[valid_ss] = 1 - (ss_res[valid_ss] / ss_tot[valid_ss])

    # RMSE per sample
    rmse_vals = np.sqrt(np.mean((true_selected - pred_selected) ** 2, axis=1))

    # Spearman correlation
    spearman_vals = np.full(n_samples, np.nan)

    sample_indices = range(n_samples) if n_samples <= 5000 else np.random.choice(n_samples, 2000, replace=False)
    for i in sample_indices:
        if valid_mask[i]:
            try:
                rho = spearmanr(true_selected[i], pred_selected[i]).correlation
                if np.isfinite(rho):
                    spearman_vals[i] = rho
            except Exception:
                pass

    if show_progress:
        print(f"    Done. Valid samples: {np.sum(valid_mask)}/{n_samples}")

    return {
        'pearson_mean': np.nanmean(pearson_vals),
        'pearson_median': np.nanmedian(pearson_vals),
        'pearson_std': np.nanstd(pearson_vals),
        'spearman_mean': np.nanmean(spearman_vals),
        'r2_true_mean': np.nanmean(r2_true_vals),
        'r2_from_pearson_mean': np.nanmean(r2_pearson_vals),
        'rmse_mean': np.nanmean(rmse_vals),
    }


def compute_prediction_bias(true_de, pred_de):
    """Check for systematic prediction biases in differential expression."""
    errors = pred_de - true_de
    true_flat = true_de.ravel()
    error_flat = errors.ravel()

    mask = np.isfinite(true_flat) & np.isfinite(error_flat)
    true_masked = true_flat[mask]
    error_masked = error_flat[mask]

    if len(true_masked) < 10:
        return {'mean_error': np.nan, 'high_value_bias': np.nan, 'low_value_bias': np.nan}

    high_true = true_masked > np.percentile(true_masked, 90)
    low_true = true_masked < np.percentile(true_masked, 10)

    high_bias = np.mean(error_masked[high_true]) if np.sum(high_true) > 0 else np.nan
    low_bias = np.mean(error_masked[low_true]) if np.sum(low_true) > 0 else np.nan

    return {
        'mean_error': np.mean(error_masked),
        'std_error': np.std(error_masked),
        'high_value_bias': high_bias,
        'low_value_bias': low_bias,
    }


def compute_percell_pearson(true_de_by_cell: dict, pred_de_by_cell: dict, k: int = 80) -> dict:
    """
    Compute per-cell Pearson on top-k DE genes, then return cell->pearson dict + mean.

    Args:
        true_de_by_cell: dict mapping cell_id -> np.array [n_samples_cell, 978]
        pred_de_by_cell: dict mapping cell_id -> np.array [n_samples_cell, 978]
        k: top-k DE genes to evaluate on (default 80)
    Returns:
        dict with keys: per_cell (dict cell_id->float), mean_pearson (float), halt_gate_2_pass (bool)
    """
    per_cell = {}
    for cell_id in true_de_by_cell:
        true_de = true_de_by_cell[cell_id]
        pred_de = pred_de_by_cell[cell_id]
        top_k_idx = compute_topk_by_differential_expression(true_de, k=k)
        metrics = compute_persample_metrics(true_de, pred_de, gene_indices=top_k_idx, show_progress=False)
        per_cell[cell_id] = metrics['pearson_mean']
    mean_pearson = float(np.nanmean(list(per_cell.values())))
    return {
        'per_cell': per_cell,
        'mean_pearson': mean_pearson,
        'halt_gate_2_pass': mean_pearson >= 0.2
    }
