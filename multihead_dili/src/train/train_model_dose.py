"""train_model_dose.py — MODEL_DOSE Stage-1 training.

Fork of MultiDCP/ehill_multidcp_pretrain.py (SHA 871b8de) with:
  - CUDA hygiene: --gpu parsed before torch import
  - --safe-parquet flag: routes to leakage-filtered parquet files
  - WandB project/group args
  - Early stopping on dev Pearson
  - Halt gate 1 computation (dev RMSE < std(dev_labels) ?)
  - Rich checkpoint dict with halt_gate_1_pass and multidcp_sha

Hard Rule 4 (CUDA hygiene): os.environ['CUDA_VISIBLE_DEVICES'] is set BEFORE `import torch`.
"""

import argparse
import os
import sys
import types

# ---- Step 0: CUDA hygiene — parse --gpu BEFORE torch import ----------------
def _parse_gpu_early():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--gpu', type=int, default=0)
    args, _ = p.parse_known_args()
    return args.gpu

_GPU = _parse_gpu_early()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_GPU)

# ---- Step 1: pkg_resources shim (pytorch-lightning uses it at import time) --
# Modern setuptools (>= 60) dropped the top-level pkg_resources re-export.
# pytorch-lightning's lightning_fabric calls pkg_resources.declare_namespace()
# which fails on Python 3.12 unless we provide a minimal shim.
if 'pkg_resources' not in sys.modules:
    _pkg_shim = types.ModuleType('pkg_resources')
    _pkg_shim.declare_namespace = lambda *a, **kw: None
    _pkg_shim.require = lambda *a, **kw: []
    sys.modules['pkg_resources'] = _pkg_shim

# ---- Step 2: safe to import torch and everything else ----------------------
import torch
from torch import save
import numpy as np
import pandas as pd
import pickle
import wandb
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# MultiDCP paths
_MULTIDCP_MODELS = '/raid/home/joshua/projects/MultiDCP/MultiDCP/models'
_MULTIDCP_UTILS  = '/raid/home/joshua/projects/MultiDCP/MultiDCP/utils'
sys.path.insert(0, _MULTIDCP_MODELS)
sys.path.insert(0, _MULTIDCP_UTILS)

import multidcp
import datareader
import metric
import data_utils
from scheduler_lr import step_lr

MULTIDCP_SHA = '871b8de'

# ---- Patch data_utils.transform_to_tensor_per_dataset_ehill ----------------
# The upstream function uses ft[1] for cell_id, but the high_confident_data*.csv
# format has columns [pert_id, pert_type, cell_id, pert_idose] (not [pert_id, cell_id, dose]).
# ft[1] = pert_type (e.g. 'trt_cp'), ft[2] = cell_id, ft[3] = pert_idose.
# This patch uses the correct column indices for the actual data format.

def _patched_transform_ehill(feature, label, drug, device, basal_expression_file):
    import pandas as _pd
    import numpy as _np
    if not basal_expression_file.endswith('csv'):
        basal_expression_file += '.csv'
    basal_csv = _pd.read_csv(basal_expression_file, index_col=0)
    drug_feature = []

    # feature columns from read_data: [pert_id(0), pert_type(1), cell_id(2), pert_idose(3)]
    cell_id_set   = sorted(list(set(feature[:, 2])))   # ft[2] = cell_id
    pert_idose_set = sorted(list(set(feature[:, 3])))  # ft[3] = pert_idose
    use_pert_type  = False
    use_cell_id    = True
    use_pert_idose = len(pert_idose_set) > 1

    pert_idose_dict = dict(zip(pert_idose_set, range(len(pert_idose_set)))) if use_pert_idose else {}

    print(f'Feature Summary (patched): {len(cell_id_set)} cell lines, {len(pert_idose_set)} doses')

    final_cell_id_feature   = []
    final_pert_idose_feature = []

    for ft in feature:
        drug_fp = drug[ft[0]]
        drug_feature.append(drug_fp)
        # cell_id is at ft[2]
        cell_id_feature = basal_csv.loc[ft[2], :]
        final_cell_id_feature.append(_np.array(cell_id_feature, dtype=_np.float64))
        if use_pert_idose:
            pert_idose_feature = _np.zeros(len(pert_idose_set))
            pert_idose_feature[pert_idose_dict[ft[3]]] = 1
            final_pert_idose_feature.append(_np.array(pert_idose_feature, dtype=_np.float64))

    feature_dict = {'drug': _np.asarray(drug_feature)}
    feature_dict['cell_id'] = torch.from_numpy(
        _np.asarray(final_cell_id_feature, dtype=_np.float64)).to(device)
    if use_pert_idose:
        feature_dict['pert_idose'] = torch.from_numpy(
            _np.asarray(final_pert_idose_feature, dtype=_np.float64)).to(device)
    label_regression = torch.from_numpy(label).to(device)
    return feature_dict, label_regression, use_pert_type, use_cell_id, use_pert_idose

data_utils.transform_to_tensor_per_dataset_ehill = _patched_transform_ehill
PRECISION_DEGREE = [10, 20, 50, 100]


# ---- Device setup (after torch import) -------------------------------------
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'GPU device: CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}, '
      f'cuda available: {torch.cuda.is_available()}')


# ---- Utilities -------------------------------------------------------------

def initialize_model_registry():
    model_param_registry = defaultdict(
        drug_input_dim={'atom': 62, 'bond': 6},
        drug_emb_dim=128,
        conv_size=[16, 16],
        degree=[0, 1, 2, 3, 4, 5],
        gene_emb_dim=128,
        gene_input_dim=128,
        cell_id_input_dim=978,
        cell_feature_emb_dim=32,
        pert_idose_emb_dim=4,
        hid_dim=128,
        num_gene=978,
        loss_type='point_wise_mse',
        initializer=torch.nn.init.kaiming_uniform_
    )
    return model_param_registry


def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f'============current learning rate is {param_group["lr"]!r}')


def validation_epoch_end(epoch_loss_ehill, lb_np, predict_np,
                          steps_per_epoch, epoch, metrics_summary, use_wandb):
    print('Dev ehill loss:')
    print(epoch_loss_ehill / steps_per_epoch)
    if use_wandb:
        wandb.log({'Dev ehill loss': epoch_loss_ehill / steps_per_epoch}, step=epoch)

    rmse_ehill = metric.rmse(lb_np, predict_np)
    metrics_summary['rmse_list_dev_ehill'].append(rmse_ehill)
    print('RMSE ehill: %.4f' % rmse_ehill)
    if use_wandb:
        wandb.log({'Dev ehill RMSE': rmse_ehill}, step=epoch)

    pearson_ehill, _ = metric.correlation(lb_np, predict_np, 'pearson')
    metrics_summary['pearson_ehill_list_dev'].append(pearson_ehill)
    print("Pearson_ehill's correlation: %.4f" % pearson_ehill)
    if use_wandb:
        wandb.log({'Dev Pearson_ehill': pearson_ehill}, step=epoch)

    spearman_ehill, _ = metric.correlation(lb_np, predict_np, 'spearman')
    metrics_summary['spearman_ehill_list_dev'].append(spearman_ehill)
    print("Spearman_ehill's correlation: %.4f" % spearman_ehill)
    if use_wandb:
        wandb.log({'Dev Spearman_ehill': spearman_ehill}, step=epoch)


def test_epoch_end(epoch_loss_ehill, lb_np, predict_np,
                   steps_per_epoch, epoch, metrics_summary, use_wandb):
    print('Test ehill loss:')
    print(epoch_loss_ehill / steps_per_epoch)
    if use_wandb:
        wandb.log({'Test ehill Loss': epoch_loss_ehill / steps_per_epoch}, step=epoch)

    rmse_ehill = metric.rmse(lb_np, predict_np)
    metrics_summary['rmse_list_test_ehill'].append(rmse_ehill)
    print('RMSE ehill: %.4f' % rmse_ehill)
    if use_wandb:
        wandb.log({'Test RMSE ehill': rmse_ehill}, step=epoch)

    pearson_ehill, _ = metric.correlation(lb_np, predict_np, 'pearson')
    metrics_summary['pearson_ehill_list_test'].append(pearson_ehill)
    print("Pearson_ehill's correlation: %.4f" % pearson_ehill)
    if use_wandb:
        wandb.log({'Test Pearson_ehill': pearson_ehill}, step=epoch)

    spearman_ehill, _ = metric.correlation(lb_np, predict_np, 'spearman')
    metrics_summary['spearman_ehill_list_test'].append(spearman_ehill)
    print("Spearman_ehill's correlation: %.4f" % spearman_ehill)
    if use_wandb:
        wandb.log({'Test Spearman_ehill': spearman_ehill}, step=epoch)


def report_final_results(metrics_summary):
    best_dev_epoch = np.argmax(metrics_summary['spearman_ehill_list_dev'])
    print("Epoch %d got best Pearson's correlation of ehill on dev set: %.4f" % (
        best_dev_epoch + 1, metrics_summary['pearson_ehill_list_dev'][best_dev_epoch]))
    print("Epoch %d got Spearman's correlation of ehill on dev set: %.4f" % (
        best_dev_epoch + 1, metrics_summary['spearman_ehill_list_dev'][best_dev_epoch]))
    print("Epoch %d got RMSE of ehill on dev set: %.4f" % (
        best_dev_epoch + 1, metrics_summary['rmse_list_dev_ehill'][best_dev_epoch]))

    if metrics_summary['pearson_ehill_list_test']:
        print("Epoch %d got Pearson's correlation of ehill on test set w.r.t dev set: %.4f" % (
            best_dev_epoch + 1, metrics_summary['pearson_ehill_list_test'][best_dev_epoch]))
        print("Epoch %d got Spearman's correlation of ehill on test set w.r.t dev set: %.4f" % (
            best_dev_epoch + 1, metrics_summary['spearman_ehill_list_test'][best_dev_epoch]))
        print("Epoch %d got RMSE of ehill on test set w.r.t dev set: %.4f" % (
            best_dev_epoch + 1, metrics_summary['rmse_list_test_ehill'][best_dev_epoch]))

        best_test_epoch = np.argmax(metrics_summary['spearman_ehill_list_test'])
        print("Epoch %d got best Pearson's correlation of ehill on test set: %.4f" % (
            best_test_epoch + 1, metrics_summary['pearson_ehill_list_test'][best_test_epoch]))
        print("Epoch %d got Spearman's correlation of ehill on test set: %.4f" % (
            best_test_epoch + 1, metrics_summary['spearman_ehill_list_test'][best_test_epoch]))
        print("Epoch %d got RMSE of ehill on test set: %.4f" % (
            best_test_epoch + 1, metrics_summary['rmse_list_test_ehill'][best_test_epoch]))


def parquet_to_tmp_csv(parquet_path: str, tmp_path: str) -> str:
    """Write parquet to temp CSV compatible with datareader.read_data() format."""
    df = pd.read_parquet(parquet_path)
    df.to_csv(tmp_path, index=False)
    return tmp_path


def model_training(args, model, hill_data, metrics_summary, use_wandb):
    """Training loop with early stopping on dev Pearson. Saves best checkpoint."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    best_dev_pearson_ehill = float('-inf')
    patience_counter = 0
    best_dev_epoch = 0

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_path)), exist_ok=True)

    for epoch in range(args.max_epoch):
        print(f'Iteration {epoch + 1}:')
        print_lr(optimizer)
        model.train()
        epoch_loss_ehill = 0

        for i, (ft, lb, _) in enumerate(hill_data.train_dataloader()):
            drug = ft['drug']
            mask = ft['mask'].to(device)
            cell_feature = ft['cell_id']
            pert_idose = ft['pert_idose']
            optimizer.zero_grad()
            predict, cell_hidden_ = model(drug, hill_data.gene.to(device), mask, cell_feature,
                                          pert_idose, job_id='pretraining', epoch=epoch)
            loss = model.loss(lb.to(device), predict)
            loss.backward()
            optimizer.step()
            if i == 1:
                print('__________________________input__________________________')
                print(cell_feature)
                print('__________________________hidden__________________________')
                print(cell_hidden_)
            epoch_loss_ehill += loss.item()

        print('Train ehill loss:')
        print(epoch_loss_ehill / (i + 1))
        if use_wandb:
            wandb.log({'Train ehill loss': epoch_loss_ehill / (i + 1)}, step=epoch)

        # --- Validation ---
        model.eval()
        epoch_loss_ehill = 0
        lb_np = np.empty([0, ])
        predict_np = np.empty([0, ])

        with torch.no_grad():
            for i, (ft, lb, _) in enumerate(hill_data.val_dataloader()):
                drug = ft['drug']
                mask = ft['mask'].to(device)
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, _ = model(drug, hill_data.gene.to(device), mask, cell_feature,
                                   pert_idose, job_id='pretraining', epoch=epoch)
                loss_ehill = model.loss(lb.to(device), predict)
                epoch_loss_ehill += loss_ehill.item()
                lb_np = np.concatenate((lb_np, lb.cpu().numpy().reshape(-1)), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy().reshape(-1)), axis=0)

            validation_epoch_end(epoch_loss_ehill, lb_np, predict_np,
                                 i + 1, epoch, metrics_summary, use_wandb)

            current_pearson = metrics_summary['pearson_ehill_list_dev'][-1]
            if current_pearson > best_dev_pearson_ehill:
                best_dev_pearson_ehill = current_pearson
                best_dev_epoch = epoch
                patience_counter = 0
                # Save best checkpoint (halt_gate_1_pass filled after training)
                torch.save({
                    'state_dict': model.state_dict(),
                    'model_params': dict(model_param_registry_snapshot),
                    'best_epoch': best_dev_epoch,
                    'dev_rmse': metrics_summary['rmse_list_dev_ehill'][-1],
                    'dev_pearson': current_pearson,
                    'baseline_rmse': None,  # filled after training
                    'halt_gate_1_pass': None,  # filled after training
                    'multidcp_sha': MULTIDCP_SHA,
                }, args.checkpoint_path)
                print(f'Best model saved at epoch {epoch + 1} (dev Pearson={current_pearson:.4f})')
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f'Early stopping at epoch {epoch + 1} (patience={args.patience})')
                    break

        # --- Test (optional — does not affect early stopping) ---
        epoch_loss_ehill = 0
        lb_np = np.empty([0, ])
        predict_np = np.empty([0, ])
        with torch.no_grad():
            try:
                for i, (ft, lb, _) in enumerate(hill_data.test_dataloader()):
                    drug = ft['drug']
                    mask = ft['mask'].to(device)
                    cell_feature = ft['cell_id']
                    pert_idose = ft['pert_idose']
                    predict, _ = model(drug, hill_data.gene.to(device), mask, cell_feature,
                                       pert_idose, job_id='pretraining', epoch=epoch)
                    loss_ehill = model.loss(lb.to(device), predict)
                    epoch_loss_ehill += loss_ehill.item()
                    lb_np = np.concatenate((lb_np, lb.cpu().numpy().reshape(-1)), axis=0)
                    predict_np = np.concatenate((predict_np, predict.cpu().numpy().reshape(-1)), axis=0)
                test_epoch_end(epoch_loss_ehill, lb_np, predict_np,
                               i + 1, epoch, metrics_summary, use_wandb)
            except Exception as e:
                print(f'Warning: test eval skipped this epoch: {e}')

    return best_dev_epoch


# ---- Registry snapshot (populated in __main__, used in model_training checkpoint) --
model_param_registry_snapshot = {}


if __name__ == '__main__':

    start_time = datetime.now()

    parser = argparse.ArgumentParser(description='MultiDCP MODEL_DOSE Stage-1 training')
    # Core MultiDCP args (preserved from upstream)
    parser.add_argument('--drug_file')
    parser.add_argument('--gene_file')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hill_train_file')
    parser.add_argument('--hill_dev_file')
    parser.add_argument('--hill_test_file')
    parser.add_argument('--train_file')
    parser.add_argument('--dev_file')
    parser.add_argument('--test_file')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--all_cells',
                        default='/raid/home/joshua/data/MultiDCP/data/ehill_data/pretrain_cell_list_ehill.p')
    parser.add_argument('--cell_ge_file',
                        default='/raid/home/joshua/data/MultiDCP/data/adjusted_ccle_tcga_ad_tpm_log2.csv',
                        help='Cell line to gene expression file')
    parser.add_argument('--linear_encoder_flag', dest='linear_encoder_flag',
                        action='store_true', default=False,
                        help='Whether the cell embedding layer only has linear layers')
    # New args (Phase 1 additions)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (sets CUDA_VISIBLE_DEVICES before torch import)')
    parser.add_argument('--safe-parquet', action='store_true', default=False,
                        help='Use leakage-filtered parquet files for train/dev instead of hill_train_file/hill_dev_file')
    parser.add_argument('--train-parquet', default='data/processed/ehill_train_safe.parquet',
                        help='Path to leakage-safe train parquet (used with --safe-parquet)')
    parser.add_argument('--dev-parquet', default='data/processed/ehill_dev_safe.parquet',
                        help='Path to leakage-safe dev parquet (used with --safe-parquet)')
    parser.add_argument('--wandb-project', default='joshroll/MultiDCP_multihead_dili',
                        help='WandB project name')
    parser.add_argument('--wandb-group', default='model_dose',
                        help='WandB run group')
    parser.add_argument('--checkpoint-path', default='results/checkpoints/chkpt_dose.pt',
                        help='Path for model checkpoint')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience on dev Pearson')

    args = parser.parse_args()

    # ---- WandB init --------------------------------------------------------
    USE_WANDB = True
    if USE_WANDB:
        # WandB project cannot contain '/'; entity is set separately
        _project = args.wandb_project.split('/')[-1] if '/' in args.wandb_project else args.wandb_project
        _entity  = args.wandb_project.split('/')[0]  if '/' in args.wandb_project else None
        wandb.init(
            entity=_entity,
            project=_project,
            group=args.wandb_group,
            config=vars(args),
            name=f'model_dose_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
    else:
        os.environ['WANDB_MODE'] = 'dryrun'

    # ---- Safe-parquet adapter (BEFORE EhillDataLoader init) ----------------
    if args.safe_parquet:
        import tempfile
        _tmp_dir = tempfile.mkdtemp(prefix='ehill_dose_')
        train_csv_path = os.path.join(_tmp_dir, 'train.csv')
        dev_csv_path   = os.path.join(_tmp_dir, 'dev.csv')
        parquet_to_tmp_csv(args.train_parquet, train_csv_path)
        parquet_to_tmp_csv(args.dev_parquet,   dev_csv_path)
        args.hill_train_file = train_csv_path
        args.hill_dev_file   = dev_csv_path
        _train_shape = pd.read_parquet(args.train_parquet).shape
        _dev_shape   = pd.read_parquet(args.dev_parquet).shape
        print(f'Safe parquet mode: train={args.train_parquet} {_train_shape}, dev={args.dev_parquet} {_dev_shape}')

    # ---- Data filter -------------------------------------------------------
    all_cells = list(pickle.load(open(args.all_cells, 'rb')))
    # Note: data_utils.read_data() uses filter['cell_id'] to filter rows by cell line.
    # The original ehill_multidcp_pretrain.py had 'cell_feature' (wrong key), so filtering
    # was silently skipped there. We use the correct 'cell_id' key here.
    DATA_FILTER = {
        'time': '24H',
        'pert_id': ['BRD-U41416256', 'BRD-U60236422'],
        'pert_type': ['trt_cp'],
        'cell_id': all_cells,
        'pert_idose': ['0.04 um', '0.12 um', '0.37 um', '1.11 um', '3.33 um', '10.0 um']
    }

    # ---- EhillDataLoader ---------------------------------------------------
    hill_data = datareader.EhillDataLoader(DATA_FILTER, device, args)

    # Fix: upstream setup() only initializes test_data; we need train + dev too
    def _fixed_setup(self, stage=None):
        import datareader as _dr
        self.train_data = _dr.EhillDataset(self.drug_file, self.train_data_file,
                                            self.data_filter, self.device, self.cell_ge_file_name)
        self.dev_data   = _dr.EhillDataset(self.drug_file, self.dev_data_file,
                                            self.data_filter, self.device, self.cell_ge_file_name)
        self.test_data  = _dr.EhillDataset(self.drug_file, self.test_data_file,
                                            self.data_filter, self.device, self.cell_ge_file_name)
        self.use_pert_type  = self.test_data.use_pert_type
        self.use_cell_id    = self.test_data.use_cell_id
        self.use_pert_idose = self.test_data.use_pert_idose

    hill_data.setup = types.MethodType(_fixed_setup, hill_data)
    hill_data.setup()

    print(f'#Train hill data: {len(hill_data.train_data)}')
    print(f'#Dev hill data:   {len(hill_data.dev_data)}')
    print(f'#Test hill data:  {len(hill_data.test_data)}')

    # ---- PerturbedDataLoader (for num_gene — graceful fallback) ------------
    num_gene = 978  # default; overridden if PerturbedDataLoader succeeds
    try:
        data = datareader.PerturbedDataLoader(DATA_FILTER, device, args)
        data.setup()
        num_gene = int(np.shape(data.gene)[0])
        print(f'#Train perturbed data: {len(data.train_data)}')
        print(f'num_gene from PerturbedDataLoader: {num_gene}')
    except Exception as e:
        print(f'Warning: PerturbedDataLoader failed ({e}) — using num_gene={num_gene}')

    # ---- Model parameters --------------------------------------------------
    model_param_registry = initialize_model_registry()
    model_param_registry.update({
        'num_gene': num_gene,
        'pert_idose_input_dim': len(DATA_FILTER['pert_idose']),
        'dropout': args.dropout,
        'linear_encoder_flag': args.linear_encoder_flag,
    })
    model_param_registry_snapshot.update(dict(model_param_registry))

    # ---- Model creation ----------------------------------------------------
    print(f'--------------with linear encoder: {args.linear_encoder_flag!r}--------------')
    model = multidcp.MultiDCPEhillPretraining(device=device,
                                              model_param_registry=model_param_registry)
    model.init_weights(pretrained=False)
    model.to(device)
    model = model.double()

    if USE_WANDB:
        wandb.watch(model, log='all')

    # ---- Training ----------------------------------------------------------
    metrics_summary = defaultdict(
        pearson_ehill_list_dev=[],
        pearson_ehill_list_test=[],
        spearman_ehill_list_dev=[],
        spearman_ehill_list_test=[],
        rmse_list_dev_ehill=[],
        rmse_list_test_ehill=[]
    )

    best_dev_epoch = model_training(args, model, hill_data, metrics_summary, USE_WANDB)

    # ---- Halt gate 1 computation -------------------------------------------
    # Collect all dev labels to compute predict-mean baseline (= std of labels)
    print('\nComputing halt gate 1 baseline (std of dev labels)...')
    dev_labels_list = []
    with torch.no_grad():
        for ft, lb, _ in hill_data.val_dataloader():
            dev_labels_list.append(lb.cpu().numpy().reshape(-1))
    dev_labels_np = np.concatenate(dev_labels_list)
    baseline_rmse = float(np.std(dev_labels_np))  # predict-mean baseline

    best_dev_rmse = float(min(metrics_summary['rmse_list_dev_ehill']))
    halt_gate_1_pass = bool(best_dev_rmse < baseline_rmse)

    print('\n=== HALT GATE 1 ===')
    print(f'Dev RMSE (best):              {best_dev_rmse:.6f}')
    print(f'Predict-mean baseline RMSE:   {baseline_rmse:.6f}')
    print(f'HALT GATE 1: {"PASS" if halt_gate_1_pass else "FAIL *** HALT ***"}')

    # Update checkpoint with halt gate result
    ckpt_path = Path(args.checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        ckpt['baseline_rmse'] = baseline_rmse
        ckpt['halt_gate_1_pass'] = halt_gate_1_pass
        torch.save(ckpt, str(ckpt_path))
        print(f'Checkpoint updated: {ckpt_path}')
    else:
        # Checkpoint was never written (model never improved? save final state)
        os.makedirs(str(ckpt_path.parent), exist_ok=True)
        torch.save({
            'state_dict': model.state_dict(),
            'model_params': dict(model_param_registry_snapshot),
            'best_epoch': best_dev_epoch,
            'dev_rmse': best_dev_rmse,
            'dev_pearson': (max(metrics_summary['pearson_ehill_list_dev'])
                            if metrics_summary['pearson_ehill_list_dev'] else None),
            'baseline_rmse': baseline_rmse,
            'halt_gate_1_pass': halt_gate_1_pass,
            'multidcp_sha': MULTIDCP_SHA,
        }, str(ckpt_path))
        print(f'Checkpoint written (final state): {ckpt_path}')

    # ---- Final results report ----------------------------------------------
    report_final_results(metrics_summary)
    end_time = datetime.now()
    print(f'\nTotal training time: {end_time - start_time}')
