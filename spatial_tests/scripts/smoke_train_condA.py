"""Condition-A smoke-train driver (WIRE-02): structure-only tox-head training.

Validates the ``src/spatial/tox_head.py`` concat-MLP, the BCEWithLogits training
loop, and wandb logging -- INDEPENDENT of the WIRE-01 frozen-model cache.

Condition A (RESEARCH Pattern 3): the GEX channel is a zero tensor
``torch.zeros(B, 10716)`` -- no frozen-model inference, no cache read. Only the
chem channel carries signal (ECFP4 2048-d on real DILI SMILES); the head must
learn from structure alone and the loss must move over >=1 epoch.

Chem channel = ECFP4 (2048-d, P1 ``smiles_to_ecfp4``). This is a SMOKE-TRAIN-ONLY
choice (RESEARCH A1 / Claude's Discretion); the headline ChemBERTa-vs-ECFP4
encoder decision is deferred to P4.

Real-data-only (Hard Rule 1): labels are DILIrank binary DILI labels and SMILES
are the P1 sibling-resolved cascade (dili_canonical -> drugbank), the SAME wiring
``scripts/run_p1_eda.py`` uses for the structure floor. NO synthetic labels, NO
fabricated SMILES.

Hard rules honored:
    - Hard Rule 5 (CUDA hygiene): ``--gpu`` is parsed and ``CUDA_VISIBLE_DEVICES``
      is set BEFORE ``import torch``; a free GPU is auto-detected via nvidia-smi
      and the LAST free GPU is never taken (always leave one free on the shared box).
    - Hard Rule 1 (real data only): real DILIrank labels + sibling-resolved SMILES;
      no synthetic data.
    - DE rule: the GEX channel is zeroed this phase (condition A); no raw expression
      is used as a feature.
    - No frozen checkpoint load, no cache read, no B/C/S-B/S-C training (Phase 4).

Analog: ``scripts/run_p1_eda.py`` (driver skeleton: sys.path injection, logging,
real-data label/SMILES/ECFP4 wiring in ``run_floor``, ``_write_report``) plus the
CUDA-hygiene block (no torch-using script existed in scripts/ before this one).

Usage::

    conda run -n dili_v04_env python scripts/smoke_train_condA.py \\
        --epochs 20 --wandb-mode offline
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import subprocess
import sys

# ---------------------------------------------------------------------------
# CUDA hygiene FIRST (Hard Rule 5 / Pitfall 7): set CUDA_VISIBLE_DEVICES BEFORE
# `import torch`. Parse --gpu, auto-detect a free GPU via nvidia-smi, never take
# the last free GPU (always leave one free on the shared box).
# ---------------------------------------------------------------------------


def _detect_free_gpu() -> str | None:
    """Pick a free GPU index via nvidia-smi, never the last free one.

    Returns the chosen index as a string, or ``None`` if no GPU can be used
    (nvidia-smi missing, no GPUs, or only one free GPU -- leave it free).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None

    free = []  # (mem_used_mib, index)
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
            mem_used = int(parts[1])
        except ValueError:
            continue
        # "free" = <1 GiB resident (no meaningful workload).
        if mem_used < 1024:
            free.append((mem_used, idx))

    # Always leave one GPU free: only claim a device if at least 2 are free.
    if len(free) < 2:
        return None
    free.sort()  # least-used first
    return str(free[0][1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Condition-A (structure-only) tox-head smoke-train (WIRE-02)."
    )
    parser.add_argument(
        "--gpu",
        default="auto",
        help="GPU index (e.g. '0'), 'auto' (nvidia-smi free-GPU pick, leave one "
        "free), or 'cpu' to force CPU. Default 'auto'.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs (>=1). Default 20."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size. Default 64.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Adam learning rate. Default 1e-3."
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb mode. Default 'online' (use 'offline' if no network).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed. Default 0."
    )
    return parser.parse_args()


_ARGS = _parse_args()

# Resolve the device selection and set CUDA_VISIBLE_DEVICES BEFORE importing torch.
if _ARGS.gpu == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _SELECTED_GPU = "cpu"
elif _ARGS.gpu == "auto":
    _SELECTED_GPU = _detect_free_gpu()
    if _SELECTED_GPU is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _SELECTED_GPU = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _SELECTED_GPU
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_ARGS.gpu)
    _SELECTED_GPU = str(_ARGS.gpu)

# ---------------------------------------------------------------------------
# sys.path injection (analog: run_p1_eda.py lines 43-44)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_train_condA")

# ---------------------------------------------------------------------------
# Library imports (AFTER CUDA_VISIBLE_DEVICES is set and sys.path injected)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import torch  # noqa: E402  -- imported AFTER CUDA_VISIBLE_DEVICES (Hard Rule 5)

from src.spatial.eda.fingerprints import smiles_to_ecfp4  # noqa: E402
from src.spatial.eda.labels import load_dilirank  # noqa: E402
from src.spatial.eda.smiles_join import join_smiles_cascade  # noqa: E402
from src.spatial.tox_head import ToxHead  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (real-data sources; SAME wiring as run_p1_eda.py)
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
DILIRANK_PATH = ROOT / "data" / "raw" / "labels" / "dilirank" / "dilirank.xlsx"
_SIBLING = ROOT.parent / "dili_downstream" / "data" / "processed"
DILI_CANONICAL_PATH = _SIBLING / "dili_canonical.csv"
DRUGBANK_PATH = _SIBLING / "drugbank_smiles_index.csv"

# Gene space (PDG; condition A zeroes this channel) -- mirror region_combiner N_PDG.
N_PDG = 10716
D_CHEM = 2048  # ECFP4 (P1)

# wandb (Claude's Discretion: project/run naming, D-05 / 02-CONTEXT).
# wandb project names cannot contain '/'; entity and project are separate fields.
WANDB_ENTITY = "joshroll"
WANDB_PROJECT = "DILI_spatial_xspecies"
WANDB_GROUP = "spatial_p2_smoke"
WANDB_RUN_NAME = "condA_structure_only_toxhead"


# ---------------------------------------------------------------------------
# Report helper (analog: run_p1_eda.py _write_report)
# ---------------------------------------------------------------------------


def _write_report(path: pathlib.Path, content: str, mode: str = "w") -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Real-data condition-A batch (DILIrank labels + sibling-resolved SMILES + ECFP4)
# ---------------------------------------------------------------------------


def _load_condA_data() -> tuple[np.ndarray, np.ndarray]:
    """Load real DILIrank labels + sibling-resolved SMILES -> ECFP4 chem channel.

    Reuses the SAME real-data wiring as run_p1_eda.py's structure floor (Hard
    Rule 1): DILIrank binary labels, the dili_canonical -> drugbank SMILES
    cascade, and ``smiles_to_ecfp4``. No synthetic labels, no fabricated SMILES.

    Returns
    -------
    fps : np.ndarray
        ECFP4 fingerprints, shape ``(n, 2048)``, float32, valid rows only.
    y : np.ndarray
        Binary DILI labels, shape ``(n,)``, float32.
    """
    labels_df = load_dilirank(str(DILIRANK_PATH))
    smiles_df = join_smiles_cascade(
        labels_df, str(DILI_CANONICAL_PATH), str(DRUGBANK_PATH)
    )
    covered = smiles_df["smiles"].notna()
    smiles = smiles_df.loc[covered, "smiles"].tolist()
    y_covered = smiles_df.loc[covered, "dili_binary"].values

    fps, valid_mask = smiles_to_ecfp4(smiles)
    fps_valid = fps[valid_mask].astype(np.float32)
    y_valid = y_covered[valid_mask].astype(np.float32)

    log.info(
        "Condition-A data: %d drugs with valid ECFP4 (real DILIrank labels, "
        "%.1f%% positive).",
        len(y_valid),
        100.0 * float(y_valid.mean()) if len(y_valid) else 0.0,
    )
    return fps_valid, y_valid


# ---------------------------------------------------------------------------
# Smoke-train
# ---------------------------------------------------------------------------


def run_smoke_train(args: argparse.Namespace) -> dict:
    """Train ToxHead on condition A (zero GEX channel) for >=1 epoch.

    GEX channel = ``torch.zeros(B, N_PDG)``; chem channel = ECFP4. BCEWithLogits.
    Asserts the final-epoch loss is below the first-epoch loss (the loss moves)
    and logs per-epoch loss to wandb.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if (_SELECTED_GPU != "cpu" and torch.cuda.is_available()) else "cpu"
    )
    log.info(
        "Device: %s (requested --gpu=%s, CUDA_VISIBLE_DEVICES=%r).",
        device,
        args.gpu,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )

    # --- Real-data condition-A batch ---
    fps, y = _load_condA_data()
    n = len(y)
    if n < 8:
        raise RuntimeError(
            f"Condition-A smoke-train needs >=8 real drugs, got {n}. "
            "Check DILIrank + SMILES wiring (do NOT fabricate labels, Hard Rule 1)."
        )

    chem_all = torch.from_numpy(fps).to(device)          # (n, 2048) float32
    y_all = torch.from_numpy(y).to(device)               # (n,) float32

    # --- wandb ---
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=WANDB_RUN_NAME,
        mode=args.wandb_mode,
        config={
            "condition": "A_structure_only",
            "d_gex": N_PDG,
            "d_chem": D_CHEM,
            "d_dr": 0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_drugs": n,
            "chem_encoder": "ECFP4_2048 (smoke-train only; ChemBERTa deferred to P4)",
            "device": str(device),
        },
    )

    # --- Head + optimizer + loss ---
    head = ToxHead(d_gex=N_PDG, d_chem=D_CHEM, d_dr=0).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    epoch_losses: list[float] = []
    head.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            if idx.numel() < 2:
                # BatchNorm1d needs batch size >= 2; skip a trailing singleton.
                continue
            chem_b = chem_all[idx]                              # (B, 2048)
            # Condition A: GEX channel is a zero tensor (no cache, no frozen pass).
            gex_b = torch.zeros(idx.numel(), N_PDG, device=device)
            y_b = y_all[idx]                                    # (B,)

            opt.zero_grad()
            out = head(gex_b, chem_b)                           # ToxHeadOutput
            loss = loss_fn(out.logit, y_b)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}: {loss.item()} "
                    "(zero-GEX-channel masking should keep the forward finite)."
                )
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        wandb.log({"epoch": epoch, "loss": mean_loss})
        log.info("epoch %d/%d  loss=%.4f", epoch + 1, args.epochs, mean_loss)

    first_loss = epoch_losses[0]
    last_loss = epoch_losses[-1]
    loss_moved = last_loss < first_loss
    run_url = getattr(run, "url", None)
    run_id = getattr(run, "id", None)
    wandb.summary["first_loss"] = first_loss
    wandb.summary["last_loss"] = last_loss
    wandb.summary["loss_moved"] = bool(loss_moved)
    wandb.finish()

    if not loss_moved:
        raise RuntimeError(
            f"Smoke-train loss did not move (first={first_loss:.4f}, "
            f"last={last_loss:.4f}). The head/loop is not learning."
        )

    log.info(
        "Smoke-train OK: loss %.4f -> %.4f over %d epochs (moved).",
        first_loss,
        last_loss,
        args.epochs,
    )
    return {
        "epochs": args.epochs,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_moved": loss_moved,
        "n_drugs": n,
        "device": str(device),
        "wandb_run_url": run_url,
        "wandb_run_id": run_id,
        "wandb_mode": args.wandb_mode,
    }


def main() -> int:
    log.info("=== Condition-A smoke-train (WIRE-02) ===")
    result = run_smoke_train(_ARGS)

    report = (
        "# Condition-A smoke-train summary (WIRE-02)\n\n"
        "Structure-only tox-head smoke-train: GEX channel = zero tensor "
        f"(d_gex={N_PDG}), chem channel = ECFP4 (d_chem={D_CHEM}); BCEWithLogits.\n\n"
        "| Field | Value |\n"
        "|-------|-------|\n"
        f"| Epochs | {result['epochs']} |\n"
        f"| n drugs (real DILIrank) | {result['n_drugs']} |\n"
        f"| First-epoch loss | {result['first_loss']:.4f} |\n"
        f"| Last-epoch loss | {result['last_loss']:.4f} |\n"
        f"| Loss moved | {result['loss_moved']} |\n"
        f"| Device | {result['device']} |\n"
        f"| wandb mode | {result['wandb_mode']} |\n"
        f"| wandb run id | {result['wandb_run_id']} |\n"
        f"| wandb run url | {result['wandb_run_url']} |\n\n"
        "Chem encoder = ECFP4 (smoke-train only, RESEARCH A1); the "
        "ChemBERTa-vs-ECFP4 headline choice is deferred to P4.\n"
    )
    _write_report(RESULTS / "P2_smoke_condA.md", report)
    log.info("Wrote smoke summary to %s", RESULTS / "P2_smoke_condA.md")

    print(
        f"smoke-train condition A OK: loss {result['first_loss']:.4f} -> "
        f"{result['last_loss']:.4f} over {result['epochs']} epochs "
        f"(device={result['device']}, wandb={result['wandb_mode']})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
