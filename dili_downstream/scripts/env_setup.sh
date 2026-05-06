#!/usr/bin/env bash
# Set up the dili_v04_env conda environment for v0.4.
#
# Strategy: clone mdcp_env (the parent project's env, shared with MultiDCP-CheMoE
# training) and add the v0.4-specific deps on top. This insulates v0.4 from
# breaking parent training if a new package conflicts.
#
# Usage:
#   bash scripts/env_setup.sh                  # idempotent; skips clone if env exists
#   bash scripts/env_setup.sh --force-clone    # destroy and re-clone

set -euo pipefail

PARENT_ENV="${PARENT_ENV:-mdcp_env}"
TARGET_ENV="${TARGET_ENV:-dili_v04_env}"
FORCE_CLONE=false

for arg in "$@"; do
    case "$arg" in
        --force-clone) FORCE_CLONE=true ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────────────

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH" >&2
    exit 1
fi

# Source conda's shell hook so `conda activate` works in this script.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$PARENT_ENV"; then
    echo "Parent env '$PARENT_ENV' not found. Set PARENT_ENV=<your_parent_env> and retry." >&2
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Clone parent env (idempotent)
# ──────────────────────────────────────────────────────────────────────────────

if conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
    if [ "$FORCE_CLONE" = "true" ]; then
        echo "→ Removing existing env: $TARGET_ENV"
        conda env remove --name "$TARGET_ENV" --yes
    else
        echo "✓ Env '$TARGET_ENV' already exists; skipping clone (use --force-clone to re-clone)"
    fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
    echo "→ Cloning '$PARENT_ENV' → '$TARGET_ENV' (this may take several minutes)"
    conda create --name "$TARGET_ENV" --clone "$PARENT_ENV" --yes
fi

# ──────────────────────────────────────────────────────────────────────────────
# Add v0.4-specific deps
# ──────────────────────────────────────────────────────────────────────────────

conda activate "$TARGET_ENV"

echo "→ Installing v0.4-specific Python packages"

# Core: TDC for DILI_Hong split
pip install --upgrade pytdc

# Chemical encoders (frozen public weights)
pip install --upgrade transformers      # ChemBERTa, MolFormer (HF)
# GIN: torch_geometric or dgl. Default to PyG; switch to dgl by editing this line.
pip install --upgrade torch_geometric
pip install --upgrade unimol_tools

# Interpretability + interpretability sanity (Phases 6–8)
pip install --upgrade gseapy captum scikit-learn

# DILIst download dependencies
pip install --upgrade requests openpyxl pandas

# Sanity: confirm imports actually load
python - <<'PYEOF'
import importlib
mods = ["tdc", "transformers", "torch_geometric", "unimol_tools",
        "gseapy", "captum", "sklearn", "requests", "openpyxl", "pandas"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print("✗ Some imports failed:")
    for m, err in missing:
        print(f"  - {m}: {err}")
    raise SystemExit(1)
print("✓ All v0.4 dependencies importable")
PYEOF

echo ""
echo "──────────────────────────────────────────────────────────────────────────"
echo "Done. Activate with:"
echo "    conda activate $TARGET_ENV"
echo ""
echo "Next: run scripts/download_dilist.py to fetch DILIst."
echo "──────────────────────────────────────────────────────────────────────────"
