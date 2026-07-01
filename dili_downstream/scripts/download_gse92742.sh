#!/usr/bin/env bash
# GSE92742 (LINCS phase 1) Level-5 standard MODZ + metadata — the expression source
# for the Li/Tong 2020 (fbioe.2020.562677) 6,000-profile DILI benchmark. Public GEO, no auth.
set -euo pipefail
DEST="/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream/data/raw/GSE92742"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl"
mkdir -p "$DEST"; cd "$DEST"
for f in \
  GSE92742_Broad_LINCS_gene_info.txt.gz \
  GSE92742_Broad_LINCS_sig_info.txt.gz \
  GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz ; do
  echo ">>> $f"
  wget -c -q --show-progress "$BASE/$f" -O "$f"
done
echo "ALL_DOWNLOADS_DONE"
ls -la "$DEST"
