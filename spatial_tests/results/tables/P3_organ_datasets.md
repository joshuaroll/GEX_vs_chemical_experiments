# Per-organ 978-gene LINCS training datasets (drug-disjoint splits)

Source: `processed_data_978modz_detplate_09012024.h5` | DE = x1(treated) - x2(control), 978 L1000 genes | basal = x2 | split seed 42, drug-disjoint (0.7, 0.15, 0.15).

| Organ | Cell lines (present) | Profiles | Unique drugs | train/dev/test profiles | train/dev/test drugs |
|---|---|---|---|---|---|
| liver | PHH, HEPG2, JHH5, HUH7 | 3876 | 3032 | 2698/580/598 | 2122/454/456 |
| kidney | HA1E, HEK293, HEK293T | 5575 | 3860 | 3898/824/853 | 2702/579/579 |
| brain | NPC, NEU, SHSY5Y, U251MG, GI1 | 6125 | 3325 | 4282/885/958 | 2327/498/500 |
| heart | none | 0 | 0 | — | — |

## Notes
- Drug-disjoint verified: no InChIKey14 appears in more than one split (assert passed).
- `x2` is the per-profile control (paired basal) -> used as input_cell_gex; target = `x1`.
- Heart has no LINCS cell line; not buildable from this corpus (needs external cardiac data).
- Lines are tissue-of-origin; PHH (primary hepatocytes) and NPC/NEU (iPSC neural) are the most organ-faithful; HEPG2/JHH5/HUH7 are hepatoma lines (volume vs fidelity tradeoff).
