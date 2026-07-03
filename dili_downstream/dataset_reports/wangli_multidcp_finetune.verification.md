# Dataset verification: wangli_multidcp_finetune

- File: `data/processed/wangli_multidcp_finetune.npz`
- Format: npz
- Verdict: **PASS_WITH_WARNINGS**
- Checked: 2026-07-01T16:51:02

## Checks

| check | status | detail |
|---|---|---|
| integrity.not_html | PASS | not an HTML error page |
| integrity.nonempty | PASS | 23300540 bytes |
| integrity.parses | WARN | load failed: ValueError: Object arrays cannot be loaded when allow_pickle=False |
| integrity.sha256 | INFO | 621f172af44348a5... (no expected hash registered) |
| provenance.recorded | PASS | source registered |

## EDA fingerprint

```json
{
  "load_note": "load failed: ValueError: Object arrays cannot be loaded when allow_pickle=False",
  "sha256": "621f172af44348a52b6c96c98a09acaff4366170a8f973d3aa99d453a0cdd547"
}
```
