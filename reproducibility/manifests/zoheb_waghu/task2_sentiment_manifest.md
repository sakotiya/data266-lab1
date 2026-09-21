# Task 2 - Run manifest (zoheb_waghu)

| run_id | model | config | checkpoint | raw log | metrics row |
|---|---|---|---|---|---|
| `t2_m1_baseline_20260919-232945` | M1 BiLSTM-mean (baseline) | `configs/m1_baseline_bilstm.yaml` | `checkpoints/t2_m1_baseline_20260919-232945_best.pt` | `logs/t2_m1_baseline_20260919-232945.jsonl` | `metrics_report.csv` row 1 |
| `t2_m2_cnn_20260919-233404` | M2 TextCNN | `configs/m2_cnn_multikernel.yaml` | `checkpoints/t2_m2_cnn_20260919-233404_best.pt` | `logs/t2_m2_cnn_20260919-233404.jsonl` | `metrics_report.csv` row 2 |
| `t2_m3_bilstm_attn_20260919-233850` | M3 BiLSTM-attention | `configs/m3_bilstm_attention.yaml` | `checkpoints/t2_m3_bilstm_attn_20260919-233850_best.pt` | `logs/t2_m3_bilstm_attn_20260919-233850.jsonl` | `metrics_report.csv` row 3 |

### The duplicate M3 run - disclosure

M3 was accidentally trained **twice, concurrently**. The first run
(`t2_m3_bilstm_attn_20260919-233621`) appeared stalled because this pipeline only logs per
epoch and its first epoch took nine minutes, so a second run was launched. Both completed
normally, both early-stopped at epoch 3 with the best checkpoint at epoch 0.

| run | best val macro-F1 | test accuracy | test macro-F1 | wall |
|---|---|---|---|---|
| `...233621` (first) | 0.93909 | 0.93966 | 0.93966 | 2460.7 s |
| `...233850` (reported) | 0.93949 | 0.93892 | 0.93892 | 2474.7 s |

Identical config and identical seed (1337); the gap is MPS kernel non-determinism, not a
difference in setup. **The reported row is `...233850`**, because both runs wrote their test
predictions to the same `test_probs_t2_m3_bilstm_attn.npy` and the surviving file is that run's
- so its metrics, predictions, error buckets and figures are mutually consistent. The first
run's raw log and checkpoint are retained; its metrics row was removed from
`metrics_report.csv` so the graded table has one row per model.

The overwrite that forced this choice is now fixed: `train.py` additionally writes an immutable
`test_probs_<run_id>.npy` alongside the stable per-tag name.

**This accident is the only run-to-run variance estimate in the whole task.** Every other number
here is a single seed, and two identical runs differed by 0.07 points of accuracy - worth
stating in the report, because it is the same order as the M1-to-M2 difference.

## Shared preprocessing

All three models consume **identical** splits and vocabulary, built once from
`configs/_shared.yaml` and cached in `data_processed/` (`ready_v1.json` is the cache stamp).
This is a correctness requirement, not an optimisation: the paired McNemar test is only valid
if every model is evaluated on the same test rows in the same order.

| | rows |
|---|---|
| train | 89,997 |
| validation (held out of train) | 9,999 |
| test (official split, evaluated once) | 38,000 |
| vocabulary | 30,000 |

## Outputs

| Artifact | Path |
|---|---|
| EDA: length distribution, class balance, summary | `outputs/plots/eda_overview.png` |
| Confusion matrices + calibration curves | `outputs/confusion_matrices/confusion_and_calibration.png` |
| Accuracy with 95% CI + per-slice macro-F1 | `outputs/plots/model_comparison.png` |
| Test probabilities per model (for McNemar) | `outputs/test_probs_<tag>.npy` |
| 20-error extraction with raw review text | `outputs/error_review_t2_m1_baseline.md` |
| Error buckets + slice metrics | `outputs/error_buckets_<tag>.json` |
| Per-epoch history | `outputs/history_<run_id>.json` |

## Environment

Python 3.9.6 (arm64), PyTorch 2.5.1, device **mps**, Apple M5 / 16 GB.
Package list: `logs/env_freeze_task2.txt`. Git commit recorded in each raw log's `run_start`.

## Reproduce

```bash
for m in m1_baseline_bilstm m2_cnn_multikernel m3_bilstm_attention; do
  ./.venv/bin/python task2_sentiment/zoheb_waghu/src/train.py --config task2_sentiment/zoheb_waghu/configs/$m.yaml
done
```

Run the baseline first - the experimental runs read `outputs/test_probs_t2_m1_baseline.npy`
to compute McNemar against it.
