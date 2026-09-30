# Task 2 - Run manifest (zoheb_waghu)

All reported Task 2 results come from the RTX 4090 runs below.

| run_id | model | config | checkpoint | raw log | metrics row |
|---|---|---|---|---|---|
| `t2_m1_baseline_20260929-211318` | M1 BiLSTM-mean (baseline) | `configs/m1_baseline_bilstm.yaml` | `checkpoints/t2_m1_baseline_20260929-211318_best.pt` | `raw_logs/zoheb_waghu/task2_sentiment/t2_m1_baseline_20260929-211318.jsonl` | `metrics_report.csv` row 1 |
| `t2_m2_cnn_20260929-211446` | M2 TextCNN | `configs/m2_cnn_multikernel.yaml` | `checkpoints/t2_m2_cnn_20260929-211446_best.pt` | `raw_logs/zoheb_waghu/task2_sentiment/t2_m2_cnn_20260929-211446.jsonl` | `metrics_report.csv` row 2 |
| `t2_m3_bilstm_attn_20260929-211521` | M3 BiLSTM-attention | `configs/m3_bilstm_attention.yaml` | `checkpoints/t2_m3_bilstm_attn_20260929-211521_best.pt` | `raw_logs/zoheb_waghu/task2_sentiment/t2_m3_bilstm_attn_20260929-211521.jsonl` | `metrics_report.csv` row 3 |

Checkpoints are gitignored (size); the IDs above identify them.

## Superseded runs - raw logs retained, results not reported

| run(s) | what happened |
|---|---|
| `t2_*_20260919-*` (M1, M2, M3 ×2) | First runs, on an Apple M5 (mps). Superseded by the RTX 4090 runs above; their metrics rows, checkpoints and per-run outputs were removed. Raw logs kept unedited as the evidence trail. |
| `t2_m1_baseline_20260929-211124` | Failed after 2 s, before training: `huggingface_hub` 1.33 rejected the short dataset id `yelp_polarity`. Fixed by pinning `huggingface_hub==1.8.0` (the version the first runs used). |
| `t2_smoke_*` | Pipeline smoke tests. |

**Run-to-run variance.** The M5 and 4090 runs share config, seed (1337), data split and
vocabulary; only the device and its kernels differ. Test accuracy moved by +0.15 (M1), +0.02 (M2)
and +0.13 (M3) points between them. The M5 runs also include M3 trained twice under an identical
setup, which differed by 0.07 points. All three orderings (M3 > M2 > M1) held on both devices.

## Shared preprocessing

All three models consume **identical** splits and vocabulary, built once from
`configs/_shared.yaml` and cached in `data_processed/` (`test_texts.json` is written last and
marks the cache complete). This is a correctness requirement, not an optimisation: the paired
McNemar test is only valid if every model is evaluated on the same test rows in the same order.

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
| Test probabilities per model (for McNemar) | `outputs/test_probs_<tag>.npy`, `outputs/test_probs_<run_id>.npy` |
| 20-error extraction with raw review text | `outputs/error_review_t2_m1_baseline.md` |
| Error buckets + slice metrics | `outputs/error_buckets_<tag>.json` |
| Per-epoch history | `outputs/history_<run_id>.json` |

## Environment

NVIDIA GeForce RTX 4090 (24 GB, driver 610.60), AMD Ryzen 9 7950X, 128 GB RAM, Windows 11.
Python 3.12.10, PyTorch 2.5.1+cu124. Package list:
`reproducibility/manifests/zoheb_waghu/env_freeze_rtx4090_20260929.txt`.

## Reproduce

```bash
python -c "import nltk; [nltk.download(p) for p in ('stopwords', 'wordnet', 'omw-1.4')]"
for m in m1_baseline_bilstm m2_cnn_multikernel m3_bilstm_attention; do
  python task2_sentiment/zoheb_waghu/src/train.py --config task2_sentiment/zoheb_waghu/configs/$m.yaml
done
python task2_sentiment/zoheb_waghu/src/plots.py        --config task2_sentiment/zoheb_waghu/configs/m1_baseline_bilstm.yaml
python task2_sentiment/zoheb_waghu/src/error_review.py --config task2_sentiment/zoheb_waghu/configs/m1_baseline_bilstm.yaml --model t2_m1_baseline
```

Run the baseline first - the experimental runs read `outputs/test_probs_t2_m1_baseline.npy`
to compute McNemar against it.
