# Task 2 - Results: Yelp polarity sentiment classification (zoheb_waghu)

> **Metrics files.** `metrics_report.csv` uses the **team-agreed column schema** so my numbers
> line up with my teammate's. The team header omits some metrics the brief requires, so
> `metrics_report_extended.csv` carries the full set alongside it (see the table below for
> which columns are extra).
>
> Extra in the extended table: confusion-matrix cells (`tn`/`fp`/`fn`/`tp`), all eight per-slice columns, `split`, `config_path`, `device`.


Three models, all with embeddings **learned from scratch**. No pretrained embeddings, no
pretrained language models. Full numbers: [metrics_report.csv](metrics_report.csv) ·
run provenance: [manifest](../../reproducibility/manifests/zoheb_waghu/)

## 0. Dataset note

Dataset confirmed: **Yelp polarity** (560K train / 38K test). Both members train on it and
evaluate on the same official 38K test split.

## 1. Data and preprocessing (10 marks)

| | value |
|---|---|
| Train | **539,947** (the full official split) |
| Validation (held out of train) | 20,000 |
| Test (official split, evaluated **once**) | 38,000 |
| Vocabulary (train only, min freq 2, cap 30K) | 30,000 |
| Token length mean / median / p95 / max | 69.95 / 51 / 194 / 758 |
| Truncated at max_len=256 | 2.13% |
| Class balance (train / test positive) | 0.4999 / 0.5000 |
| Test OOV rate | 1.15% |

EDA figure: [outputs/plots/eda_overview.png](outputs/plots/eda_overview.png)

Malformed handling: rows with null/empty text, text shorter than 3 characters, labels outside
{0,1}, or reviews that become empty after cleaning are dropped and counted in the raw log
(`prepare` events).

Pipeline: lowercase → expand contractions → strip punctuation and special characters →
stopword removal → lemmatisation (WordNet) → tokenise → index.

### Negation handling and its justification

NLTK's English stopword list has 198 entries, **39 of which invert sentiment**: `not`, `no`,
`nor`, and every `n't` contraction form (`didn't`, `wasn't`, `couldn't`, …). Removing them turns

> "the food was **not** good and I **won't** be coming back"

into `food good coming back`, which reads as positive. Two decisions follow:

1. **Contractions are expanded before stopword removal.** Otherwise `don't` tokenises to `don`
   + `t`, and `don` is itself dropped - the negation disappears before the stopword filter ever
   sees it.
2. **The 39 inverting words are excluded from removal** (`get_stopwords()` in `src/data.py`),
   leaving 158 stopwords actually removed.

Whether this paid off is measured, not asserted: the `contains_negation` slice scores macro-F1
0.9294 / 0.9307 / 0.9361 across the three models, against 0.9346 / 0.9357 / 0.9402 overall -
a penalty of roughly 0.5 points rather than the collapse that discarding negations would cause.

Embeddings: `nn.Embedding`, uniform init in [-0.1, 0.1], `padding_idx=0` held at zero, trained
end to end. Nothing pretrained enters the pipeline.

## 2. The three models (15 marks)

| | M1 BiLSTM-mean (baseline) | M2 TextCNN | M3 BiLSTM-attention |
|---|---|---|---|
| Encoder | 1-layer BiLSTM, hidden 128 | Conv1d, kernels 2/3/4/5 × 128 filters | 2-layer BiLSTM, hidden 256 |
| Pooling | mean over non-pad steps | global max per filter | additive (Bahdanau) attention |
| Embedding dim | 128 | 128 | 256 |
| Dropout | 0.3 | 0.5 | 0.4 |
| Optimiser | Adam 1e-3, no schedule | Adam 1e-3, wd 1e-5 | AdamW 2e-3, cosine + 300-step warm-up |
| Batch / epochs | 64 / 8 | 64 / 8 | 32 / 12 |
| Parameters | 4,104,449 | 4,135,681 | 10,441,217 |

**What each one tests.** M1 is the baseline: a sequential encoder with no learned weighting -
every timestep contributes equally, so it cannot discount padding-adjacent filler or emphasise
a decisive clause. M2 changes the *encoder family*: it drops recurrence entirely and asks
whether local n-gram evidence (2- to 5-grams) is sufficient for polarity, which is a genuinely
different hypothesis rather than a hyperparameter tweak. M3 keeps the baseline's recurrent
encoder and adds the thing the baseline lacks - learned token weighting - plus the depth, width
and schedule to support it, isolating "does attention pooling beat mean pooling here".

## 3. Metrics (test split, evaluated once)

| Metric | M1 baseline | M2 TextCNN | M3 BiLSTM-attn |
|---|---|---|---|
| Accuracy | 0.95350 | 0.95021 | **0.95529** |
| Macro-F1 | 0.95350 | 0.95021 | **0.95529** |
| Micro-F1 | 0.95350 | 0.95021 | **0.95529** |
| Weighted F1 | 0.95350 | 0.95021 | **0.95529** |
| ROC-AUC | 0.99097 | 0.98962 | **0.99156** |
| PR-AUC | 0.99118 | 0.98985 | **0.99170** |
| MCC | 0.90703 | 0.90043 | **0.91062** |
| Brier score | 0.03499 | 0.03710 | **0.03354** |
| ECE | 0.01002 | **0.00440** | 0.00533 |
| Confusion (TN / FP / FN / TP) | 18,042 / 958 / 809 / 18,191 | 18,009 / 991 / 901 / 18,099 | 18,064 / 936 / 763 / 18,237 |
| Accuracy 95% CI | [0.95134, 0.95555] | [0.94800, 0.95237] | [0.95313, 0.95729] |
| MCC 95% CI | [0.90269, 0.91111] | [0.89602, 0.90478] | [0.90629, 0.91462] |
| McNemar vs baseline | - | χ²=10.14, **p=1.45e-03** | χ²=4.51, **p=0.034** |
| Parameters | 4,104,449 | 4,135,681 | 10,441,217 |
| Train time | 226.8 s | 260.5 s | 3,684.3 s |
| Examples/sec | 11,903.5 | **14,508.0** | 879.3 |
| Peak memory | 0.74 GB | **0.22 GB** | 1.53 GB |

Confusion matrices and calibration curves:
[outputs/confusion_matrices/confusion_and_calibration.png](outputs/confusion_matrices/confusion_and_calibration.png)

### Per-slice robustness (macro-F1)

| Slice | M1 | M2 | M3 | n |
|---|---|---|---|---|
| short reviews (≤50 tokens) | 0.9530 | 0.9509 | **0.9542** | 18,860 |
| long reviews (>200 tokens) | **0.9408** | 0.9275 | 0.9396 | 1,697 |
| contains negation | 0.9508 | 0.9477 | **0.9518** | 22,583 |
| ≥3 exclamation marks | 0.9684 | 0.9627 | **0.9697** | 6,944 |

Comparison figure: [outputs/plots/model_comparison.png](outputs/plots/model_comparison.png)

## 4. Comparative analysis (5 marks)

**M2 is significantly *worse* than the baseline.** The TextCNN scores **0.33 points below** it
(0.95021 vs 0.95350) and McNemar **separates them** (χ²=10.14, p=1.45e-03). With this much data the
recurrent encoder makes better use of it than the convolutional one. That is the expected direction
once there is enough data to learn long-range order: max-pooled
n-gram detectors saturate, because each filter can only report its strongest local match no matter
how much more text it sees, while the BiLSTM keeps accumulating sentence-level state.

**M3 wins, but narrowly.** Its accuracy CI [0.95313, 0.95729] **overlaps** the baseline's
[0.95134, 0.95555]. The paired McNemar test still separates them (χ²=4.51, p=0.034), so the gain is
real but small: on the full training set all three models are close together.

It is also a concrete argument for why both statistics belong in the report. On CIs alone I would
now conclude "no difference"; on McNemar I would conclude "M3 is better". The paired test is the
sensitive one because it conditions on the ~1,700 reviews where the two models disagree instead of
comparing two whole-test-set accuracies.

**Long reviews are only slightly harder.** The long-review slice scores 0.9275-0.9408 against
0.950-0.955 overall, a gap of about 1.3-2.3 points, and the baseline is **best** on it.

**Cost.** M3 buys 0.18 points over the baseline for **2.5× the parameters and 16× the training
time** (3,684 s vs 227 s). M2 is the cheapest to serve — 0.22 GB peak memory and the best
calibration (ECE 0.0044) — but it is now the least accurate of the three.

**Caveat on all of the above:** single-seed runs on one A100, so there is no independent-seed
variance estimate. The CI overlap above means the M1-vs-M3 difference is near the resolution of
this test set, and a second seed could plausibly reorder them.

## 5. Strengths, weaknesses, limitations

**Strengths.** Every model clears 95% with embeddings learned from scratch on the full 540K split.
Calibration is good (ECE ≤ 0.0100 and Brier ≤ 0.037 everywhere), so the probabilities are usable as confidences, not just rankings.
The preprocessing decision on negation is measured rather than assumed, and the negation slice now
scores within 0.4 points of overall.

**Weaknesses.** Single seed per model, which matters more now that the M1-vs-M3 confidence
intervals overlap - the ordering of the two best models is not robustly established.
`max_len=256` still truncates 2.13% of reviews. Lemmatisation without POS tags is crude (`better`
does not reduce to `good`). M3 costs 16× the baseline's training time for 0.18 points.

**Limitations.** Yelp polarity is binary and balanced by construction, so these numbers say
nothing about the imbalanced, multi-class case. The label noise visible in the error review
(reviews whose text is plainly positive but labelled negative) puts a ceiling on achievable
accuracy that none of these models can cross.

## 6. Improvements and future work

1. **Multiple seeds, now the highest-value next step.** The M1-vs-M3 accuracy CIs overlap, so the
   ordering of my two best models rests on a single McNemar test at p=0.034. Three seeds per model
   would settle it, and it is far cheaper than any architecture change.
2. **Diagnose why the TextCNN falls behind the baseline.** Widening the kernel set or stacking a second convolutional block would test whether
   the limit is receptive field rather than capacity.
3. **Raise or remove truncation.** 2.13% of reviews are still cut at 256 tokens; measure the
   effect on the long-review slice.
4. **Sentence-level aggregation for mixed sentiment**, which the error review repeatedly surfaces
   as a cause of confident mistakes.

## 7. Hardware disclosure

| Model | Device | Processor | Peak memory | Train time |
|---|---|---|---|---|
| M1 BiLSTM-mean | cuda | NVIDIA A100-SXM4-40GB | 0.74 GB | 226.8 s |
| M2 TextCNN | cuda | NVIDIA A100-SXM4-40GB | 0.22 GB | 260.5 s |
| M3 BiLSTM-attention | cuda | NVIDIA A100-SXM4-40GB | 1.53 GB | 3,684.3 s |

Host: Google Colab, Linux 6.6.122 x86_64, 42.4 GB GPU memory. PyTorch 2.11.0+cu130, Python
3.13.15. Peak memory is `torch.cuda.max_memory_allocated` - the true peak of GPU tensor
allocations during the run.

> Wall-clock and examples/sec are only partly comparable with my teammate's, who used a T4 for
> Task 2.