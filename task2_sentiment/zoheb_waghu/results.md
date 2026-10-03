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
| Train | 89,997 |
| Validation (held out of train) | 9,999 |
| Test (official split, evaluated **once**) | 38,000 |
| Vocabulary (train only, min freq 2, cap 30K) | 30,000 |
| Token length mean / median / p95 / max | 70.1 / 51 / 194 / 558 |
| Truncated at max_len=256 | 2.19% |
| Class balance (train / test positive) | 0.5019 / 0.5000 |
| Test OOV rate | 1.28% |

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
| Accuracy | 0.93463 | 0.93574 | **0.94024** |
| Macro-F1 | 0.93463 | 0.93573 | **0.94023** |
| Micro-F1 | 0.93463 | 0.93574 | **0.94024** |
| Weighted F1 | 0.93463 | 0.93573 | **0.94023** |
| ROC-AUC | 0.98300 | 0.98359 | **0.98533** |
| PR-AUC | 0.98352 | 0.98398 | **0.98564** |
| MCC | 0.86926 | 0.87164 | **0.88063** |
| Brier score | 0.04891 | 0.04807 | **0.04492** |
| ECE | 0.01341 | **0.00537** | 0.00878 |
| Confusion (TN / FP / FN / TP) | 17,766 / 1,234 / 1,250 / 17,750 | 17,963 / 1,037 / 1,405 / 17,595 | 18,046 / 954 / 1,317 / 17,683 |
| Accuracy 95% CI | [0.93210, 0.93705] | [0.93326, 0.93826] | [0.93789, 0.94263] |
| Macro-F1 95% CI | [0.93210, 0.93705] | [0.93326, 0.93826] | [0.93789, 0.94263] |
| MCC 95% CI | [0.86421, 0.87411] | [0.86670, 0.87670] | [0.87595, 0.88547] |
| McNemar vs baseline | - | χ²=0.808, **p=0.369** | χ²=26.52, **p=2.6e-7** |
| Parameters | 4,104,449 | 4,135,681 | 10,441,217 |
| Train time | 19.4 s | **16.6 s** | 436.4 s |
| Examples/sec | 18,568.8 | **21,723.1** | 825.0 |
| Peak memory | 1.28 GB | **0.22 GB** | 1.53 GB |
| Epochs run (early stop) / best epoch | 4 / 1 | 4 / 1 | 4 / 0 |

Confusion matrices and calibration curves:
[outputs/confusion_matrices/confusion_and_calibration.png](outputs/confusion_matrices/confusion_and_calibration.png)

### Per-slice robustness (macro-F1)

| Slice | M1 | M2 | M3 | n |
|---|---|---|---|---|
| short reviews (≤50 tokens) | 0.9349 | 0.9362 | **0.9406** | 18,860 |
| long reviews (>200 tokens) | 0.9151 | 0.9077 | **0.9194** | 1,697 |
| contains negation | 0.9294 | 0.9307 | **0.9361** | 22,583 |
| ≥3 exclamation marks | 0.9547 | 0.9522 | **0.9576** | 6,944 |

Comparison figure: [outputs/plots/model_comparison.png](outputs/plots/model_comparison.png)

## 4. Comparative analysis (5 marks)

**M2's apparent win over the baseline is not real.** M2 scores 0.2 points higher, but its
accuracy CI [0.93326, 0.93826] overlaps the baseline's [0.93210, 0.93705], so the two are not
distinguishable on this test set from the point estimates alone. The paired McNemar test is the
sensitive comparison - it conditions only on the 1,000-odd reviews where the two models
disagree - and it returns **p = 0.369**, which fails at α = 0.05. The honest conclusion is that
swapping recurrence for convolutions changed the cost profile, not the accuracy: M2 trains
**1.17× faster**, uses **about one-sixth the memory**, and is **the best-calibrated model** (ECE
0.00537 vs 0.01341) for statistically indistinguishable accuracy.

**M3's win is real.** Its CI [0.93789, 0.94263] is fully separated from the baseline's, and
McNemar gives p = 2.6e-7. The cost is severe: 2.5× the parameters, **22× the training time**,
1.2× the memory, for 0.56 percentage points of accuracy.

**Every model is worst on long reviews** (0.9077-0.9194 vs ~0.94 overall), and that gap is far
larger than any between-model gap. 2.19% of reviews are truncated at 256 tokens, and long
reviews are also where mixed sentiment concentrates - a review that praises the food and damns
the service. M3 attention recovers some of this gap but does not eliminate it, which suggests
the remaining bottleneck is truncation and mixed sentiment rather than pooling alone.

**Caveat on all of the above:** these are single-seed RTX 4090 runs, so no independent-seed
variance estimate is available. See the
[manifest](../../reproducibility/manifests/zoheb_waghu/).

## 5. Strengths, weaknesses, limitations

**Strengths.** Every model clears 93% with embeddings learned from scratch on 90K reviews.
Calibration is good (ECE ≤ 0.0135 everywhere, Brier ≤ 0.049), so the probabilities are usable as
confidences, not just rankings. The preprocessing decision on negation is measured rather than
assumed.

**Weaknesses.** Single seed per model. Only 90K of the available 560K training rows are used.
`max_len=256` truncates 2.19% of reviews and the long-review slice is the worst performer.
Lemmatisation without POS tags is crude (`better` does not reduce to `good`).

**Limitations.** Yelp polarity is binary and balanced by construction, so these numbers say
nothing about the imbalanced, multi-class case. The label noise visible in the error review
(reviews whose text is plainly positive but labelled negative) puts a ceiling on achievable
accuracy that none of these models can cross.

## 6. Improvements and future work

1. **Train on more data before touching architecture.** M3 bought 0.56 points for 22× compute;
   going 90K → 560K rows is likely cheaper per point.
2. **Raise or remove truncation** and re-measure the long-review slice specifically - the
   largest single gap in the table.
3. **Multiple seeds.** Three seeds per model would make the M1-vs-M2 verdict decidable instead
   of borderline.
4. **Sentence-level aggregation for mixed sentiment**, aimed at the same long-review weakness.

## 7. Hardware disclosure

| Model | Device | Processor | Peak memory | Train time |
|---|---|---|---|---|
| M1 BiLSTM-mean | cuda | NVIDIA GeForce RTX 4090 (24 GB) | 1.28 GB | 19.4 s |
| M2 TextCNN | cuda | NVIDIA GeForce RTX 4090 (24 GB) | 0.22 GB | 16.6 s |
| M3 BiLSTM-attention | cuda | NVIDIA GeForce RTX 4090 (24 GB) | 1.53 GB | 436.4 s |

Host: AMD Ryzen 9 7950X (16 cores), 128 GB RAM, Windows 11. PyTorch 2.5.1+cu124, Python 3.12.10,
NVIDIA driver 610.60. Peak memory is `torch.cuda.max_memory_allocated` - the true peak of GPU
tensor allocations during the run.
