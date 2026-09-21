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

The brief's task text says **Yelp polarity**; the folder tree in the same brief labels the
shared data IMDB. Proceeding on Yelp polarity - `data.dataset` in `configs/_shared.yaml`
switches it in one line if the instructor rules otherwise. **Still unconfirmed.**

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
2. **The 39 inverting words are excluded from removal** (`stopwords: nltk_english_minus_negations`),
   leaving 158 stopwords actually removed.

Whether this paid off is measured, not asserted: the `contains_negation` slice scores macro-F1
0.9272 / 0.9304 / 0.9334 across the three models, against 0.9331 / 0.9355 / 0.9389 overall -
a penalty of roughly 0.6 points rather than the collapse that discarding negations would cause.

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
| Accuracy | 0.93313 | 0.93553 | **0.93892** |
| Macro-F1 | 0.93312 | 0.93553 | **0.93892** |
| Micro-F1 | 0.93313 | 0.93553 | **0.93892** |
| Weighted F1 | 0.93312 | 0.93553 | **0.93892** |
| ROC-AUC | 0.98282 | 0.98359 | **0.98523** |
| PR-AUC | 0.98336 | 0.98410 | **0.98542** |
| MCC | 0.86664 | 0.87105 | **0.87801** |
| Brier score | 0.05038 | 0.04799 | **0.04560** |
| ECE | 0.01660 | 0.00566 | **0.00514** |
| Accuracy 95% CI | [0.93071, 0.93566] | [0.93297, 0.93808] | [0.93653, 0.94126] |
| Macro-F1 95% CI | [0.93069, 0.93564] | [0.93297, 0.93808] | [0.93651, 0.94125] |
| MCC 95% CI | [0.86144, 0.87137] | [0.86600, 0.87621] | [0.87309, 0.88255] |
| McNemar vs baseline | - | χ²=3.745, **p=0.053** | χ²=24.47, **p=7.5e-7** |
| Parameters | 4,104,449 | 4,135,681 | 10,441,217 |
| Train time | 213.8 s | **124.8 s** | 2,427.6 s |
| Examples/sec | 1,683.9 | **2,884.4** | 148.3 |
| Peak memory | 2.10 GB | **1.14 GB** | 5.68 GB |

Confusion matrices and calibration curves:
[outputs/confusion_matrices/confusion_and_calibration.png](outputs/confusion_matrices/confusion_and_calibration.png)

### Per-slice robustness (macro-F1)

| Slice | M1 | M2 | M3 | n |
|---|---|---|---|---|
| short reviews (≤50 tokens) | 0.9332 | 0.9369 | **0.9398** | large |
| long reviews (>200 tokens) | 0.9067 | 0.9053 | **0.9092** | small |
| contains negation | 0.9272 | 0.9304 | **0.9334** | large |
| ≥3 exclamation marks | 0.9512 | 0.9532 | **0.9563** | small |

Comparison figure: [outputs/plots/model_comparison.png](outputs/plots/model_comparison.png)

## 4. Comparative analysis (5 marks)

**M2's apparent win over the baseline is not real.** M2 scores 0.2 points higher, but its
accuracy CI [0.93297, 0.93808] overlaps the baseline's [0.93071, 0.93566], so the two are not
distinguishable on this test set from the point estimates alone. The paired McNemar test is the
sensitive comparison - it conditions only on the 1,000-odd reviews where the two models
disagree - and it returns **p = 0.053**, which fails at α = 0.05. The honest conclusion is that
swapping recurrence for convolutions changed the cost profile, not the accuracy: M2 trains
**1.7× faster**, uses **half the memory**, and is **substantially better calibrated** (ECE
0.0057 vs 0.0166) for statistically indistinguishable accuracy.

**M3's win is real.** Its CI [0.93653, 0.94126] is fully separated from the baseline's, and
McNemar gives p = 7.5e-7. The cost is severe: 2.5× the parameters, **11× the training time**,
2.7× the memory, for 0.58 points of accuracy.

**Every model is worst on long reviews** (0.905-0.909 vs ~0.94 overall), and that gap is far
larger than any between-model gap. 2.19% of reviews are truncated at 256 tokens, and long
reviews are also where mixed sentiment concentrates - a review that praises the food and damns
the service. Attention helps least exactly where the models are weakest, which suggests the
bottleneck is truncation and mixed sentiment rather than pooling strategy.

**Caveat on all of the above:** these are single-seed runs. M3 was accidentally trained twice
under an identical config and seed, and the two runs differed by 0.07 points of accuracy
(0.93892 vs 0.93966) - the same order as the M1→M2 gap I just declared insignificant. See
[manifest](../../reproducibility/manifests/zoheb_waghu/).

## 5. Strengths, weaknesses, limitations

**Strengths.** Every model clears 93% with embeddings learned from scratch on 90K reviews.
Calibration is good (ECE ≤ 0.017 everywhere, Brier ≤ 0.051), so the probabilities are usable as
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

1. **Train on more data before touching architecture.** M3 bought 0.58 points for 11× compute;
   going 90K → 560K rows is likely cheaper per point.
2. **Raise or remove truncation** and re-measure the long-review slice specifically - the
   largest single gap in the table.
3. **Multiple seeds.** Three seeds per model would make the M1-vs-M2 verdict decidable instead
   of borderline.
4. **Sentence-level aggregation for mixed sentiment**, aimed at the same long-review weakness.

## 7. Hardware disclosure

| Model | Device | Processor | Peak memory | Train time |
|---|---|---|---|---|
| M1 BiLSTM-mean | mps | Apple M5 (arm64), 16 GB | 2.10 GB | 213.8 s |
| M2 TextCNN | mps | Apple M5 (arm64), 16 GB | 1.14 GB | 124.8 s |
| M3 BiLSTM-attention | mps | Apple M5 (arm64), 16 GB | 5.68 GB | 2,427.6 s |

PyTorch 2.5.1, Python 3.9.6 (arm64). MPS exposes no true peak-memory counter, so peak memory is
`torch.mps.driver_allocated_memory` sampled at end of training - not directly comparable to
CUDA's `max_memory_allocated`.
