# Task 2 - Error review (20 errors, manual)

Model under review: **M1 BiLSTM-mean (baseline)**, run `t2_m1_baseline_20261006-210947` (Colab A100)
Test accuracy 0.95350 · macro-F1 0.95350 · MCC 0.90703 · Brier 0.03499 · ECE 0.01002

> **Re-extraction pending annotation.** These 20 errors were re-extracted from the retrained
> 540K model (`t2_m1_baseline_20261006-210947`); the extraction is in
> [outputs/error_review_t2_m1_baseline.md](outputs/error_review_t2_m1_baseline.md) with the error
> type and testable fix blank for each case. They are **different reviews** from the 20 analysed
> below, which came from the superseded 89,997-row model, so the annotation counts and case
> references in "Annotation summary" do not describe this model and must be redone against the new
> extraction. The measured figures on this page are already updated.

The 20 errors - five confident false positives, five confident false negatives, five
near-threshold errors, and five from the worst robustness slice - are extracted **with the raw
review text** into [outputs/error_review_t2_m1_baseline.md](outputs/error_review_t2_m1_baseline.md)
by `src/error_review.py`, which also records each review's length, exclamation count and whether
it contains a negation.

Each extracted error is manually annotated with an error type, a testable intervention, and the
metric expected to move. The labels describe the most plausible cause visible in the displayed
review; they are hypotheses to test, not additional ground-truth annotations.

Worst slice (selected automatically): **`long_reviews`** - macro-F1 0.9408 against 0.9535
overall, the largest gap of any slice for every one of the three models.

## Group summary

| Group | n | Definition | Selected by |
|---|---|---|---|
| A. Confident false positives | 5 | predicted positive, actually negative, highest p(pos) | `error_buckets` |
| B. Confident false negatives | 5 | predicted negative, actually positive, lowest p(pos) | `error_buckets` |
| C. Near-threshold errors | 5 | wrong with \|p − 0.5\| ≤ 0.05 | `error_buckets` |
| D. Worst-slice errors | 5 | wrong within `long_reviews` | `error_buckets` |

## Annotation summary (SUPERSEDED - describes the 89,997-row model)

| Error type | Count |
|---|---:|
| mixed sentiment | **12** |
| rating-text mismatch | 2 |
| domain term | 2 |
| length truncation | 2 |
| negation | 1 |
| sarcasm/irony | 1 |

Mixed sentiment is the dominant type (12/20). Mean pooling gives every retained token equal
influence, so an opening complaint, an obsolete review section, or a negative secondary aspect
can overwhelm the clause that determines the rating. This appears in both directions: cases 2-5
are negative labels with prominent positive language, while cases 7, 9, and 10 are positive
labels containing strong complaints.

The ten confident errors were deliberately selected from the most extreme mistakes, so their
frequency is not an estimate of overall calibration. Even so, cases 1 and 6 are especially
important: their visible text directly contradicts the supplied label, making rating-text
mismatch more plausible than model uncertainty. The remaining confident errors mostly contain
mixed aspects, temporal pivots, or an idiom. Together with Brier 0.04891 and ECE 0.01341, this
suggests calibration is good on average but can still be sharply wrong on particular discourse
patterns and noisy labels. The near-threshold group is qualitatively different: its probabilities
show appropriate uncertainty when positive and negative evidence compete.

The worst slice partly shares the same mixed-sentiment cause, but review length adds a distinct
failure. Two of its five examples exceed `max_len=256`, so the decisive closing text may never
reach the encoder. The other long-review cases contain many clauses whose evidence is diluted by
mean pooling. This matched the measured long-review macro-F1 of 0.9151, 1.95 points below overall **on the
89,997-row model**. On the retrained 540K model that gap narrows to 1.27 points (0.9408 vs
0.9535), and the baseline is now the *best* of the three models on long reviews - so the
length-truncation mechanism argued here needs re-testing rather than assuming it carries over.

**First intervention:** test a chunked hierarchical sentence encoder over up to 512 tokens,
holding the data split and optimiser fixed. It addresses both dominant mechanisms: sentence
attention can weight contrastive or concluding clauses, while chunking retains evidence after
token 256. The primary success metric is `long_reviews` macro-F1; secondary checks are overall
macro-F1, positive/negative recall, and Brier score. A useful result must improve long-review
macro-F1 without materially worsening calibration.

## Evidence already in hand

These are measurements, not interpretations - use them to support or reject whatever type
assignment you make:

| Signal | Value |
|---|---|
| `long_reviews` macro-F1 vs overall | 0.9408 vs 0.9535 (gap 1.27 pts, was 1.95 at 90K) |
| `contains_negation` macro-F1 vs overall | 0.9508 vs 0.9535 (gap 0.27 pts, was 0.52) |
| Reviews truncated at `max_len=256` | 2.13% |
| Test OOV rate | 1.15% |
| ECE / Brier (calibration) | 0.01002 / 0.03499 |
| Negation words deliberately kept from the stopword list | 39 |
