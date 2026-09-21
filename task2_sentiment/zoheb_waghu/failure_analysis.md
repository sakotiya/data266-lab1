# Task 2 - Error review (20 errors, manual)

Model under review: **M1 BiLSTM-mean (baseline)**, run `t2_m1_baseline_20260919-232945`
Test accuracy 0.93313 · macro-F1 0.93312 · MCC 0.86664 · Brier 0.05038

The 20 errors - five confident false positives, five confident false negatives, five
near-threshold errors, and five from the worst robustness slice - are extracted **with the raw
review text** into [outputs/error_review_t2_m1_baseline.md](outputs/error_review_t2_m1_baseline.md)
by `src/error_review.py`, which also records each review's length, exclamation count and whether
it contains a negation.

> **The error type, the testable fix and the metric it should move are left blank in that file
> on purpose. They are judgement calls you defend at the viva, so they must be yours.**
> Work through the extracted file, fill the three fields per error, then summarise below.

Worst slice (selected automatically): **`long_reviews`** - macro-F1 0.9067 against 0.9331
overall, the largest gap of any slice for every one of the three models.

## Group summary

| Group | n | Definition | Selected by |
|---|---|---|---|
| A. Confident false positives | 5 | predicted positive, actually negative, highest p(pos) | `error_buckets` |
| B. Confident false negatives | 5 | predicted negative, actually positive, lowest p(pos) | `error_buckets` |
| C. Near-threshold errors | 5 | wrong with \|p − 0.5\| ≤ 0.05 | `error_buckets` |
| D. Worst-slice errors | 5 | wrong within `long_reviews` | `error_buckets` |

## What to fill in

For each of the 20, assign one type from:
`negation` · `sarcasm/irony` · `mixed sentiment` · `aspect confusion` ·
`rating-text mismatch` · `domain term` · `length truncation` ·
`rare vocabulary / OOV` · `label noise` · `other`

then one **testable** fix and the metric that should move if the fix works.

## Synthesis (complete after annotating)

- Dominant error type across the 20: `<type>` (`<n>`/20)
- Confident errors vs near-threshold errors - what the split says about calibration. Cross-check
  against the measured Brier 0.05038 and ECE 0.01660: the model is well calibrated overall, so a
  large number of *confident* errors points at label or annotation problems rather than at
  under-confidence.
- Does the worst slice (`long_reviews`) share a cause with the confident errors?
- The single fix to run first, and why: `<...>`

## Evidence already in hand

These are measurements, not interpretations - use them to support or reject whatever type
assignment you make:

| Signal | Value |
|---|---|
| `long_reviews` macro-F1 vs overall | 0.9067 vs 0.9331 |
| `contains_negation` macro-F1 vs overall | 0.9272 vs 0.9331 |
| Reviews truncated at `max_len=256` | 2.19% |
| Test OOV rate | 1.28% |
| ECE / Brier (calibration) | 0.01660 / 0.05038 |
| Negation words deliberately kept from the stopword list | 39 |
