# Task 2 — full-scale retrain (Colab A100)

**Why:** the first run trained on a 89,997-row subsample while shreya_akotiya used all 540K,
so the cross-member comparison in report §2.4 confounded architecture with training-set size
(the ◆ note). This retrain removes the confound by matching the data size.

**What changes:** `train_size: null` (full official train split) and `val_size: 20000`
(absolute, same budget as my teammate).
**What deliberately does not change:** my `max_len` 256, `max_vocab` 30000, my three
architectures and their hyperparameters. The brief requires each member to design their own
setup; only the data-size confound is being removed.

Cache lives in `data_processed/full540k/`, so the old 100K cache cannot be silently reused.

---

## Run

**Simplest route — open the notebook in Colab and run all cells.**

[`src/task2_sentiment.ipynb`](src/task2_sentiment.ipynb) section 0 now clones the repo, installs
the dependencies, trains all three models in order, and generates the figures and error review.
Everything below section 0 then reports the results, so one top-to-bottom execution both trains
and produces the deliverable notebook-with-outputs.

Section 0 is guarded by `IN_COLAB = "google.colab" in sys.modules`, so off Colab it prints a skip
message and the notebook stays a pure reporting document.

> **Pick a GPU runtime first** (Runtime → Change runtime type → A100). The setup cell prints the
> GPU name, or `NONE` if you forgot.

> **If a model fails partway**, re-run only cell 0.2 after fixing it — the preprocessing cache and
> the finished models' outputs are kept. The training cell stops on the first failure rather than
> continuing, so the later sections never report a mix of old and new runs.

### Equivalent, as separate shell commands

Prefer this if you want each model in its own cell, so a disconnect only costs one model:

```python
!git clone https://github.com/sakotiya/data266-lab1.git
%cd data266-lab1
!pip -q install datasets nltk statsmodels
import nltk; [nltk.download(p, quiet=True) for p in ("stopwords","wordnet","omw-1.4","punkt")]
import os; os.environ["LAB1_ROOT"] = "/content/data266-lab1"
```

```python
# baseline FIRST - the experimental runs read its predictions for McNemar
!python task2_sentiment/zoheb_waghu/src/train.py --config task2_sentiment/zoheb_waghu/configs/m1_baseline_bilstm.yaml
```
```python
!python task2_sentiment/zoheb_waghu/src/train.py --config task2_sentiment/zoheb_waghu/configs/m2_cnn_multikernel.yaml
```
```python
!python task2_sentiment/zoheb_waghu/src/train.py --config task2_sentiment/zoheb_waghu/configs/m3_bilstm_attention.yaml
```
```python
!python task2_sentiment/zoheb_waghu/src/plots.py        --config task2_sentiment/zoheb_waghu/configs/m1_baseline_bilstm.yaml
!python task2_sentiment/zoheb_waghu/src/error_review.py --config task2_sentiment/zoheb_waghu/configs/m1_baseline_bilstm.yaml --model t2_m1_baseline
```

### Getting the results back

Commit from Colab, or download:

```
task2_sentiment/zoheb_waghu/metrics_report.csv
task2_sentiment/zoheb_waghu/metrics_report_extended.csv
task2_sentiment/zoheb_waghu/outputs/                      (plots, probs, error review, history)
task2_sentiment/zoheb_waghu/src/task2_sentiment.ipynb     (with its new outputs)
reproducibility/raw_logs/zoheb_waghu/task2_sentiment/     (the evidence trail - required)
```

## Expected cost

Scaling the committed RTX 4090 timings by 6× (90K → 540K): M1 ~2 min, M2 ~2 min, M3 ~45 min,
plus one-off preprocessing. Budget about an hour on an A100.

## After the run — what must be updated

1. `metrics_report.csv` / `_extended.csv` are rewritten by the runs themselves.
2. **The hardware column will say A100, not RTX 4090** — report §2.4's hardware row and the
   speed note must change for my three models.
3. Report §2.2 "training data used" → 540K for both members.
4. Report §2.4: the ◆ note should narrow to the remaining differences (vocab cap 30K vs 50K,
   max_len 256 vs 200) now that training-set size is matched.
5. §2.6 point 1 currently says the size difference is the likely main factor — that claim
   becomes testable and must be rewritten against the new numbers.
6. The 20-error review in `failure_analysis.md` refers to specific test rows of the old model;
   re-extract and re-annotate, or state that it describes the superseded 90K model.
7. **Re-execute `src/task2_sentiment.ipynb`.** Its saved outputs are from the 90K / RTX 4090 run.
   It now reads the cache path from the config rather than hard-coding `data_processed/`, so it
   will pick up `full540k` automatically — but the committed outputs stay stale until it is run
   again. Add it to the Colab sequence:

   ```python
   !pip -q install jupyter nbconvert
   !jupyter nbconvert --to notebook --execute --inplace \
       task2_sentiment/zoheb_waghu/src/task2_sentiment.ipynb
   ```
