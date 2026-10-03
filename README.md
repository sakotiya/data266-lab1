# DATA 266 — Lab 1 (Team 32)

LLM pretraining · sentiment classification · CycleGAN style transfer.

**Members:** shreya_akotiya, zoheb_waghu

Each member builds their own model for all three tasks — own architecture, own
hyperparameters, own results — and commits under their own named folder inside each
task folder. No two members' models share the same architecture *and* hyperparameters.

## Layout

```
task1_llm/          TinyStories, GPT from scratch (no prebuilt Transformer/attention)
task2_sentiment/    Yelp polarity, 3 models per member (no pretrained embeddings or LMs)
task3_gan/          CycleGAN, Monet <-> photo, Kaggle submission
reproducibility/    manifests/ (env + checkpoint mapping), raw_logs/ (unedited)
report/             DATA266_Lab1_Report_Team_32.pdf
```

Per member, per task:

```
<member>/
  src/                 code (.ipynb with outputs)
  data_processed/      your own preprocessing output - never shared
  checkpoints/         your trained weights
  outputs/             samples, predictions, plots, confusion matrices
  metrics_report.csv   every required metric for this task
  failure_analysis.md  required failure/error write-up
  results.md           architecture + hyperparameter justification
```

## Setup

```bash
git clone <repo-url>
cd data266-lab1
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   macOS/Linux: source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements.txt` pins the versions used for zoheb_waghu's reported runs (RTX 4090,
Python 3.12.10). Install torch from the CUDA index first - PyPI's Windows torch wheel is
CPU-only. `device: auto` in every config resolves cuda > mps > cpu.

For notebooks, register the venv as its own kernel:

```bash
python -m ipykernel install --user --name data266-lab1 --display-name "DATA266 Lab1 (py3.12 CUDA)"
```

## Assignment source of truth

- [`temp/DATA266_Lab1_Fall_2026.pdf`](temp/DATA266_Lab1_Fall_2026.pdf) - the brief itself. Where any
  derived checklist disagrees with it, the PDF wins.
- [`temp/Part3_Evaluation_Script.ipynb`](temp/Part3_Evaluation_Script.ipynb) - the
  **instructor-provided CycleGAN evaluator**. Task 3 numbers should come from this, not from a
  home-grown metric script. It expects:

  ```
  Part 3/Data/
    monet_jpg/   real Monet   = domain A real
    photo_jpg/   real Photo   = domain B real
    pred_A2B/    generated Photo  (Monet -> Photo)
    pred_B2A/    generated Monet  (Photo -> Monet)
  ```

  and reports **FID and MiFID in both directions**. Note the direction convention: `pred_A2B`
  is Monet→Photo. `zoheb_waghu/evaluate_local.py` reproduces its FID/MiFID exactly (same
  Inception, transforms, N_EVAL cap, index pairing; checked value-for-value on a test set) and
  writes `submission.csv` in its format. It adds KID, precision/recall, density/coverage, LPIPS,
  content cosine, cycle L1, the training-log stability columns and the human-audit summary:

  ```bash
  python task3_gan/zoheb_waghu/evaluate_local.py metrics --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml \
      --train-run-id <training run id> --checkpoint <checkpoint file> [--zip]
  python task3_gan/zoheb_waghu/evaluate_local.py audit-sheet --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml
  ```

  The script header documents the files it reads (`pred_*`, `cycle_*`, the training-log events).

> Per brief section 8, Claude or any similar AI assistant must not make the core architecture
> decisions or write the analysis - those must be each member's own understanding, defended at
> the viva. Session transcripts and assistant logs are gitignored and kept outside this repo.

## Open items for the team

- [ ] GPU Lab booking, Kaggle competition registration
- [ ] Agree per-member architectures and data splits so no two models match
- [x] **Task 2 dataset:** confirmed - **Yelp polarity** (560K train / 38K test). Both members
      use it.
- [x] **Task 3 submission format:** resolved - the class competition takes `submission.csv`
      (ID, FID, MiFID; values must match the instructor's script). Leaderboard score =
      (FID + MiFID) / 2, lower is better
- [ ] Confirm measurement-only pretrained nets (InceptionV3 for FID/KID, AlexNet for LPIPS) are
      acceptable - those metrics cannot be computed otherwise
- [ ] Widen or confirm the Task 2 / Task 3 metric headers (see above)

## Data

Raw datasets are not committed — they are too large for git. Each task's
`data/README.md` holds the fetch command; both members fetch into the same shared
`data/` folder so paths match.

| Task | Dataset | Location |
| --- | --- | --- |
| 1 | TinyStories | `task1_llm/data/` |
| 2 | Yelp polarity | `task2_sentiment/data/` |
| 3 | Monet / photo | `task3_gan/data/{monet_jpg,photo_jpg}/` |

## Reproducing a run

**Harness smoke test** - no GPU, no dataset download, a few seconds. Verifies that every
config loads and inherits correctly, that the run logger writes an append-only trail, and that
the metrics writers reject incomplete rows:

```bash
python common/smoke_test.py
```

**Full reproduction of zoheb_waghu's Task 1 run** - one command, ~7 min on an RTX 4090,
downloads TinyStories on first use:

```bash
python task1_llm/zoheb_waghu/src/train.py --config task1_llm/zoheb_waghu/configs/gpt_baseline.yaml
python task1_llm/zoheb_waghu/src/plots.py --history task1_llm/zoheb_waghu/outputs/history_<run_id>.json
```

**Task 2** - run the baseline first; the two experimental runs read its saved test predictions
to compute the paired McNemar test against it:

```bash
python -c "import nltk; [nltk.download(p) for p in ('stopwords', 'wordnet', 'omw-1.4')]"   # once
for m in m1_baseline_bilstm m2_cnn_multikernel m3_bilstm_attention; do
  python task2_sentiment/zoheb_waghu/src/train.py --config task2_sentiment/zoheb_waghu/configs/$m.yaml
done
python task2_sentiment/zoheb_waghu/src/plots.py        --config task2_sentiment/zoheb_waghu/configs/m1_baseline_bilstm.yaml
python task2_sentiment/zoheb_waghu/src/error_review.py --config task2_sentiment/zoheb_waghu/configs/m1_baseline_bilstm.yaml --model t2_m1_baseline
```

**Task 3 (zoheb_waghu)** - data from `task3_gan/data/README.md`. Training is ~6.7 h on an RTX 4090
(40 epochs); it checkpoints every epoch and resumes with `--run-id <id> --resume <id>_last.pt`:

```bash
python task3_gan/zoheb_waghu/src/train.py    --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml
python task3_gan/zoheb_waghu/src/infer.py    --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml --checkpoint <run_id>_final.pt
python task3_gan/zoheb_waghu/evaluate_local.py metrics --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml \
    --train-run-id <run_id> --checkpoint <run_id>_final.pt
```

Without retraining, `infer.py` also runs from the committed fp16 generators
(`--checkpoint t3_baseline_20260929-235720_generators_fp16.pt`).

**shreya_akotiya's Task 1 and Task 2** - one notebook per task, driven by the `config.yaml`
next to it. Fetch the data once, then run the notebook top to bottom:

```bash
python task1_llm/shreya_akotiya/src/fetch_data.py --chars 36000000 --stream
jupyter lab task1_llm/shreya_akotiya/src/task1_char_gpt.ipynb

python task2_sentiment/shreya_akotiya/src/fetch_data.py
jupyter lab task2_sentiment/shreya_akotiya/src/task2_sentiment.ipynb
```

Shared, task-agnostic utilities live in `common/`: config loading with `extends:` inheritance,
seeding, device selection, the append-only run logger, and the metrics writers.

Runs must be config-driven. No hard-coded personal paths, credentials or API keys
anywhere in the repo.

## Where results live

| Task | Member | Status | Headline |
| --- | --- | --- | --- |
| 1 | shreya_akotiya | **complete** | val loss 0.6327 · ppl 1.88 · bpc 0.913 · top-1 79.9% |
| 1 | zoheb_waghu | **complete** (RTX 4090) | val loss 0.7019 · ppl 2.02 · bpc 1.013 · top-1 77.7% |
| 2 | shreya_akotiya | **complete** | baseline 93.14% · TextCNN 94.35% · BiLSTM **94.85%** |
| 2 | zoheb_waghu | **complete** (RTX 4090) | M1 93.46% · M2 93.57% (n.s.) · M3 **94.02%** |
| 3 | shreya_akotiya | **trained** (Tesla T4); audit, Kaggle rank pending | FID photo→Monet 80.5 · Monet→photo 84.1 · submission FID 79.15 / MiFID 0.404 |
| 3 | zoheb_waghu | **trained** (RTX 4090); audit, Kaggle rank pending | FID A2B 103.2 · B2A 98.9 · submission FID 101.03 / MiFID 0.411 |

Each member folder holds `metrics_report.csv`, `results.md` and `failure_analysis.md`.

### Spec 1.1: "training (100K) and validation (10K)"

Read as **sequences**, not characters - so at `block_size`/`seq_len` 256 with non-overlapping
windows that is ~25.6M training characters per member, not 100K.

Both members read one canonical download, `task1_llm/data/tinystories_raw.txt`, at different
`member_offset_chars`, so the two training slices are **disjoint** ("each member creates their
own split", spec 1.1.4):

| Member | Slice | Train / val |
| --- | --- | --- |
| shreya_akotiya | chars [0, 34M) | 100,000 / 10,000 sequences |
| zoheb_waghu | chars [34M, 62.2M) | 100,000 / 10,000 sequences |

zoheb_waghu's first submission mistakenly read those counts as *characters* and trained on
100,000 characters; that run is superseded and documented in
`reproducibility/manifests/zoheb_waghu/task1_llm_manifest.md`.

### Metrics schema

`metrics_report.csv` uses the **team-agreed column schema** in every member folder, so the two
members' numbers can be placed side by side in the report without reconciliation.

The team header omits some metrics the brief's "metrics to report" list requires, so
each member's folders also carry `metrics_report_extended.csv` with the full set. Each training
run writes both files from the same in-memory row, so they cannot drift.

| Task | Team header | Missing from it, kept in the extended table |
| --- | --- | --- |
| 1 | 22 cols | `config_path`, `device`, `model_name` |
| 2 | 31 cols | confusion-matrix cells (`tn`/`fp`/`fn`/`tp`), all 8 per-slice robustness columns, `split`, `config_path`, `device` |

**For the team to decide:** whether to widen the agreed Task 2 header to include the
confusion-matrix and per-slice columns, since the brief lists both as required metrics. Task 3's
header is one row per direction, which differs from the one-row-per-run shape used in Tasks 1
and 2 - worth confirming before anyone trains a CycleGAN.

## Evidence trail

- Raw training logs: `reproducibility/raw_logs/<member>/` — unedited after the run.
- Manifests: `reproducibility/manifests/<member>/` — package versions, environment
  file, and which checkpoint maps to which reported result.

## References

- Vaswani et al., *Attention Is All You Need*
- Eldan & Li, *TinyStories*
- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*
