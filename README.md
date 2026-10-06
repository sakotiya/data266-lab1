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
report/             DATA266_Lab1_Report_Team_32.pdf (final report) + team_report.md (its source)
```

Per member, per task:

```
<member>/
  src/                 code (.ipynb with outputs)
  data_processed/      your own preprocessing output - never shared
  checkpoints/         your trained weights
  outputs/             samples, predictions, plots, confusion matrices
  config.yaml | configs/   run configuration (runs are config-driven)
  metrics_report.csv   team-format metrics (same columns for both members)
  metrics_report_extended.csv   full metric set where the team header is narrower (Tasks 1-2)
  failure_analysis.md  required failure/error write-up
  results.md           architecture + hyperparameter justification, metrics, hardware
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

shreya_akotiya's runs used Google Colab: Tasks 1-2 on a Tesla T4 (Python 3.13, torch
2.11.0+cu128), Task 3 on an A100 40GB (torch 2.11.0+cu130). Exact package lists are in
`task1_llm/shreya_akotiya/requirements.txt`, `task2_sentiment/shreya_akotiya/requirements.txt`
and the run manifests under `reproducibility/manifests/shreya_akotiya/`.

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

- [x] **GPU Lab booking and Kaggle competition registration** - done
- [x] **Per-member architectures and data splits agreed** - no two models match (see each task's
      `results.md` and the comparison tables in `report/team_report.md`)
- [x] **Task 2 dataset:** confirmed - **Yelp polarity** (560K train / 38K test). Both members
      use it.
- [x] **Task 3 submission format:** resolved - the class competition takes `submission.csv`
      (ID, FID, MiFID; values must match the instructor's script). Leaderboard score =
      (FID + MiFID) / 2, lower is better
- [x] **Measurement-only pretrained nets: confirmed allowed by the instructor.** Pretrained
      Inception (FID / MiFID / KID) and AlexNet/VGG (LPIPS) may be used **to measure images
      only**. The CycleGAN itself must be trained from scratch and the submitted images must come
      directly from it. Both members comply: pretrained networks appear only in the evaluation
      scripts, never in the model, training or inference path
- [x] **Metric headers decided** - Tasks 1-2 keep the team header plus
      `metrics_report_extended.csv`; Task 3 uses one row per direction for both members
      (see "Metrics schema" below)
- [x] **Task 3 human audit** - done for both members: 30 blinded samples, 2 raters, shared
      rubric (`outputs/RATING_GUIDE.md`); Cohen's kappa 0.500 (shreya_akotiya) and 0.149
      (zoheb_waghu). See each member's `failure_analysis.md` and `report/team_report.md` §3.3
- [x] **Task 3 Kaggle score and leaderboard rank** - recorded for both members (single
      leaderboard, no public/private split; team rank 1). See `report/team_report.md` §3.3
- [x] **shreya_akotiya Task 3 checkpoint** - fp16 generators committed
      (`task3_gan/shreya_akotiya/checkpoints/*_G_AB_fp16.pt`, `*_G_BA_fp16.pt`)
- [x] **Final report** - `report/DATA266_Lab1_Report_Team_32.pdf`, built from
      `report/team_report.md`; all three tasks plus an appendix with each member's failure/error
      snippets

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

On Colab the Task 1 and Task 2 notebooks look for the repo at `/content/drive/MyDrive/DATA266_Lab1`
by default; set the `LAB1_ROOT` environment variable to use a different location.

**shreya_akotiya's Task 3** - data via Kaggle (see `task3_gan/data/README.md`; needs
`~/.kaggle/kaggle.json` and a joined competition). Train in the first notebook, then run
the evaluation (instructor's method -> `submission.csv`, team metrics -> `metrics_report.csv`)
with `evaluate_local.py`, which executes the two evaluation notebooks in order:

```bash
python task3_gan/shreya_akotiya/src/fetch_data.py
jupyter lab task3_gan/shreya_akotiya/src/task3_cyclegan.ipynb            # train, translate, clean-fid metrics
python task3_gan/shreya_akotiya/evaluate_local.py metrics                # both evaluation notebooks -> submission.csv, metrics_report.csv
python task3_gan/shreya_akotiya/evaluate_local.py check                  # files complete and consistent
python task3_gan/shreya_akotiya/src/human_audit.py make                  # blinded audit sheet for 2 raters
python task3_gan/shreya_akotiya/src/human_audit.py score                 # rating means + Cohen's kappa
```

`config.yaml` holds the values the reported run `t3_shreya_unet_128_20261002_023102` trained with
(**256px, 80 epochs**). The run ID keeps "128" from an earlier plan; the run itself is 256px. The
committed fp16 generators (`checkpoints/*_G_AB_fp16.pt`, `*_G_BA_fp16.pt`) are enough for
inference without retraining. See `task3_gan/shreya_akotiya/results.md`.

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
| 3 | shreya_akotiya | **complete** (A100 40GB) | FID A2B 102.8 · B2A 97.9 · submission FID 100.34 / MiFID 0.412 (score 50.38) · audit κ 0.500 |
| 3 | zoheb_waghu | **complete** (RTX 4090) | FID A2B 103.2 · B2A 98.9 · submission FID 101.03 / MiFID 0.411 (score 50.72) · audit κ 0.149 |

| What | Where |
| --- | --- |
| Write-up (architecture, hyperparameters, metrics, hardware) | `<task>/<member>/results.md` |
| Failure / error analysis | `<task>/<member>/failure_analysis.md` |
| Team-format metrics | `<task>/<member>/metrics_report.csv` |
| Extended metrics | Tasks 1-2: `metrics_report_extended.csv` · Task 3: `full_metrics_report.csv` (both members; same team-schema rows as `metrics_report.csv`) |
| Checkpoints | `<task>/<member>/checkpoints/` |
| Plots, samples, predictions | `<task>/<member>/outputs/` |
| Kaggle submission | `task3_gan/<member>/submission.csv` |
| Raw training logs (unedited) | `reproducibility/raw_logs/<member>/<task>/` |
| Manifests (versions, run -> checkpoint -> log) | `reproducibility/manifests/<member>/` |
| Team report | `report/DATA266_Lab1_Report_Team_32.pdf` (source: `report/team_report.md`) |

**Task 3 direction names.** The instructor's script and the team metrics use A = Monet,
B = photo, so `A2B` = Monet -> photo and `B2A` = photo -> Monet (the Kaggle direction).
zoheb_waghu's folders follow this. shreya_akotiya's notebook uses A = photo, so her
`outputs/pred_A2B/` is the team's `B2A` and vice versa; her `metrics_report.csv` uses the team
names.

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
| 3 | 30 cols, **one row per direction** (A2B, B2A) | MiFID (per direction in each member's `results.md`; mean in `submission.csv`) |

**Decided:** the Task 2 team header stays as agreed; confusion matrices and per-slice metrics
live in each member's extended CSV and `results.md`. Task 3 uses one row per direction for both
members. Measurements not taken are written as `NOT_MEASURED`, never left blank.

## Evidence trail

- Raw training logs: `reproducibility/raw_logs/<member>/` — unedited after the run.
- Manifests: `reproducibility/manifests/<member>/` — package versions, environment
  file, and which checkpoint maps to which reported result.

## References

- Vaswani et al., *Attention Is All You Need*
- Eldan & Li, *TinyStories*
- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*
- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks* (pix2pix: UNet, PatchGAN)
- Zhang, Zhao & LeCun, *Character-level Convolutional Networks for Text Classification* (Yelp Polarity)
- Kim, *Convolutional Neural Networks for Sentence Classification*
- Hochreiter & Schmidhuber, *Long Short-Term Memory*
- Bahdanau, Cho & Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*
- Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID)
