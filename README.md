# DATA 266 — Lab 1 · Team 32

**GPT from scratch · Sentiment classification · CycleGAN style transfer**

**Members:** Shreya Akotiya (`shreya_akotiya`) · Zoheb Waghu (`zoheb_waghu`)
**Final report:** [`report/DATA266_Lab1_Report_Team_32.pdf`](report/DATA266_Lab1_Report_Team_32.pdf)

Each member designed, trained and evaluated their own model for every task, in their own folder.
The team agreed on shared data rules, metric formats and evaluation methods so the results can be
compared side by side.

---

## Results at a glance

| Task | shreya_akotiya | zoheb_waghu |
|---|---|---|
| **1 · GPT from scratch** | 12-layer deep-narrow GPT · **0.913 bits/char** · 79.9% top-1 | 4-layer shallow-wide GPT · 1.013 bits/char · 77.7% top-1 |
| **2 · Yelp sentiment** | mean-pool 93.14% · TextCNN 94.35% · **BiLSTM 94.85%** | BiLSTM 93.46% · TextCNN 93.57% · **BiLSTM-attention 94.02%** |
| **3 · CycleGAN** | UNet · FID 97.9 (photo→Monet) / 102.8 · **Kaggle score −50.38 (team final)** | ResNet-9 · FID 98.9 / 103.2 · Kaggle score −50.72 |

### Hardware and training time

| Task | shreya_akotiya | zoheb_waghu |
|---|---|---|
| 1 · GPT | NVIDIA Tesla T4 16 GB (Google Colab) · 176 min | NVIDIA RTX 4090 24 GB · 6.4 min |
| 2 · Sentiment | NVIDIA Tesla T4 16 GB (Google Colab) · 41 s / 248 s / 350 s (3 models) | NVIDIA RTX 4090 24 GB · 19 s / 17 s / 436 s (3 models) |
| 3 · CycleGAN | NVIDIA A100 40 GB (Google Colab) · 9.7 h | NVIDIA RTX 4090 24 GB · 6.7 h |

Zoheb's machine: AMD Ryzen 9 7950X, 128 GB RAM. Full hardware details (software versions, peak
memory, throughput) are in each member's `results.md`. Training times are not comparable across
GPUs.

---

## What we did

### Task 1 — Character-level GPT from scratch (TinyStories)
- Built a GPT with **hand-written** attention, causal mask, LayerNorm, feed-forward and residuals.
  No `nn.Transformer` or `nn.MultiheadAttention`.
- Each member used their own **non-overlapping slice** of TinyStories: 100K training / 10K
  validation sequences of 256 characters. The vocabulary is built from the training split only.
- **Shreya:** 12 layers × width 256 (9.57M params), 10 epochs on a Tesla T4.
  **Zoheb:** 4 layers × width 256 (3.27M params), 10 epochs on an RTX 4090.
- The deeper model predicts characters about 10% better (0.913 vs 1.013 bits/char). Both models
  spell well but lose meaning across sentences.

### Task 2 — Yelp Polarity sentiment classification
- Three models per member, all with **embeddings learned from scratch** (no GloVe / word2vec /
  pretrained LMs). Negation words were kept during stopword removal.
- Both members tested on the **same official 38K test set**, with full metrics: accuracy,
  P/R/F1, ROC-AUC, PR-AUC, MCC, Brier, ECE, bootstrap CIs, McNemar and per-slice robustness.
- **Shreya:** mean-pool baseline → TextCNN → BiLSTM, trained on all 540K reviews.
  **Zoheb:** BiLSTM baseline → TextCNN → BiLSTM with attention, trained on a 90K subsample.
- Models that read word order beat bag-of-words, and long, mixed reviews are the hardest for
  every model.

### Task 3 — CycleGAN Monet ↔ photo (Kaggle)
- Two generators + two discriminators, trained at 256px with adversarial, cycle-consistency and
  identity losses. Scored with the **instructor's evaluation script** (FID / MiFID, both
  directions), plus KID, precision/recall, density/coverage, cycle L1, LPIPS and content cosine.
- **Shreya:** UNet generator + PatchGAN (85.0M params), 80 epochs on an A100.
  **Zoheb:** ResNet-9 generator + 70×70 PatchGAN (28.3M params), 40 epochs on an RTX 4090.
- Image quality is effectively tied on FID. The UNet preserves content better. A blinded human
  audit (30 samples, 2 raters, Cohen's κ) was done for both models.

---

## Quick start

```bash
git clone https://github.com/sakotiya/data266-lab1.git
cd data266-lab1
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Smoke test (one command, no GPU, no data download, a few seconds):** checks that every config
loads, the run logger writes its log, and the metrics writers reject incomplete rows.

```bash
python common/smoke_test.py
```

`requirements.txt` pins the versions of zoheb_waghu's runs (Python 3.12, PyTorch 2.5.1, CUDA).
shreya_akotiya's runs used Google Colab (Tasks 1–2: Tesla T4; Task 3: A100). Her package lists are
in `task1_llm/shreya_akotiya/requirements.txt`, `task2_sentiment/shreya_akotiya/requirements.txt`
and `reproducibility/manifests/shreya_akotiya/`. Every run is config-driven; `device: auto` picks
cuda > mps > cpu.

---

## Reproducing a run

Datasets are not committed. Each task's `data/README.md` has the download command.

**zoheb_waghu** — Python scripts driven by YAML configs:

```bash
# Task 1 (~7 min on an RTX 4090; downloads TinyStories on first use)
python task1_llm/zoheb_waghu/src/train.py --config task1_llm/zoheb_waghu/configs/gpt_baseline.yaml

# Task 2 (baseline first: the experimental runs use its predictions for McNemar)
python -c "import nltk; [nltk.download(p) for p in ('stopwords', 'wordnet', 'omw-1.4')]"
for m in m1_baseline_bilstm m2_cnn_multikernel m3_bilstm_attention; do
  python task2_sentiment/zoheb_waghu/src/train.py --config task2_sentiment/zoheb_waghu/configs/$m.yaml
done

# Task 3 (~6.7 h on an RTX 4090; resumable)
python task3_gan/zoheb_waghu/src/train.py --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml
python task3_gan/zoheb_waghu/src/infer.py --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml \
    --checkpoint t3_baseline_20260929-235720_generators_fp16.pt
python task3_gan/zoheb_waghu/evaluate_local.py metrics --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml \
    --train-run-id <run_id> --checkpoint <run_id>_final.pt
```

**shreya_akotiya** — one notebook per task, driven by the `config.yaml` next to it (run top to
bottom; on Colab set `LAB1_ROOT` if the repo is not at `MyDrive/DATA266_Lab1`):

```bash
# Task 1
python task1_llm/shreya_akotiya/src/fetch_data.py --chars 36000000 --stream
jupyter lab task1_llm/shreya_akotiya/src/task1_char_gpt.ipynb

# Task 2
python task2_sentiment/shreya_akotiya/src/fetch_data.py
jupyter lab task2_sentiment/shreya_akotiya/src/task2_sentiment.ipynb

# Task 3 (train, then evaluate: instructor's method -> submission.csv, team metrics -> metrics_report.csv)
python task3_gan/shreya_akotiya/src/fetch_data.py
jupyter lab task3_gan/shreya_akotiya/src/task3_cyclegan.ipynb
python task3_gan/shreya_akotiya/evaluate_local.py metrics
python task3_gan/shreya_akotiya/src/human_audit.py make      # blinded audit sheet; then `score`
```

---

## Repository structure

```
data266-lab1/
├── README.md
├── requirements.txt
├── common/                    shared helpers: config loading, run logger, metrics writers, smoke_test.py
├── task1_llm/
│   ├── data/                  TinyStories (fetched, not committed)
│   ├── shreya_akotiya/
│   └── zoheb_waghu/
├── task2_sentiment/
│   ├── data/                  Yelp Polarity (fetched, not committed)
│   ├── shreya_akotiya/
│   └── zoheb_waghu/
├── task3_gan/
│   ├── data/                  monet_jpg/, photo_jpg/ (fetched, not committed)
│   ├── shreya_akotiya/
│   └── zoheb_waghu/
├── reproducibility/
│   ├── raw_logs/<member>/<task>/     unedited training logs
│   └── manifests/<member>/           package versions + run → checkpoint → log mapping
├── report/
│   ├── DATA266_Lab1_Report_Team_32.pdf   final combined report
│   ├── team_report.md                    its source
│   └── figures/
└── temp/                      assignment brief + instructor's Task 3 evaluation notebook
```

Inside every member folder:

```
<member>/
├── src/                       code (executed notebooks with outputs, and scripts)
├── config.yaml | configs/     run configuration
├── checkpoints/               trained weights
├── outputs/                   samples, plots, predictions, confusion matrices
├── metrics_report.csv         every required metric (same columns for both members)
├── failure_analysis.md        failure / error analysis
└── results.md                 what was built, why, metrics, hardware
```

Task 3 folders also contain `evaluate_local.py`, `submission.csv` (Kaggle) and
`full_metrics_report.csv`.

---

## Where to find what

| Looking for | Where |
|---|---|
| Combined team report (comparison tables, joint analysis, all snippets) | `report/DATA266_Lab1_Report_Team_32.pdf` |
| A member's write-up: architecture, hyperparameters, metrics, hardware | `<task>/<member>/results.md` |
| Failure cases (Task 1), 20-error review (Task 2), image failures (Task 3) | `<task>/<member>/failure_analysis.md` |
| All metrics for a model | `<task>/<member>/metrics_report.csv` (Tasks 1–2 also `metrics_report_extended.csv`) |
| Trained weights | `<task>/<member>/checkpoints/` |
| Loss curves, samples, confusion matrices, predictions | `<task>/<member>/outputs/` |
| Kaggle submission | `task3_gan/<member>/submission.csv` |
| Human audit (Task 3): sheets, ratings, kappa | `task3_gan/<member>/outputs/human_audit*` |
| Raw training logs | `reproducibility/raw_logs/<member>/<task>/` |
| Which checkpoint and log produced each number | `reproducibility/manifests/<member>/` |
| Notebook / code for a run | `<task>/<member>/src/` |

### Quick links

| | shreya_akotiya | zoheb_waghu |
|---|---|---|
| Task 1 · GPT | [results](task1_llm/shreya_akotiya/results.md) · [failure analysis](task1_llm/shreya_akotiya/failure_analysis.md) · [metrics](task1_llm/shreya_akotiya/metrics_report.csv) · [folder](task1_llm/shreya_akotiya/) | [results](task1_llm/zoheb_waghu/results.md) · [failure analysis](task1_llm/zoheb_waghu/failure_analysis.md) · [metrics](task1_llm/zoheb_waghu/metrics_report.csv) · [folder](task1_llm/zoheb_waghu/) |
| Task 2 · Sentiment | [results](task2_sentiment/shreya_akotiya/results.md) · [failure analysis](task2_sentiment/shreya_akotiya/failure_analysis.md) · [metrics](task2_sentiment/shreya_akotiya/metrics_report.csv) · [folder](task2_sentiment/shreya_akotiya/) | [results](task2_sentiment/zoheb_waghu/results.md) · [failure analysis](task2_sentiment/zoheb_waghu/failure_analysis.md) · [metrics](task2_sentiment/zoheb_waghu/metrics_report.csv) · [folder](task2_sentiment/zoheb_waghu/) |
| Task 3 · CycleGAN | [results](task3_gan/shreya_akotiya/results.md) · [failure analysis](task3_gan/shreya_akotiya/failure_analysis.md) · [metrics](task3_gan/shreya_akotiya/metrics_report.csv) · [folder](task3_gan/shreya_akotiya/) · [Kaggle submission](task3_gan/shreya_akotiya/submission.csv) | [results](task3_gan/zoheb_waghu/results.md) · [failure analysis](task3_gan/zoheb_waghu/failure_analysis.md) · [metrics](task3_gan/zoheb_waghu/metrics_report.csv) · [folder](task3_gan/zoheb_waghu/) · [Kaggle submission](task3_gan/zoheb_waghu/submission.csv) |
| Logs and manifests | [raw logs](reproducibility/raw_logs/shreya_akotiya/) · [manifests](reproducibility/manifests/shreya_akotiya/) | [raw logs](reproducibility/raw_logs/zoheb_waghu/) · [manifests](reproducibility/manifests/zoheb_waghu/) |

**Team report:** [PDF](report/DATA266_Lab1_Report_Team_32.pdf) · [source](report/team_report.md)

---

## Notes

- **Task 1 split.** "Training (100K) and validation (10K)" is read as **sequences** of 256
  characters (about 25.6M training characters). Members read disjoint slices of the same file:
  shreya_akotiya chars [0, 34M), zoheb_waghu from 34M.
- **Task 2 dataset.** Yelp Polarity (560K train / 38K test), confirmed with the instructor.
- **Task 3 direction names.** The instructor's script uses A = Monet, B = photo, so `A2B` =
  Monet → photo and `B2A` = photo → Monet (the Kaggle direction). shreya_akotiya's notebook uses
  A = photo, so her `outputs/pred_A2B/` is photo → Monet; her metric files use the team names.
- **Kaggle score** = −(FID + MiFID) / 2, single leaderboard (no public/private split); less
  negative is better.
- **Pretrained networks for measurement only.** Inception (FID / KID / MiFID) and AlexNet (LPIPS)
  are used only to score images, as confirmed by the instructor. All models are trained from
  scratch, and submitted images come directly from each member's own CycleGAN.
- **Metrics files.** `metrics_report.csv` uses the same team-agreed columns for both members.
  Extra columns are in `metrics_report_extended.csv` (Tasks 1–2). Values that were not measured
  are written as `NOT_MEASURED`, never left blank.

> Per brief section 8, Claude or any similar AI assistant must not make the core architecture
> decisions or write the analysis - those must be each member's own understanding, defended at
> the viva. Session transcripts and assistant logs are gitignored and kept outside this repo.

---

## References

- Vaswani et al. (2017), *Attention Is All You Need*
- Eldan & Li (2023), *TinyStories*
- Zhang, Zhao & LeCun (2015), *Character-level Convolutional Networks for Text Classification* (Yelp Polarity)
- Kim (2014), *Convolutional Neural Networks for Sentence Classification*
- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*
- Bahdanau, Cho & Bengio (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*
- Zhu et al. (2017), *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (CycleGAN)
- Isola et al. (2017), *Image-to-Image Translation with Conditional Adversarial Networks* (UNet, PatchGAN)
- Heusel et al. (2017), *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID)
