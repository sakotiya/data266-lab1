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
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # TODO: add once the team agrees on versions
```

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

<!-- TODO: one documented command that runs a smoke test of ONE member's run.
     This is graded (section 5): a grader must be able to clone and run it. -->

```bash
# e.g. python task1_llm/shreya_akotiya/src/train.py --config <config> --smoke-test
```

Runs must be config-driven. No hard-coded personal paths, credentials or API keys
anywhere in the repo.

## Where results live

| Task | Member | Metrics | Write-up | Failure analysis |
| --- | --- | --- | --- | --- |
| 1 | shreya_akotiya | `task1_llm/shreya_akotiya/metrics_report.csv` | `results.md` | `failure_analysis.md` |
| 1 | zoheb_waghu | `task1_llm/zoheb_waghu/metrics_report.csv` | `results.md` | `failure_analysis.md` |
| 2 | shreya_akotiya | `task2_sentiment/shreya_akotiya/metrics_report.csv` | `results.md` | `failure_analysis.md` |
| 2 | zoheb_waghu | `task2_sentiment/zoheb_waghu/metrics_report.csv` | `results.md` | `failure_analysis.md` |
| 3 | shreya_akotiya | `task3_gan/shreya_akotiya/full_metrics_report.csv` | `results.md` | `failure_analysis.md` |
| 3 | zoheb_waghu | `task3_gan/zoheb_waghu/full_metrics_report.csv` | `results.md` | `failure_analysis.md` |

## Evidence trail

- Raw training logs: `reproducibility/raw_logs/<member>/` — unedited after the run.
- Manifests: `reproducibility/manifests/<member>/` — package versions, environment
  file, and which checkpoint maps to which reported result.

## References

- Vaswani et al., *Attention Is All You Need*
- Eldan & Li, *TinyStories*
- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*
