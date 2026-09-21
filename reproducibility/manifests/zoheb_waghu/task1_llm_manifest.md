# Task 1 - Run manifest (zoheb_waghu)

Maps every reported number back to the config, checkpoint and raw log that produced it.

| run_id | config | checkpoint | raw log | env freeze | metrics row | reported in |
|---|---|---|---|---|---|---|
| `t1_baseline_20260919-232507` | `configs/gpt_baseline.yaml` | `checkpoints/t1_baseline_20260919-232507_best.pt` | `reproducibility/raw_logs/zoheb_waghu/task1_llm/t1_baseline_20260919-232507.jsonl` | `logs/env_freeze_t1_baseline_20260919-232507.txt` | `metrics_report.csv:t1_baseline_20260919-232507` | `results.md` §5, `failure_analysis.md` |
| `t1_smoke_20260919-232407` | (deleted) | (deleted) | `logs/t1_smoke_20260919-232407.jsonl` | - | removed from CSV | **not reported** - 1-epoch pipeline check. Its raw log is kept as evidence; its config, checkpoint and metrics row were removed so the graded table holds only the real run. |

## Outputs traceable to the baseline run

| Artifact | Path |
|---|---|
| Loss / perplexity / stability curves | `outputs/plots/loss_curves_t1_baseline_20260919-232507.png` |
| Generalization gap curve | `outputs/plots/generalization_gap_t1_baseline_20260919-232507.png` |
| Generated samples (3 prompts × 3 strategies) | `outputs/samples/samples_t1_baseline_20260919-232507.txt` |
| Per-epoch history, per-step losses, grad norms, causal probe | `outputs/history_t1_baseline_20260919-232507.json` |
| Character vocabulary and encoded splits | `data_processed/{vocab.json,train_ids.npy,val_ids.npy,stats.json}` |

## Environment

- Python **3.9.6 (arm64)**, PyTorch **2.5.1**, device **mps**, Apple M5 / 16 GB
- Full package list: `logs/env_freeze_t1_baseline_20260919-232507.txt`
- Git commit at run time is recorded inside the raw log (`run_start.git_commit`)

> Note: the only arm64 Python on this machine is Apple's `/usr/bin/python3` (3.9.6). Every
> Homebrew/miniforge Python present is x86_64 under Rosetta, which caps torch at 2.2.x and has
> **no MPS support at all**. The venv must be created with `/usr/bin/python3` to reproduce.

## Reproduce

```bash
/usr/bin/python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python task1_llm/zoheb_waghu/src/train.py --config task1_llm/zoheb_waghu/configs/gpt_baseline.yaml
./.venv/bin/python task1_llm/zoheb_waghu/src/plots.py --history task1_llm/zoheb_waghu/outputs/history_<run_id>.json
```
