# Task 1 - Run manifest (shreya_akotiya)

Maps every reported number back to the config, checkpoint and raw log that produced it.

| run_id | status | checkpoint | raw log | manifest |
|---|---|---|---|---|
| `task1_shreya_deep_narrow_20260920_194002` | **reported** | `checkpoints/task1_shreya_deep_narrow_20260920_194002_best.pt` (sha256 `f540786b...`) | `raw_logs/shreya_akotiya/task1_llm/task1_shreya_deep_narrow_20260920_194002.log` | `task1_shreya_deep_narrow_20260920_194002.json` |
| `task1_shreya_deep_narrow_20260918_231117` | not reported | not included | `raw_logs/shreya_akotiya/task1_llm/task1_shreya_deep_narrow_20260918_231117.log` | none - run stopped before the manifest step |
| `task1_shreya_deep_narrow_20260920_193016` | not reported | none | `raw_logs/shreya_akotiya/task1_llm/task1_shreya_deep_narrow_20260920_193016.log` | none - stopped during data prep |
| `task1_shreya_deep_narrow_20260920_192324` | not reported | none | `raw_logs/shreya_akotiya/task1_llm/task1_shreya_deep_narrow_20260920_192324.log` | none - stopped during data prep |

The 20260918 run used the same config and finished all 10 epochs (val CE 0.6334); the
20260920_194002 rerun is the one reported because it produced the full set of outputs and the
manifest. Its val CE is 0.6327.

Config: `task1_llm/shreya_akotiya/config.yaml`. Environment: Colab, Tesla T4, CUDA, torch
2.11.0+cu128; full package list in the manifest JSON (`pip_freeze`) and `requirements.txt`.

Logs and manifests are unedited. They record paths as `task1_llm/shreya/...` because the runs
happened before the folder was renamed to `shreya_akotiya`.
