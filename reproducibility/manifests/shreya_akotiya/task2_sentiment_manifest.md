# Task 2 - Run manifest (shreya_akotiya)

Maps every reported number back to the config, checkpoint and raw log that produced it.
One run trains all three models.

| run_id | status | checkpoints | raw log | manifest |
|---|---|---|---|---|
| `task2_shreya_main_20260921_222011` | **reported** | `checkpoints/{baseline_meanpool,exp_textcnn,exp_bilstm}.pt` (sha256 in the manifest) | `raw_logs/shreya_akotiya/task2_sentiment/task2_shreya_main_20260921_222011.log` | `task2_shreya_main_20260921_222011.json` |
| `task2_shreya_main_20260922_015716` | not reported | not included (same filenames as the reported run) | `raw_logs/shreya_akotiya/task2_sentiment/task2_shreya_main_20260922_015716.log` | `task2_shreya_main_20260922_015716.json` |

The 20260922 run is a rerun with the same config. Baseline and BiLSTM are identical; TextCNN
test accuracy moved from 0.9435 to 0.9429.

Config: `task2_sentiment/shreya_akotiya/config.yaml`. Environment: Colab, Tesla T4, CUDA, torch
2.11.0+cu128, Python 3.13; full package list in the manifest JSON and `requirements.txt`.

Logs and manifests are unedited. They record paths as `task2_sentiment/shreya/...` because the
runs happened before the folder was renamed to `shreya_akotiya`.
