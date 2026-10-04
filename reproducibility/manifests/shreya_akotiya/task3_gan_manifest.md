# Task 3 - Run manifest (shreya_akotiya)

Maps every reported number back to the config, checkpoint and raw log that produced it.

| run_id | status | checkpoint | raw log | manifest |
|---|---|---|---|---|
| `t3_shreya_unet_128_20261002_023102` | **reported** (run 1) | `checkpoints/t3_shreya_unet_128_20261002_023102_epoch080.pt` (local, 1.0 GB, sha256 `6ffb6155...`) | `raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_128_20261002_023102.log` | `t3_shreya_unet_128_20261002_023102.json` |
| `t3_shreya_unet_256_v2` | not reported (worse) | `t3_shreya_unet_256_v2_latest.pt`, epoch 80 (Drive only, sha256 `66956bb7...`) | `raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_256_v2.log` | `t3_shreya_unet_256_v2.json` |
| `t3_shreya_unet_256_v3` | not reported (worse) | `t3_shreya_unet_256_v3_latest.pt`, epoch 80 (Drive only, sha256 `4a945030...`) | `raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_256_v3.log` (epochs 75–80 only) | `t3_shreya_unet_256_v3.json` |
| `t3_shreya_unet_256_v4` | not reported (worse) | `t3_shreya_unet_256_v4_latest.pt`, epoch 80 (Drive only, not hashed) | `raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_256_v4.log` (epochs 61–80 only) | none - not saved |

The v2–v4 runs are described in `task3_gan/shreya_akotiya/failure_analysis.md` ("Other models
tried"). Their per-epoch FID history is in
`raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_256_v{2,3,4}_fid_monitor.csv`. v3 and v4 were
resumed after Colab disconnects; their logs kept only the last session, but the FID history files
cover all 80 epochs.

## Reported numbers → source

| Number | Value | Produced by | Stored in |
|---|---|---|---|
| Submission FID / MiFID | 100.34379 / 0.41244 | instructor's `Part3_Evaluation_Script.ipynb`, run copy `task3_gan/shreya_akotiya/src/part3_evaluation_shreya.ipynb` | `task3_gan/shreya_akotiya/submission.csv` |
| Team-format metrics (both directions) | FID 97.904 / 102.784, KID, P/R, D/C, cycle L1, LPIPS, content cosine | `task3_gan/shreya_akotiya/src/run1_team_metrics.ipynb` on the epoch-80 checkpoint | `task3_gan/shreya_akotiya/metrics_report.csv` |
| clean-fid FID / KID, all images | 80.55 / 84.05 | `task3_gan/shreya_akotiya/src/task3_cyclegan.ipynb` (saved cell output) | `t3_shreya_unet_128_20261002_023102.json` (`metrics_summary`) |
| Training losses per epoch | — | `task3_cyclegan.ipynb` | `task3_gan/shreya_akotiya/outputs/train_history.csv` |
| Gradient norms per step, peak GPU memory | grad_G mean 38.40, peak 1.78 GB | `task3_gan/shreya_akotiya/src/run1_grad_diagnostic.ipynb` on the epoch-80 checkpoint | `raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_128_20261002_023102_graddiag.{csv,json}` |
| Failure candidates | 15 per direction | `run1_team_metrics.ipynb` | `task3_gan/shreya_akotiya/outputs/failure_candidates.csv`, `outputs/plots/failure_candidates_{A2B,B2A}.jpg` |

The submission FID/MiFID were recomputed by `run1_team_metrics.ipynb` from the checkpoint and match
`submission.csv` to the third decimal in each direction.

## Outputs traceable to run 1

| Artifact | Path | In repo |
|---|---|---|
| Photo → Monet translations | `task3_gan/shreya_akotiya/outputs/pred_A2B/` | first 300 of 7,038 (the evaluated set) |
| Monet → photo translations | `task3_gan/shreya_akotiya/outputs/pred_B2A/` | all 300 |
| Generators, fp16 (what inference needs) | `task3_gan/shreya_akotiya/checkpoints/t3_shreya_unet_128_20261002_023102_G_AB_fp16.pt` (photo→Monet, sha256 `2ce06ca5...`) and `..._G_BA_fp16.pt` (Monet→photo, sha256 `5e7b55c7...`) | yes, 84 MB each |
| Full checkpoint (all four networks, optimizers, history) | `task3_gan/shreya_akotiya/checkpoints/t3_shreya_unet_128_20261002_023102_epoch080.pt` | no (1.0 GB, over GitHub's 100 MB limit) |

The fp16 files were cut from the full epoch-80 checkpoint (`{"run_id", "epoch", "source", "G_AB" or
"G_BA": state_dict}`). Loaded back as fp32, they give the same images as the full checkpoint (mean
difference 0.01 / 255), and they reproduce the committed predictions to about 3 / 255 (JPEG and
CPU-vs-GPU rounding). Load with the `UNetGenerator` class in `src/run1_team_metrics.ipynb`.

Direction names: in my notebooks A = photo and B = Monet, so `pred_A2B` is Photo→Monet. The
instructor's script and the team metrics use A = Monet; `metrics_report.csv` uses the team names.

## Data

Kaggle `gan-getting-started`: `task3_gan/data/photo_jpg/` (7,038 photos) and
`task3_gan/data/monet_jpg/` (300 Monet paintings), all 256×256.

Config: `task3_gan/shreya_akotiya/config.yaml` (the values run 1 trained with). Environment: Colab,
NVIDIA A100-SXM4-40GB, CUDA, torch 2.11.0+cu130, Python 3.13.

Logs and manifest JSONs are unedited. Run 1's run_id contains `128` because the run tag was set
before the notebook switched the run to 256px; the log confirms 256px. Run 1's JSON manifest has
`checkpoint: null` because the notebook looked for the checkpoint under the wrong tag; the sha256
above was computed from the checkpoint file itself.
