# Task 3 — CycleGAN style transfer — results (zoheb_waghu)

> Sections marked **✍ your analysis** are left for me to write: brief section 8 requires the
> design justification and analysis to be my own. Everything else is measured.

Run `t3_baseline_20260929-235720` (RTX 4090) · config [configs/cyclegan_baseline.yaml](configs/cyclegan_baseline.yaml)
· notebook [src/task3_cyclegan.ipynb](src/task3_cyclegan.ipynb)
· raw log [t3_baseline_20260929-235720.jsonl](../../reproducibility/raw_logs/zoheb_waghu/task3_gan/t3_baseline_20260929-235720.jsonl)
· manifest [task3_gan_manifest.md](../../reproducibility/manifests/zoheb_waghu/task3_gan_manifest.md)

Convention (the instructor's evaluator): **A = Monet** (300 images), **B = photo** (7,038 images),
unpaired, 256×256. `pred_A2B` = Monet → photo, `pred_B2A` = photo → Monet (the Kaggle direction).

## What I built

Two generators (G_AB: Monet → photo, G_BA: photo → Monet) and two discriminators (D_A judges
Monet, D_B judges photos), trained jointly with an adversarial loss in both directions, a
cycle-consistency loss (x → G → G' → x) and an identity loss. Code: [src/model.py](src/model.py),
[src/data.py](src/data.py), [src/train.py](src/train.py), [src/infer.py](src/infer.py).

## Architecture and why

| Component | Choice | Parameters | Why |
|---|---|---|---|
| Generator ×2 | ResNet: 7×7 conv, 2 stride-2 downsamples, **9 residual blocks**, 2 transposed-conv upsamples, 7×7 conv, tanh; reflection padding | 11,378,179 each | ✍ your analysis |
| Discriminator ×2 | **70×70 PatchGAN**: 3 stride-2 4×4 convs (64→128→256), 1 stride-1 conv (512), 1-channel score map | 2,764,737 each | ✍ your analysis |
| Normalisation | Instance norm (generators and discriminators, not the first D layer) | - | ✍ your analysis |
| Initialisation | N(0, 0.02) | - | ✍ your analysis |
| **Total** | | **28,285,832** | |

## Hyperparameters and why

| Hyperparameter | Value | Why |
|---|---|---|
| GAN objective | LSGAN (MSE to 1 for real, 0 for fake) | ✍ your analysis |
| λ cycle (A, B) | 10, 10 | ✍ your analysis |
| λ identity | 0.5 × λ cycle | ✍ your analysis |
| Fake-image pool | 50 past fakes per discriminator | ✍ your analysis |
| Optimiser | Adam, lr 2e-4, β = (0.5, 0.999), G and D separate | ✍ your analysis |
| Schedule | 20 epochs constant + 20 epochs linear decay to 0 | ✍ your analysis |
| Epoch definition | 7,038 steps = one pass over the photos; Monet sampled with replacement | ✍ your analysis |
| Batch size | 1 | ✍ your analysis |
| Augmentation | resize to 286, random 256 crop, horizontal flip | ✍ your analysis |
| Epoch budget | 20 + 20 (the paper's 100 + 100 measured at ~35 h on this GPU) | ✍ your analysis |

## Training

40 epochs, **281,520 steps**, 6 h 41 m (24,082.6 s of training, ~602 s per epoch). Checkpoint
every epoch (resumable), permanent snapshots every 5 epochs.

| Signal | First 50 logged steps (mean) | Final 10% of steps (mean) |
|---|---|---|
| Generator loss (total) | 8.151 | 2.970 |
| Discriminator loss (D_A + D_B) | 0.483 | 0.251 |
| Cycle L1 (unweighted, both directions) | 0.497 | 0.132 |
| Identity L1 (unweighted, both directions) | 0.460 | 0.097 |
| Generator gradient norm | 44.4 | mean over run 23.36 |

- NaN / non-finite steps: **0**.
- Largest gradient-norm spikes: **527.3** at step 142,550 (epoch 21; neighbours 13.7-47.1),
  165.3 at step 218,050, 131.8 at step 36,800, 130.5 at step 148,000. No loss divergence followed any of them.
- Curves: [outputs/plots/training_curves_t3_baseline_20260929-235720.png](outputs/plots/training_curves_t3_baseline_20260929-235720.png)

Convergence and stability: ✍ your analysis

## Metrics

Full rows in [full_metrics_report.csv](full_metrics_report.csv) (team schema, both directions).
FID and MiFID are computed exactly as the instructor's `Part3_Evaluation_Script.ipynb` (300
images per set, Inception-v3) - re-checked on these predictions, identical to the last digit.

| Metric | Monet → photo (A2B) | Photo → Monet (B2A) |
|---|---|---|
| FID | 103.197 | 98.855 |
| MiFID (mean cosine distance) | 0.4181 | 0.4047 |
| KID (± std over 50 subsets of 100) | 0.0182 ± 0.0023 | 0.0076 ± 0.0013 |
| Precision / recall (k = 5) | 0.703 / 0.480 | 0.507 / 0.680 |
| Density / coverage | 1.013 / 0.897 | 0.433 / 0.690 |
| Cycle-reconstruction L1 ([0, 1] pixels) | 0.0332 | 0.0398 |
| LPIPS, input vs translation (AlexNet) | 0.354 | 0.378 |
| Content cosine, input vs translation (Inception) | 0.797 | 0.772 |
| Human audit (style / content / artifacts) | pending - 2 raters | pending |
| Inter-rater agreement (Cohen's κ) | pending | pending |
| Parameters / training time / images per s / peak memory | 28,285,832 / 24,082.6 s / 11.69 / 12.78 GB | (same run) |

**Kaggle submission** ([submission.csv](submission.csv), the instructor's format: mean of both
directions): **FID 101.026, MiFID 0.4114** → leaderboard score (FID + MiFID) / 2 = 50.72.
Public / private score and rank: pending submission.

Interpretation of the numbers (which direction is better and why, precision vs recall): ✍ your analysis

## Evidence

| What | Where |
|---|---|
| Translations, as evaluated | `outputs/pred_A2B/` (all 300), `outputs/pred_B2A/` (first 300 of 7,038 - the evaluated set; `src/infer.py` regenerates all) |
| Example triplets (input / translation / reconstruction) | [outputs/plots/translation_examples.jpg](outputs/plots/translation_examples.jpg) |
| Failure-case candidates | [outputs/plots/failure_candidates.jpg](outputs/plots/failure_candidates.jpg), [outputs/failure_candidates.csv](outputs/failure_candidates.csv) |
| Loss / gradient / LR curves | [outputs/plots/training_curves_t3_baseline_20260929-235720.png](outputs/plots/training_curves_t3_baseline_20260929-235720.png) |
| Raw training and evaluation logs | `reproducibility/raw_logs/zoheb_waghu/task3_gan/` |

## Checkpoints

| File | In repo | Notes |
|---|---|---|
| `t3_baseline_20260929-235720_generators_fp16.pt` | **yes** (43 MB) | G_AB + G_BA in float16. Regenerating the evaluated sets from it gives FID 103.301 (A2B, +0.105) and 98.896 (B2A, +0.041) |
| `t3_baseline_20260929-235720_final.pt` | no (108 MB, over GitHub's 100 MB limit) | All four networks, float32 - produced every reported number |
| `..._last.pt`, `..._epoch005.pt` … `..._epoch040.pt` | no (324 MB each) | Resumable full state (nets + optimisers) |

## Hardware disclosure

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB, driver 610.60 |
| CPU | AMD Ryzen 9 7950X (16 cores) |
| RAM | 128 GB |
| Where it ran | Local workstation, Windows 11 · Python 3.12.10 · PyTorch 2.5.1+cu124 |

## Reproduce

```bash
python task3_gan/zoheb_waghu/src/train.py --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml
python task3_gan/zoheb_waghu/src/infer.py --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml --checkpoint <run_id>_final.pt
python task3_gan/zoheb_waghu/evaluate_local.py metrics --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml --train-run-id <run_id> --checkpoint <run_id>_final.pt
```

## What I would try next

✍ your analysis
