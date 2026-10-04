# Task 3 — CycleGAN style transfer — results (shreya_akotiya)

## What I built

A CycleGAN that translates between photographs and Monet paintings without paired examples:
two generators (Photo→Monet, Monet→Photo) and two discriminators (one per domain), trained with
adversarial, cycle-consistency and identity losses on the Kaggle `gan-getting-started` data
(7,038 photos, 300 Monet paintings, all 256×256).

In my notebook, domain A = photo and domain B = Monet, so my `pred_A2B/` is Photo→Monet and
`pred_B2A/` is Monet→Photo. The instructor's evaluation script and the team metrics use the
opposite naming (A = Monet), so my `pred_A2B/` is its `pred_B2A`, and vice versa.

## Architecture and why

**Generator: UNet with skip connections** (41.8M parameters each)
- 7 encoder blocks (4×4 stride-2 conv, InstanceNorm, LeakyReLU 0.2), taking 256×256 down to a 2×2 bottleneck
- 6 decoder blocks (transposed conv, InstanceNorm, ReLU) plus a final transposed conv with Tanh;
  dropout 0.5 in the first three decoder blocks
- Each decoder block is concatenated with the matching encoder output (skip connection)
- Why UNet instead of the ResNet-9 generator in the CycleGAN paper: the skip connections carry
  edges and layout straight to the output, so the generator only has to change colour and texture,
  not rebuild the scene structure.
- The 256px run set `gen_down_blocks = 8`, but the generator code always builds 7 encoder blocks,
  so that setting had no effect. The trained model is the 7-block version described above.

**Discriminator: PatchGAN** (662K parameters each)
- 3 stride-2 conv layers (64→128→256 filters) plus a stride-1 output conv
- Outputs a 31×31 grid of real/fake scores for a 256×256 image, so each score judges a local patch
  (about 46×46 pixels) - suited to brushstroke texture rather than whole-image layout

**Total parameters:** 84,967,560

The design follows the CycleGAN paper (Zhu et al., 2017) for the losses and training scheme, and
pix2pix (Isola et al., 2017) for the UNet generator and PatchGAN discriminator.

## Hyperparameters and why

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Image size | 256px | Native resolution of the dataset; no detail thrown away |
| Epochs | 80 (1 epoch = 7,038 photo steps) | Fit within one Colab session (9.66 h) |
| LR schedule | 2e-4 constant to epoch 61, then linear decay over epochs 62–80 | Standard CycleGAN schedule, shortened to fit 80 epochs; ends at 1.9e-5, not 0 |
| Optimizer | Adam, betas (0.5, 0.999) | Standard for GANs; low beta1 damps oscillation |
| Batch size | 1 | Standard for CycleGAN with instance normalization |
| GAN loss | LSGAN (MSE) | More stable gradients than the original cross-entropy GAN loss |
| Lambda cycle | 10.0 (both directions) | Forces translations to keep the source content |
| Lambda identity | 0.5 × lambda cycle = 5.0 | Discourages unnecessary colour shifts |
| Replay buffer | 50 images | Discriminators see older fakes too, which reduces oscillation |
| Augmentation | Resize to 286, random crop 256, horizontal flip | Extra variety, important for only 300 Monet images |

## Training behaviour

From `outputs/train_history.csv` (per-epoch averages) and the raw log:

| Epoch | G loss | Avg D loss | Cycle A | Cycle B | G adversarial (Photo→Monet) | LR |
|------:|-------:|-----------:|--------:|--------:|---------------------------:|---:|
| 1 | 2.755 | 0.227 | 0.731 | 0.649 | 0.395 | 2.0e-4 |
| 10 | 2.191 | 0.163 | 0.451 | 0.449 | 0.540 | 2.0e-4 |
| 30 | 2.290 | 0.119 | 0.410 | 0.409 | 0.658 | 2.0e-4 |
| 60 | 2.379 | 0.098 | 0.410 | 0.386 | 0.688 | 2.0e-4 |
| 80 | 2.461 | 0.082 | 0.405 | 0.395 | 0.759 | 1.9e-5 |

(Cycle columns are weighted by lambda 10.)

- **Stable:** no NaN or Inf values in the log, and neither discriminator collapsed to zero.
- **Cycle consistency converged early:** cycle loss fell fast in the first 10 epochs and was flat at about
  0.41 from epoch 30 on.
- **The discriminators slowly got ahead:** after epoch 5 the generator's adversarial loss rose
  (0.40 → 0.76) while discriminator loss fell (0.23 → 0.08). Total G loss was lowest at epoch 5 (2.05).
- **Effect of the LR decay is not measured:** the decay did not reverse that trend, and FID was only measured
  once, after epoch 80, so this run cannot show whether the decay improved image quality.

## Metrics

Computed by `src/task3_cyclegan.ipynb` (saved cell output) with `clean-fid` on all generated images
(7,038 Photo→Monet, 300 Monet→Photo):

| Metric | Photo → Monet | Monet → Photo |
|--------|--------------:|--------------:|
| FID (clean-fid) | 80.55 | 84.05 |
| KID (clean-fid) | 0.0110 | 0.0213 |

### Kaggle submission (instructor's evaluation script)

Computed with the instructor's `Part3_Evaluation_Script.ipynb`: first 300 sorted images per set,
Inception-v3 (Resize 299 + CenterCrop 299), MiFID = mean cosine distance paired by index, and the
submission value = average of the two directions. Run copy with outputs:
`src/part3_evaluation_shreya.ipynb`.

| | Photo → Monet | Monet → Photo | **Submission (mean)** |
|---|---:|---:|---:|
| FID | 97.904 | 102.784 | **100.344** |
| MiFID | 0.4043 | 0.4206 | **0.4124** |

The clean-fid FIDs above are lower (80.55 / 84.05) because they use all 7,038 Photo→Monet images and
a different resize. FID gets lower as more images are used, so the two sets of numbers are not
comparable. Only the 300-image numbers match the leaderboard and teammates' submissions.

### Team-format metrics

`metrics_report.csv`, computed from the epoch-80 checkpoint and its
predictions by `src/run1_team_metrics.ipynb`. Same 300-image method as the submission (FID and
MiFID match it exactly). Team direction names: A = Monet, B = photo. The last column is the
teammate's ResNet-9 (zoheb_waghu), measured the same way.

| Metric | Photo → Monet (B2A) | Monet → Photo (A2B) | zoheb_waghu (B2A / A2B) |
|---|---:|---:|---:|
| FID | 97.904 | 102.784 | 98.855 / 103.197 |
| KID | 0.0069 | 0.0185 | 0.0076 / 0.0182 |
| Precision / recall | 0.540 / 0.730 | 0.730 / 0.447 | 0.507 / 0.680 · 0.703 / 0.480 |
| Density / coverage | 0.489 / 0.763 | 1.005 / 0.850 | |
| Cycle L1 ([0, 1] scale) | 0.0216 | 0.0223 | 0.0398 / 0.0332 |
| LPIPS input vs translation | 0.304 | 0.244 | 0.378 / 0.354 |
| Content cosine | 0.821 | 0.861 | 0.772 / 0.797 |

The two models are tied on FID and KID: about 1 FID point apart in each direction, which is within
the noise of 300-image FID. My UNet keeps content better (higher content cosine,
smaller change) and has about half the round-trip error, consistent with its skip connections. The
low cycle error partly reflects hidden information rather than faithful translation (see
`failure_analysis.md`).

Training stats from the checkpoint and raw log: 0 non-finite steps in 11,260 logged steps, final-10%
generator loss 2.439, discriminator loss 0.171 (D_A + D_B). `training_time_s` (34,775 s) is the sum
of the logged epoch times; the 34,793 s in the hardware table is the whole session.

Still to measure: human audit (30 samples, 2 raters, Cohen's kappa; sheet generated by
`src/human_audit.py`) and Kaggle public/private score and rank. Gradient norms and peak memory were
not logged during this run.

Evidence: loss curves in `outputs/plots/loss_curves.png`, per-epoch losses in
`outputs/train_history.csv`, run manifest in
`reproducibility/manifests/shreya_akotiya/task3_gan_manifest.md` (run → checkpoint → log) and
`reproducibility/manifests/shreya_akotiya/t3_shreya_unet_128_20261002_023102.json`.

| Result | Checkpoint | Raw log |
|--------|------------|---------|
| All metrics above | `t3_shreya_unet_128_20261002_023102_epoch080.pt` (1.0 GB, not in repo); committed generators `checkpoints/t3_shreya_unet_128_20261002_023102_G_{AB,BA}_fp16.pt` | `reproducibility/raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_128_20261002_023102.log` |

The run ID contains `128` because the run tag was set before the notebook switched this run to 256px.
The log confirms the run used 256px images.

## Hardware disclosure

| Item | Value |
|------|-------|
| GPU | NVIDIA A100-SXM4-40GB (Google Colab) |
| GPU memory | 40 GB |
| Framework | PyTorch 2.11.0+cu130, CUDA |
| Training time | 9.66 hours (34,793 s), about 430 s per epoch after the first |
| Throughput | ~16.2 images/sec (563,040 steps, batch size 1) |
| Where it ran | Google Colab |

## What I would try next

1. **Repeat with three seeds** to measure run-to-run variance; with 300 images per set, FID
   differences of a few points may be noise.
2. **Rebalance the discriminators** - they gradually overpowered the generators; try fewer
   discriminator updates.
3. **Longer schedule** - 100 constant + 100 decay epochs, as in the CycleGAN paper.
