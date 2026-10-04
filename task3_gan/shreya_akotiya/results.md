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

Computed by the notebook with `clean-fid` on all generated images (7,038 Photo→Monet, 300 Monet→Photo):

| Metric | Photo → Monet | Monet → Photo |
|--------|--------------:|--------------:|
| FID (clean-fid) | 80.55 | 84.05 |
| KID (clean-fid) | 0.0110 | 0.0213 |
| Cycle L1 (200 images, [-1, 1] pixel scale) | 0.042 (photo→Monet→photo) | 0.044 (Monet→photo→Monet) |
| LPIPS, input vs translation (200 images) | 0.316 | not measured |

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

Same method, teammate's model (zoheb_waghu, ResNet-9): FID 98.855 / 103.197, submission 101.026;
MiFID 0.4047 / 0.4181, submission 0.4114. My UNet is about 1 FID point lower in each direction,
and MiFID is almost the same. With 300 images per set, that difference is within run-to-run noise,
so the two models are effectively tied on these metrics.

### Team-format metrics

`metrics_report.csv` / `metrics_report_full.csv`, computed afterwards from the epoch-80 checkpoint
and the saved predictions by `src/run1_team_metrics.ipynb` (no retraining). Same 300-image method;
its FID and MiFID reproduce the submission exactly. Team direction names: A = Monet, B = photo.

| Metric | Photo → Monet (B2A) | Monet → Photo (A2B) | zoheb_waghu (B2A / A2B) |
|---|---:|---:|---:|
| FID | 97.904 | 102.784 | 98.855 / 103.197 |
| KID | 0.0069 | 0.0185 | 0.0076 / 0.0182 |
| Precision / recall | 0.540 / 0.730 | 0.730 / 0.447 | 0.507 / 0.680 · 0.703 / 0.480 |
| Density / coverage | 0.489 / 0.763 | 1.005 / 0.850 | |
| Cycle L1 ([0, 1] scale) | 0.0216 | 0.0223 | 0.0398 / 0.0332 |
| LPIPS input vs translation | 0.304 | 0.244 | 0.378 / 0.354 |
| Content cosine | 0.821 | 0.861 | 0.772 / 0.797 |

The two models are tied on FID and KID. My UNet keeps content better (higher content cosine,
smaller change) and has about half the round-trip error, consistent with its skip connections. The
low cycle error partly reflects hidden information rather than faithful translation (see
`failure_analysis.md`).

Training stats from the checkpoint and raw log: 0 non-finite steps in 11,260 logged steps, final-10%
generator loss 2.439, discriminator loss 0.171 (D_A + D_B), 34,775 s of epoch time (16.2 images/s).

Still to measure: human audit (30 samples, 2 raters, Cohen's kappa; sheet generated by
`src/human_audit.py`) and Kaggle public/private score and rank. Gradient norms and peak memory were
not logged during this run.

## Follow-up runs (v2, v3, v4) — none beat run 1

I trained three more 80-epoch runs to try to improve on run 1. Each was scored with the same
instructor method on its final epoch; no checkpoint was picked after seeing results. Run 1 is still
the best and stays the Kaggle submission. Only run 1's notebook (`src/task3_cyclegan.ipynb`) is in
the repo; the follow-up runs are described in full in `failure_analysis.md`.

| Run | Changes from run 1 | FID | MiFID | **Score** |
|---|---|---:|---:|---:|
| **Run 1** | — | 100.344 | 0.4124 | **−50.38** |
| v2 | DiffAugment; discriminator LR × 0.5; 40 + 40 epoch schedule | 108.937 | 0.4127 | −54.67 |
| v3 | Identity weight 1.0; outermost UNet skip removed; EMA generators | 111.379 | 0.4263 | −55.90 |
| v4 | Identity weight 0.5; outermost skip removed; EMA; 40 + 40 schedule | 107.909 | 0.4117 | −54.16 |

What they showed (details and per-epoch monitor scores in `failure_analysis.md`):
- **Lowering the identity weight helped Photo→Monet during training** (monitor FID about 101–103 in
  v4 against 109 in v3), but the gain did not carry over to the submitted 300 photos.
- **Removing the outermost skip connection is the most likely reason v3 and v4 fell short.** It is
  the change they share, and run 1, which kept it, beat both.
- **DiffAugment with a weaker discriminator (v2) made Photo→Monet worse** (clean-fid 94.09 vs 80.55).
- **EMA weights were at least as good as the raw weights** at the final epoch in v3 and v4.
- **The training monitor** (same method, photos 301–600) ranked the runs correctly, but was 2.0–3.0
  points more optimistic than the real score every time.
- **Scores swing between checkpoints:** v4's Photo→Monet FID moved 5–8 points between consecutive
  10-epoch checks. Single-seed differences of a few points between runs are within that noise.

Leaderboard context: the top scores are −41 to −45. My best honest score, −50.38, is about 5–10
points behind. The settings changed in v2–v4 moved the score by a few points at most, so closing
that gap would take a different approach rather than more tuning of this one.

Evidence: loss curves in `outputs/plots/loss_curves.png`, per-epoch losses in
`outputs/train_history.csv`, manifest in
`reproducibility/manifests/shreya_akotiya/t3_shreya_unet_128_20261002_023102.json`.

| Result | Checkpoint | Raw log |
|--------|------------|---------|
| All metrics above | `t3_shreya_unet_128_20261002_023102_epoch080.pt` | `reproducibility/raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_128_20261002_023102.log` |

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

1. **Isolate the one useful change.** Keep the outermost skip connection (as in run 1) and only
   lower the identity weight to 0.5, since that is what improved Photo→Monet in v4.
2. **Repeat with three seeds.** Checkpoint-to-checkpoint swings of 5–8 FID points mean single-run
   differences of a few points cannot be called significant.
3. **Rebalance the Monet discriminator** without weakening its signal (v2's approach hurt), e.g.
   fewer discriminator updates instead of a lower learning rate.
4. **Longer schedule** - 100 constant + 100 decay epochs, as in the CycleGAN paper.
