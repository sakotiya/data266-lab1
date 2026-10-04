# Task 3 — failure analysis (shreya_akotiya)

Submitted model: run 1 (`t3_shreya_unet_128_20261002_023102`, epoch 80), UNet generator + PatchGAN.
All scores below use the instructor's method (`Part3_Evaluation_Script.ipynb`): first 300 sorted
images per set, Inception-v3, submission = mean of the two directions, score = −(FID + MiFID) / 2.

## Visual quality

| Measure (run 1) | Photo → Monet | Monet → Photo |
|---|---:|---:|
| FID, instructor method (300 images) | 97.904 | 102.784 |
| MiFID | 0.4043 | 0.4206 |
| LPIPS, input vs translation (200 images) | 0.316 | not measured |

Monet→Photo is the weaker direction: about 5 FID points worse than Photo→Monet. It is scored on
only 300 source paintings, so it also has the larger sampling noise.

## Cycle-consistency verification

| Measure (run 1, 200 images, [-1, 1] pixel scale) | Photo → Monet → Photo | Monet → Photo → Monet |
|---|---:|---:|
| Cycle L1 | 0.042 | 0.044 |

Both round trips restore the source closely and the two directions are nearly symmetric. Low
cycle error alone does not prove the translation looks faithful: CycleGANs can satisfy the cycle
loss by hiding source detail in subtle signals (Chu, Zhmoginov and Sandler, 2017). The UNet skip
connections make this easy, because full-resolution detail can bypass the bottleneck.

## Training stability

- No NaN or Inf values in run 1. (v2–v4 count non-finite steps in their `metrics_report.csv`.)
- In run 1, total generator loss was lowest at epoch 5 (2.05). After that the discriminators slowly
  got ahead: average D loss fell 0.23 → 0.08 while the Photo→Monet adversarial loss rose 0.40 → 0.76.
- v3 shows the same imbalance in its most extreme form: the Monet discriminator's loss fell to
  about 0.001 by epoch 6 and stayed between 0.0001 and 0.002 for the rest of the run.

## Experiments that did not help (v2, v3, v4)

After run 1 I trained three more full 80-epoch runs, each changing the settings to try to beat it.
None did.

| Run | Changes from run 1 | Real FID | Real MiFID | **Score** | vs run 1 |
|---|---|---:|---:|---:|---:|
| Run 1 (submitted) | — | 100.344 | 0.4124 | **−50.38** | — |
| v2 | DiffAugment (colour, translation, cutout); discriminator LR × 0.5; LR decay over 40 epochs instead of 20 | 108.937 | 0.4127 | **−54.67** | −4.29 |
| v3 | Identity weight 5.0 → 1.0; outermost UNet skip removed; EMA generators (decay 0.9999) | 111.379 | 0.4263 | **−55.90** | −5.52 |
| v4 | As v3, but identity weight 0.5 and LR decay over 40 epochs | 107.909 | 0.4117 | **−54.16** | −3.78 |

From v2 on, every run also logged a monitor score every 10 epochs, using the same method on photos
301–600 (images the submission is not scored on). It was used for monitoring only; every submission
is the final epoch.

| Monitor score | e10 | e20 | e30 | e40 | e50 | e60 | e70 | e80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 | −56.32 | −65.83 | −58.54 | −56.64 | −56.57 | −53.59 | −53.07 | −51.68 |
| v3 | −54.84 | −54.46 | −53.94 | −53.73 | −53.56 | −53.46 | −53.40 | −53.52 |
| v4 | −53.16 | −52.30 | −52.03 | — | −51.41 | −52.75 | −50.79 | −52.14 |

(v4's epoch-40 line was not recorded.)

### Why v2 did not help

- **Photo→Monet got worse, not better.** clean-fid FID on all generated images was 94.09 for
  Photo→Monet against run 1's 80.55, and 86.27 for Monet→Photo against run 1's 84.05.
- **It was unstable early on.** The monitor score dropped from −56.32 at epoch 10 to −65.83 at epoch
  20 before recovering. Most of the final gain came during the LR decay (−56.57 at epoch 50 to
  −51.68 at epoch 80).
- **Likely cause:** with only 300 Monet paintings, DiffAugment's colour and cutout changes, plus
  the halved discriminator learning rate, gave the Monet discriminator a weaker and noisier signal
  about what real Monet style looks like. The generators then had less to learn from.
- **Limitation:** three things changed at once, so this run cannot say which one did the damage.

### Why v3 did not help

- **Photo→Monet barely moved for 80 epochs:** 110.2 at epoch 10, 108.8 at epoch 80 on the monitor.
  All of v3's improvement came from Monet→Photo (108.3 → 104.4).
- **The Monet discriminator won too easily.** Its loss fell to about 0.001 by epoch 6 and stayed
  there. A discriminator that separates real from fake almost perfectly gives the generator very
  little useful gradient.
- **The LR decay did nothing.** The monitor score was flat from epoch 40 to 80 (−53.73 → −53.52).
- **Likely cause:** removing the outermost skip connection forced all full-resolution detail through
  the encoder. That made reconstruction harder: in the step logs the cycle loss (weighted, both
  directions) was still about 1.0 at epochs 30–40, against a per-epoch average of about 0.8 in run 1. With the identity weight still at 1.0, the
  generator kept close to its input instead of moving towards Monet style, and the discriminator
  could tell its outputs apart.
- **What it did show:** EMA weights beat the raw weights at 6 of 8 checkpoints and at the final
  epoch (−53.52 vs −53.80), so EMA was not the problem.

### Why v4 did not help

- **On the monitor it was the best run:** Photo→Monet fell to about 100–103 (97.6 at epoch 70),
  against v3's 109. The Monet discriminator stayed balanced (loss 0.02–0.15 instead of 0.001).
  Lowering the identity weight to 0.5 was the change that freed up Photo→Monet.
- **But the real score was still 3.8 points worse than run 1** (−54.16 vs −50.38).
- **The scores swung a lot between checkpoints.** Photo→Monet moved 5–8 FID points between
  consecutive monitor checks, even at low learning rates (100.1 → 105.4 → 97.6 → 102.9 over
  epochs 50–80). On 300 images, the epoch where training stops matters about as much as the
  setting being tested.
- **Likely cause:** v3 and v4 both removed the outermost skip connection, and both ended about 4–6
  points behind run 1, which kept it. The skip removal is the common factor, so it is the most
  likely reason both runs fell short. This is not proven: v3 and v4 also added EMA. The EMA
  comparison within each run (EMA ≥ raw at the final epoch) makes EMA an unlikely cause.

### Monitor vs real score

| Run | Monitor, epoch 80 | Real score | Monitor too optimistic by |
|---|---:|---:|---:|
| v2 | −51.68 | −54.67 | 2.99 |
| v3 | −53.52 | −55.90 | 2.38 |
| v4 | −52.14 | −54.16 | 2.02 |

The monitor ranked the three runs correctly (v4 > v2 > v3 on both). Its absolute value was
2.0–3.0 points too optimistic every time, so it is useful for comparing runs but not for predicting
the leaderboard number.

## Failure cases

**Pending.** Needs visual inspection of the run 1 predictions (on Google Drive). The plan is to
take the five lowest-LPIPS (least changed), five highest-LPIPS (most changed) and five worst
cycle-L1 images per direction, as in the team's protocol.

| # | Sample | Direction | What went wrong | Suspected cause |
| --- | --- | --- | --- | --- |
| 1 |  |  |  |  |

## Human audit (30 fixed samples, 2 raters)

**Pending.** The blinded sheet is generated by `src/human_audit.py make`. It produces 15 samples
per direction, picked with seed 42 and shuffled, each shown as source | translation, with file names
stripped. Raters get only `outputs/human_audit/`, `outputs/human_audit_ratings.csv` and
`outputs/RATING_GUIDE.md`, and score style, content and artifacts from 1 to 5. Then
`src/human_audit.py score` computes the means and Cohen's kappa below.

| Metric | Rater 1 | Rater 2 | Agreement |
| --- | --- | --- | --- |
| Style |  |  |  |
| Content |  |  |  |
| Artifacts |  |  |  |

Inter-rater agreement (Cohen's kappa / % agreement):

## Shortcomings

- **The best model is the first one.** Three follow-up runs, about 30 GPU hours, did not beat the
  baseline. Each changed several settings at once, so they show what doesn't work together but not
  which single change is responsible.
- **One seed per setting.** The 5–8 point swings between checkpoints in v4 suggest run-to-run
  variance of several points, about the size of every difference measured here. Without repeated
  seeds, none of the gaps between runs can be called significant.
- **300-image scoring.** FID on 300 images is noisy and biased upward; the monitor-vs-real gap of
  2–3 points between two different sets of 300 photos shows how much the image sample alone moves
  the score.
- **Domain imbalance.** 300 Monet paintings against 7,038 photos: each painting is seen about 23
  times per epoch, which makes it easy for the Monet discriminator to memorise them (most visible in
  v3).
- **Gap to the leaderboard.** The top scores are −41 to −45. My best is −50.38. The settings I
  changed moved the score by a few points at most; closing a 5–10 point gap would need a different
  approach, not more tuning of this one.

What I would test next, one change at a time: put the outermost skip back with identity weight 0.5
(isolating v4's useful change), and repeat the best setting with three seeds to measure the noise.
