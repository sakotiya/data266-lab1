# Task 3 — failure analysis (shreya_akotiya)

Submitted model: run 1 (`t3_shreya_unet_128_20261002_023102`, epoch 80), UNet generator + PatchGAN.
All scores below use the instructor's method (`Part3_Evaluation_Script.ipynb`): first 300 sorted
images per set, Inception-v3, submission = mean of the two directions, score = −(FID + MiFID) / 2.

## Visual quality

Team-format metrics for run 1 (`metrics_report_full.csv`, first 300 images per set, computed by
`src/run1_team_metrics.ipynb` from the epoch-80 checkpoint). Team names: A = Monet, B = photo.

| Measure (run 1) | Photo → Monet (B2A) | Monet → Photo (A2B) |
|---|---:|---:|
| FID / MiFID (= submission) | 97.904 / 0.4043 | 102.784 / 0.4206 |
| KID | 0.0069 ± 0.0013 | 0.0185 ± 0.0021 |
| Precision (realism) / recall (diversity) | 0.540 / 0.730 | 0.730 / 0.447 |
| Density / coverage | 0.489 / 0.763 | 1.005 / 0.850 |
| LPIPS input vs translation (how much changed) | 0.304 | 0.244 |
| Content cosine input vs translation | 0.821 | 0.861 |

Evidence: `outputs/plots/failure_candidates_B2A.jpg` and `..._A2B.jpg` (source | translation |
cycle reconstruction).

The two directions fail in opposite ways:
- **Photo→Monet: diverse but less convincing.** Recall is 0.73 but precision only 0.54. The outputs
  cover most of the variety of real Monet paintings, but many don't look like one.
- **Monet→Photo: convincing but narrow.** Precision is 0.73 but recall only 0.45. The outputs look
  like photos, but stick to a narrower range than the real photo set.
- **Monet→Photo changes the image less** (LPIPS 0.244 vs 0.304) and keeps content better (cosine
  0.861 vs 0.821). Its higher FID is therefore about missing photo variety, not about losing content.

## Cycle-consistency verification

| Measure (run 1, 300 images, [0, 1] pixel scale) | Photo → Monet → Photo | Monet → Photo → Monet |
|---|---:|---:|
| Mean cycle L1 | 0.0216 (≈ 5.5 / 255) | 0.0223 (≈ 5.7 / 255) |
| Worst image | 0.0455 (`08b790bca7.jpg`) | 0.0448 (`a619072f82.jpg`) |
| Training cycle loss, final 10% of epochs (unweighted, both directions) | 0.080 | |

Both round trips restore the source closely and are nearly symmetric. For comparison, the team's
ResNet-9 (zoheb_waghu) has about twice the error (0.040 / 0.033). The UNet's skip connections carry
full-resolution detail straight to the output, which makes reconstruction easy.

**Low cycle error does not mean faithful translations.** The clearest evidence is in the "most
changed" photos (rows 6–10 of the B2A grid). These night scenes come out as a barely recognisable
tiled pattern, yet they have the *lowest* cycle errors in the set (0.013–0.021). The reconstruction
restores the stars, city lights and corridor almost perfectly from an image in which they are hard
to see. The source information must be carried in subtle signals the eye doesn't pick up. This is
the "steganography" behaviour CycleGANs are known for (Chu, Zhmoginov and Sandler, 2017), and the
UNet's skip connections make it especially easy.

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

Candidates were selected by measurement (`src/run1_team_metrics.ipynb`) over the 300 evaluated
images per direction, then inspected by eye. *Least changed* = lowest LPIPS (translation barely
departs from the input), *most changed* = highest LPIPS, *worst cycle* = highest cycle L1. All 30
candidates: `outputs/failure_candidates.csv`; images: `outputs/plots/failure_candidates_{B2A,A2B}.jpg`.

| # | Sample | Direction | Selected as | LPIPS | Cycle L1 | What went wrong | Suspected cause |
|---|---|---|---|---:|---:|---|---|
| 1 | `09fc404e31.jpg` | Photo→Monet | most changed | 0.721 | 0.013 | A dark starry sky becomes a blotchy blue-grey field with a repeated tiled motif along the top. The reconstruction restores the night sky almost exactly. | Night photos have almost no contrast. InstanceNorm divides by a very small standard deviation, so tiny pixel noise is amplified and the decoder fills the image with learned texture. The source survives as a hidden signal (lowest cycle error in the set). |
| 2 | `07054731ab.jpg` | Photo→Monet | most changed | 0.643 | 0.014 | City lights at dusk: the same tiled band appears across the top and the sky turns to mottled texture; the lights survive. | Same as #1. The motif is identical across different night images, so it is a learned pattern, not content from the source. |
| 3 | `08341635fa.jpg` | Photo→Monet | most changed | 0.629 | 0.016 | A dark corridor is washed out to grey-green with the tiled band on top; the lit doorways are kept. | Same as #1; the bright lamps give the only strong signal, so only they are kept. |
| 4 | `02ded12bbd.jpg` | Photo→Monet | most changed | 0.623 | 0.014 | A dull sunset becomes a flat grey haze; the red horizon band disappears, though it returns in the reconstruction. | Low-detail input: a patch discriminator rewards Monet-like local texture but nothing protects the global colour of the scene. |
| 5 | `063ab57d41.jpg` | Photo→Monet | least changed | 0.120 | 0.025 | An office building and road sign stay sharply photographic, with only a pale wash. | Hard-edged modern scenes are rare in Monet's work; the identity loss (weight 5.0) makes keeping the photo cheap. |
| 6 | `04b8bfdb1c.jpg` | Photo→Monet | least changed | 0.113 | 0.024 | Cows on grass: slightly softened and faded, but clearly still a photo. | Same as #5: a small colour shift satisfies the losses. |
| 7 | `0962094f25.jpg` | Photo→Monet | least changed + worst cycle | 0.129 | 0.044 | A banana plant under a glass roof barely changes, yet its round trip is among the worst. The stock-photo watermark text is copied through. | Dense fine detail (leaves, roof grid, text) is hard to reproduce exactly even when the style change is small. |
| 8 | `08b790bca7.jpg` | Photo→Monet | worst cycle | 0.276 | 0.045 | A rope bridge in a forest gets convincing brush texture, but the reconstruction loses fine leaf detail. | High-frequency foliage is where the painted texture and the original detail collide; the cycle loss only constrains the average pixel error. |
| 9 | `b1ea5d5a7d.jpg` | Monet→Photo | most changed | 0.509 | 0.019 | A golden Houses of Parliament sunset turns almost black; the reconstruction restores the gold. | The generator maps hazy Monet light to a dark, high-contrast "photo" look. The source colour is hidden in the dark image and recovered on the way back. |
| 10 | `6a03aea8be.jpg` | Monet→Photo | most changed | 0.482 | 0.016 | A misty Parliament in blue-green becomes a murky dark field, with bright horizontal streaks across the top that also appear in the reconstruction. | The streaks are a generator artifact in flat, low-texture regions, likely from the transposed-convolution decoder. |
| 11 | `2cca56415e.jpg` | Monet→Photo | most changed | 0.465 | 0.029 | A haystack at sunset: the sky becomes a saturated orange-red flare and the field goes nearly black. | The photo domain has many high-contrast sunsets, so the generator exaggerates contrast instead of keeping Monet's soft light. |
| 12 | `a619072f82.jpg` | Monet→Photo | worst cycle | 0.268 | 0.045 | A coastal painting gets a rainbow-coloured horizontal band across the sky, which stays in the reconstruction. | Same streak artifact as #10. Because it survives the round trip, it is a fault of the generators, not of the source image. |
| 13 | `676a5a4c2e.jpg` | Monet→Photo | least changed | 0.046 | 0.022 | A dense hillside painting is returned almost unchanged; it still looks fully painted. | Busy brushwork already contains photo-like local texture, so the patch discriminator accepts it; the identity loss makes copying cheap. |
| 14 | `10c555c1b1.jpg` | Monet→Photo | least changed | 0.124 | 0.023 | A bridge with boats keeps its painted look, and the sky gets smeared horizontal streaks. | Combination of #13 (little change) and the sky streaks seen in #10 and #12. |

**Hard images are shared across models.** Zoheb's ResNet-9 failure list includes several of the
same files: `b1ea5d5a7d` (dark Parliament), `a619072f82` (coast), `6782e7cb2a` (footbridge, also in
my worst-cycle list), `063ab57d41` (building), `04f59976b5` (snowy rocks, also in my least-changed list) and
`02ded12bbd` (sunset). These images are difficult
for CycleGAN in general, not just for my architecture. What *is* specific to my UNet is the tiled
motif on night photos (#1–3) and the horizontal streaks (#10, #12, #14).

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
