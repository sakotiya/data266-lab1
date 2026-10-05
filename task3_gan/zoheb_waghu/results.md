# Task 3 — CycleGAN style transfer — results (zoheb_waghu)

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
| Generator ×2 | ResNet: 7×7 conv, 2 stride-2 downsamples, **9 residual blocks**, 2 transposed-conv upsamples, 7×7 conv, tanh; reflection padding | 11,378,179 each | The nine residual blocks repeatedly transform texture and colour at 64×64 while skip connections make it easier to retain scene structure. Fewer blocks would reduce compute, but would also give the generator less depth for the domain mapping. Reflection padding reduces artificial borders. |
| Discriminator ×2 | **70×70 PatchGAN**: 3 stride-2 4×4 convs (64→128→256), 1 stride-1 conv (512), 1-channel score map | 2,764,737 each | Each score covers a local 70×70 region, so the discriminator focuses on brush texture, edges and other repeated details without requiring paired global layouts. Its limited field of view also means that locally plausible texture can coexist with poor global colour or structure. |
| Normalisation | Instance norm (generators and discriminators, not the first D layer) | - | Batch statistics are unreliable at batch size 1. Instance norm instead uses each image's channel statistics, which suits the training setup and helps the network alter style. Removing those statistics can also shift colour and contrast. |
| Initialisation | N(0, 0.02) | - | Small, zero-centred convolution weights limit extreme initial activations and give both networks usable gradients at the start of training. |
| **Total** | | **28,285,832** | |

## Hyperparameters and why

| Hyperparameter | Value | Why |
|---|---|---|
| GAN objective | LSGAN (MSE to 1 for real, 0 for fake) | MSE gives a gradient proportional to the score error, while the gradient from BCE with logits is bounded. This can provide a stronger correction when a generated sample is scored far from the real target. The logged generator norm combines adversarial, cycle and identity gradients, so it cannot attribute any observed spike to LSGAN. |
| λ cycle (A, B) | 10, 10 | Cycle loss provides the content constraint in this unpaired setting. If it is too weak, geometry can drift or several inputs can map to similar outputs. If it is too strong, the safest solution is close to copying the input. A weight of 10 keeps reconstruction influential while leaving room for the adversarial objective. |
| λ identity | 0.5 × λ cycle | Passing target-domain images through the matching generator penalises unnecessary colour and tone changes. It reduces that risk, although the orange-cloud example shows that it does not guarantee colour preservation. |
| Fake-image pool | 50 past fakes per discriminator | The pool mixes recent and older outputs, so each discriminator does not adapt only to the generator's latest artifacts. This slows rapid feedback between the two networks. |
| Optimiser | Adam, lr 2e-4, β = (0.5, 0.999), G and D separate | Setting β1 to 0.5 shortens the momentum history, which helps each optimiser respond to the other network's changing objective. Separate optimisers keep the generator and discriminator updates independent. |
| Schedule | 20 epochs constant + 20 epochs linear decay to 0 | The fixed phase allows full-size updates before the learning rate is reduced gradually. Scores change little during epochs 20–40, but that interval is also the decay phase. Since no run kept the learning rate constant beyond epoch 20, these results do not show whether decay was better than continued constant-rate training. |
| Epoch definition | 7,038 steps = one pass over the photos; Monet sampled with replacement | This exposes every photo once per epoch, but repeats each of the 300 Monet images about 23 times. It avoids discarding photos while making the domain imbalance explicit. |
| Batch size | 1 | This keeps memory use manageable and fits instance normalisation. It also produces variable per-image gradients, which the fake-image pool helps smooth. |
| Augmentation | resize to 286, random 256 crop, horizontal flip | Crops and flips provide more spatial variation from the 300 Monet images and keep the final evaluation size at 256×256. |
| Epoch budget | 20 + 20 (the paper's 100 + 100 was projected at ~35 h on this GPU) | Before training, the projected cost of 100+100 epochs made a 20+20 run the practical choice for the available compute. The completed run took 6 h 41 m. The later snapshot sweep is useful for planning another run, but it was not evidence available when this budget was chosen. |

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

Checkpoint quality was measured on the same fixed 300 images per direction. The proxy uses the
submission formula, `(mean FID + mean MiFID) / 2`. Full per-direction values are in
[outputs/snapshot_metrics.csv](outputs/snapshot_metrics.csv), generated by
[src/snapshot_sweep.py](src/snapshot_sweep.py).

| Epoch | Mean FID | Mean MiFID | Proxy score |
|---:|---:|---:|---:|
| 5 | 111.891 | 0.4210 | 56.156 |
| 10 | 109.108 | 0.4196 | 54.764 |
| 15 | 107.978 | 0.4184 | 54.198 |
| 20 | 101.952 | 0.4143 | 51.183 |
| 25 | 101.510 | 0.4105 | 50.960 |
| 30 | 101.146 | 0.4130 | 50.779 |
| 35 | 101.799 | 0.4133 | 51.106 |
| 40 | **101.026** | **0.4114** | **50.719** |

- NaN / non-finite steps: **0**.
- Largest gradient-norm spikes: **527.3** at step 142,550 (epoch 21),
  165.3 at step 218,050, 131.8 at step 36,800, 130.5 at step 148,000. No loss divergence followed any of them.
- Curves: [outputs/plots/training_curves_t3_baseline_20260929-235720.png](outputs/plots/training_curves_t3_baseline_20260929-235720.png)

The training losses fall fastest in the first ten epochs. Mean generator loss drops from 5.78 in
epochs 1–5 to 4.44 in epochs 6–10, and cycle L1 drops from 0.33 to 0.24. Checkpoint quality follows
a different pattern: the proxy improves by 1.39 points from epoch 5 to 10 and by 3.58 points from
epoch 10 to 20. From epochs 11–40, the per-epoch mean discriminator loss ranges from 0.245 to
0.310. For reference, a discriminator that outputs 0.5 everywhere gives a combined loss of 0.5
in this implementation, while perfect separation gives 0. Values near 0.25 therefore indicate
useful real/fake separation; they do not show that both sides were evenly matched.

After epoch 20, the proxy varies by 0.46 points and epoch 30 is 0.061 above the final score. This
describes a plateau under the schedule that was run. The learning rate decays throughout the
same interval, and the experiment has no constant-rate control beyond epoch 20. It therefore
does not test whether more constant-rate training would help.

The 527.3 gradient norm at step 142,550 is isolated at the logging resolution. The log contains
one record every 50 steps, so it cannot establish that only one batch was affected. At step
142,450, generator loss was 6.56 and cycle loss was 0.338; at 142,550 they were 4.07 and 0.19,
while discriminator loss was 0.17. The next logged norm, at 142,600, was 20.1, and no non-finite
value or sustained loss jump followed. Gradients were measured but not clipped, so the evidence
supports recovery by the next logged point without identifying the cause of the spike.

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

At the final checkpoint, B2A has numerically lower FID and KID than A2B. That does not establish
that one direction is better. A2B is compared with real photos and B2A with real Monet images, so
their FID and precision/recall values do not share one reference distribution. The apparent FID
ordering also changes during training: A2B is lower at epochs 10, 20 and 25, and over epochs
20–40 A2B spans 4.7 FID points and B2A 6.5. That variation is similar to the final
4.34-point gap.

The useful interpretation is within each target domain. For A2B, precision is higher than recall
(0.703 versus 0.480), which is consistent with many outputs looking locally photographic while
covering a narrower part of the photo distribution. The fixed Monet compositions and the cycle
constraint limit how broadly those 300 inputs can cover varied photographs. For B2A, recall is
higher than precision (0.680 versus 0.507). Its outputs cover more of the measured Monet feature
space, but fewer fall in dense real-image regions. The examples show the same unevenness: some
photos gain convincing brush texture, while sparse skies can become broad colour fields.

B2A also has slightly higher LPIPS and lower content cosine than A2B (0.378 versus 0.354, and
0.772 versus 0.797). With no error bars, these gaps are only consistent with B2A changing its
inputs more strongly; they do not confirm a reliable directional difference.

## Kaggle submission

`submission.csv` is produced by `src/infer.py` from checkpoint
`t3_baseline_20260929-235720_final.pt` — the direct output of my own trained CycleGAN, with no
manual editing, hand-picking, or external images, and no pretrained or foundation image model
used to generate or touch it. Pretrained networks appear only in *measurement* (InceptionV3 for
FID/KID/MiFID, AlexNet for LPIPS) — still to be confirmed with the instructor as acceptable.

| | value |
|---|---|
| Competition | DATA 266 Fall 2026 — GAN Image Style Transfer |
| Team | PairProgramming_Team_32 |
| Scoring | `-(FID + MiFID) / 2`, one leaderboard over **all** test data (no public/private split) |
| My submission | FID 101.0258, MiFID 0.4114 → **-50.7186** |
| Team's most recent entry (shreya_akotiya) | FID 100.3438, MiFID 0.4124 → -50.3781 |
| **Team rank** | **1** (6 entries) |

**What the rank reflects.** Kaggle ranks a team by its **best** entry, not its latest. The team's
best is an earlier entry scoring **-39.7787** (FID 79.154), which is neither my model nor the
`submission.csv` currently committed by my teammate. The rank-1 standing is therefore a team
result that is not attributable to the model documented on this page, and I report my own
submission's score separately above rather than claiming the leaderboard score as my model's.

## Human audit (blinded, 30 samples, 2 raters)

Protocol: 30 fixed samples, 15 per direction, drawn with seed 42 from the first 300 sorted
predictions — the same set the instructor's evaluator scores. Each sheet shows **source |
translation** side by side, so content preservation can actually be judged; sheets are shuffled
and file names stripped. Raters: zoheb_waghu (rater1) and shreya_akotiya (rater2), scoring
style / content / artifacts 1–5 where **5 is best on every axis**, independently, without
discussing scores. Materials: [outputs/human_audit/](outputs/human_audit/),
[outputs/RATING_GUIDE.md](outputs/RATING_GUIDE.md).

| Direction | style | content | artifacts |
|---|---|---|---|
| A2B (Monet → photo) | 3.17 | 4.57 | 4.63 |
| B2A (photo → Monet) | 3.37 | 4.47 | 4.83 |

**Inter-rater agreement: Cohen's κ = 0.1487**, exact agreement 63.3%, and **100% of ratings
within one point** on all three axes.

### Two rounds, and why the first was discarded

| | round 1 (no rubric) | round 2 (rubric) |
|---|---|---|
| Exact agreement | 23.3% | **63.3%** |
| Within 1 point | 72% | **100%** |
| κ overall | **−0.050** | **+0.149** |
| κ, artifacts axis | −0.130 | +0.153 |

Round 1 was run without an agreed rubric and produced κ = −0.05 — agreement *worse than chance*.
Per-axis diagnosis showed the cause was definitional, not perceptual: `artifacts` was negative on
every measure, the signature of the two raters scoring the axis in opposite directions
(disagreeing on whether 5 meant "clean" or "many artifacts"), while `content` already agreed
(quadratic κ 0.41). A rubric with explicit 1–5 anchors was written, and the **same 30 sheets**
were re-rated. Round 1 is retained as evidence in
[outputs/human_audit_ratings_round1.csv](outputs/human_audit_ratings_round1.csv).

### Why κ is low even though the raters barely disagree

κ = 0.149 reads as "slight agreement" on the Landis–Koch scale, which understates what happened.
Both raters used only **two of the five categories** on every axis, so chance agreement is
already high and κ gives little credit for beating it:

| Axis | categories used | chance agreement | observed | κ |
|---|---|---|---|---|
| style | {3, 4} | 60.7% | 66.7% | 0.153 |
| content | {4, 5} | 49.6% | 56.7% | 0.141 |
| artifacts | {4, 5} | 60.7% | 66.7% | 0.153 |

This is the well-known kappa paradox: high observed agreement with skewed marginals yields a low
κ. It is also why unweighted and quadratic-weighted κ are **identical here** — quadratic
weighting only differs from unweighted when some disagreements span more than one category, and
no disagreement in this audit does. The honest summary is: the raters never differed by more than
one point, and κ is deflated by low category variance rather than by real disagreement.

### What the audit says about the model

Content (4.47–4.57) and artifacts (4.63–4.83) are strong: the model preserves the scene and
produces clean images. **Style (3.17–3.37) is the weak axis in both directions** — the human
verdict matches the computed metrics, where FID is 103.2 / 98.9 and LPIPS shows relatively modest
change from the input. This model is conservative: it protects content and avoids artifacts at
the cost of committing to the target style. See [failure_analysis.md](failure_analysis.md).

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
python task3_gan/zoheb_waghu/src/snapshot_sweep.py --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml --train-run-id <run_id>
```

## What I would try next

I would reserve a validation subset before training and use its FID, KID and precision/recall for
checkpoint selection. The small change after epoch 20 makes a 20+10 schedule worth testing, but
epoch 30 in this run is not the result of that schedule: its learning rate was still about
1.05e-4 midway through a 20-epoch decay. A proper 20+10 experiment would decay to zero by epoch
30 and compare against both the current schedule and a constant-rate control. Each condition
needs at least three seeds before a difference as small as 0.061 is treated as meaningful.

For image quality, I would test a second discriminator with a larger receptive field, trained
from scratch, and measure whether B2A precision improves without reducing its 0.680 recall. A
separate identity-loss sweep could test the colour shifts in the orange-cloud and storm-sky
cases. I would also compare the current sampling with stronger augmentation of the official
Monet images while holding total updates fixed. A pretrained perceptual loss or an expanded
Monet set would require confirmation that pretrained training components or external data are
allowed, so neither is part of these proposed experiments.
