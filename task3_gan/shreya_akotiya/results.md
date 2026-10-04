# Task 3 - CycleGAN Style Transfer Results

**Student:** Shreya Akotiya  
**Run:** `t3_shreya_unet_128_20261002_023102`  
**Actual image size used:** 256 x 256  
**Notebook:** `src/task3_cyclegan.ipynb`

## 1. What I built

I trained a CycleGAN to translate between photographs and Monet paintings. The training was unpaired, so the model did not see a matching photo and painting during training. It learned two translations:

- Photo to Monet
- Monet to photo

The model has two generators and two discriminators. The generators create the translated images, while the discriminators try to tell real images from generated images. The cycle-consistency loss then checks whether translating an image to the other domain and back can approximately recover the original image.

The dataset contained 7,038 photographs and 300 Monet paintings. The images were resized and trained at 256 x 256 pixels.

## 2. Model architecture

### Generator

I used a UNet generator with skip connections. Each generator has about 41.8 million parameters. The encoder reduces the image to a small bottleneck, and the decoder reconstructs the image in the other style. The skip connections pass edge and layout information directly from the encoder to the decoder.

I chose a UNet because the translation should mainly change colour and texture while keeping the original scene recognizable. The skip connections make it easier for the model to preserve objects, edges, and the general layout. This is different from the ResNet-9 generator used by Zoheb.

### Discriminator

I used a PatchGAN discriminator with about 662,000 parameters. Instead of giving one real/fake decision for the whole image, it produces a grid of decisions for small image patches. This is useful for checking local details such as brush texture and colour patterns.

The full model contained **84,967,560 parameters**.

## 3. Training settings

| Setting | Value | Reason |
|---|---|---|
| Image size | 256 x 256 | Keeps the image details while fitting the GPU memory. |
| Epochs | 80 | Fit within the available Colab session. |
| Batch size | 1 | Common for CycleGAN with instance normalization. |
| Optimizer | Adam, beta values `(0.5, 0.999)` | A common stable starting point for GAN training. |
| GAN loss | Least-squares GAN loss | Gives smoother gradients than the original binary cross-entropy loss. |
| Cycle-loss weight | 10.0 in both directions | Encourages the model to preserve the original content. |
| Identity-loss weight | 5.0 | Discourages unnecessary colour changes. |
| Replay buffer | 50 generated images | Lets the discriminators see older generated examples. |
| Augmentation | Resize to 286, random crop to 256, horizontal flip | Adds variation, which is useful because there are only 300 Monet images. |
| Learning rate | 0.0002 until epoch 61, then linearly decreased over epochs 62-80 (to 0.000019) | Follows the usual CycleGAN schedule, shortened to fit 80 epochs. |

## 4. Training behaviour

The training was stable in the basic numerical checks. There were no NaN or Inf values, and neither discriminator collapsed to zero.

The cycle loss decreased quickly during the first 10 epochs and then stayed close to 0.41. This means that the model learned to reconstruct the original images fairly early. However, the discriminators slowly became stronger later in training. The average discriminator loss decreased from about 0.23 to 0.08, while the Photo-to-Monet adversarial loss increased from about 0.40 to 0.76.

The generator loss was lowest around epoch 5. Since the FID was measured only after epoch 80, I cannot determine from this run whether the later learning-rate decay improved the final image quality.

| Epoch | Generator loss | Discriminator loss | Cycle A | Cycle B | Photo-to-Monet adversarial loss | Learning rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.755 | 0.227 | 0.731 | 0.649 | 0.395 | 0.0002 |
| 10 | 2.191 | 0.163 | 0.451 | 0.449 | 0.540 | 0.0002 |
| 30 | 2.290 | 0.119 | 0.410 | 0.409 | 0.658 | 0.0002 |
| 60 | 2.379 | 0.098 | 0.410 | 0.386 | 0.688 | 0.0002 |
| 80 | 2.461 | 0.082 | 0.405 | 0.395 | 0.759 | 0.000019 |

## 5. Image-quality metrics

The clean-fid evaluation used all generated images. The instructor-style evaluation used the first 300 sorted images in each direction. These two evaluations use different image counts and resize procedures, so their FID values should not be compared directly.

### Clean-fid results using all generated images

| Metric | Photo to Monet | Monet to photo |
|---|---:|---:|
| FID | 80.55 | 84.05 |
| KID | 0.0110 | 0.0213 |

### Instructor-style evaluation using 300 images per direction

| Metric | Photo to Monet | Monet to photo | Mean submission value |
|---|---:|---:|---:|
| FID | 97.904 | 102.784 | **100.344** |
| MiFID | 0.4043 | 0.4206 | **0.4124** |

The competition score is (FID + MiFID) / 2 = (100.344 + 0.4124) / 2 = **50.38**, where lower is
better. This is the value in `submission.csv`. The run copy of the instructor's notebook, with
outputs, is `src/part3_evaluation_shreya.ipynb`.

### Team metrics

The team uses A = Monet and B = photo. Therefore, B2A means Photo to Monet and A2B means Monet to photo.

Computed by `src/run1_team_metrics.ipynb` from the epoch-80 checkpoint, with the same 300-image
method as the submission, and stored in `metrics_report.csv`. The last column is Zoheb's ResNet-9,
measured the same way.

| Metric | Photo to Monet | Monet to photo | Zoheb (Photo to Monet / Monet to photo) |
|---|---:|---:|---:|
| FID | 97.904 | 102.784 | 98.855 / 103.197 |
| KID | 0.0069 | 0.0185 | 0.0076 / 0.0182 |
| Generative precision | 0.540 | 0.730 | 0.507 / 0.703 |
| Generative recall | 0.730 | 0.447 | 0.680 / 0.480 |
| Density | 0.489 | 1.005 | 0.433 / 1.013 |
| Coverage | 0.763 | 0.850 | 0.690 / 0.897 |
| Cycle L1 | 0.0216 | 0.0223 | 0.0398 / 0.0332 |
| LPIPS change | 0.304 | 0.244 | 0.378 / 0.354 |
| Content cosine similarity | 0.821 | 0.861 | 0.772 / 0.797 |

My model and Zoheb's were close on FID and KID: about one FID point apart in each direction,
which is small for an evaluation based on only 300 images. The UNet preserved content better than
Zoheb's ResNet model: it had higher content-cosine values, smaller LPIPS change and about half the
cycle error. Its skip connections likely helped with this.

Photo-to-Monet had higher recall but lower precision. In other words, it produced a wider range of Monet-like results, but some of them did not look very convincing. Monet-to-photo had higher precision but lower recall. Those outputs looked more like photos, but they covered a smaller range of the real photo distribution.

## 6. What the images showed

The model learned to change colours and add painterly texture while often keeping the main objects and scene layout. However, it had problems with dark scenes, flat regions, and images containing a lot of fine detail.

Some night photos changed into a repeated blue-grey tiled pattern. Some Monet paintings became very dark when translated into photos. I also saw horizontal streaks in several outputs. These artifacts were especially visible in areas such as skies and other regions with little texture.

The low cycle error should also be interpreted carefully. A low reconstruction error does not always mean that the translated image looks good. In some night scenes, the translated image looked very different, but the cycle reconstruction still recovered the original details. This suggests that the model may be hiding information in small signals that are difficult for a person to see. This behaviour is sometimes called steganography in CycleGANs.

## 7. Hardware and evidence

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB (Google Colab) |
| Framework | PyTorch 2.11.0+cu130, CUDA |
| Parameters | 84,967,560 (two generators of 41,821,187, two discriminators of 662,593) |
| Training time | 9.66 hours (34,793 s), about 430 s per epoch |
| Throughput | about 16.2 images/sec (563,040 steps, batch size 1) |
| Numerical stability | 0 NaN/Inf values in 11,260 logged steps |
| Generator gradient norm per step | mean 38.40, median 36.11, 95th percentile 58.61, max 232.91 |
| Peak GPU memory | 1.78 GB (batch size 1, fp32) |

| Evidence | Path |
|---|---|
| Checkpoint (generators, fp16) | `checkpoints/t3_shreya_unet_128_20261002_023102_G_AB_fp16.pt`, `checkpoints/t3_shreya_unet_128_20261002_023102_G_BA_fp16.pt` (cut from the 1.0 GB epoch-80 checkpoint, which is not in the repo) |
| Raw log | `reproducibility/raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_128_20261002_023102.log` |
| Gradient norms / peak memory | `reproducibility/raw_logs/shreya_akotiya/task3_gan/t3_shreya_unet_128_20261002_023102_graddiag.{csv,json}` |
| Manifest | `reproducibility/manifests/shreya_akotiya/t3_shreya_unet_128_20261002_023102.json`, `task3_gan_manifest.md` |
| Metrics | `metrics_report.csv`, `submission.csv` |
| Training losses / plots | `outputs/train_history.csv`, `outputs/plots/loss_curves.png` |
| Generated images | `outputs/pred_A2B/` (Photo to Monet, 300), `outputs/pred_B2A/` (Monet to photo, 300) |

The run ID contains "128" because the run tag was set before the run was switched to 256px; the
log confirms 256 x 256 training.

## 8. Human audit and Kaggle results still missing

The assignment requires a blinded audit of 30 images using two raters. The `human_audit.py` script is ready to create the 30-image sheet and rating file, but the raters' scores have not been entered yet. Therefore, the following values cannot be filled in honestly:

| Audit result | Status |
|---|---|
| Style score | Pending two-rater audit |
| Content score | Pending two-rater audit |
| Artifact score | Pending two-rater audit |
| Cohen's kappa | Pending two-rater audit |
| Percentage agreement | Pending two-rater audit |

The Kaggle public/private score and leaderboard rank are also not available in the current files. They should be added after the generated submission is uploaded to the class competition.

The current metrics file correctly records the following items as `NOT_MEASURED`:

- Human-audit scores.
- Kaggle public score.
- Kaggle private score.
- Leaderboard rank.

## 9. Other runs

I also tested three alternatives, but none performed better than the submitted run.

| Run | Change | Score, lower is better |
|---|---|---:|
| Run 1, submitted | Original UNet setup | **50.38** |
| v2 | DiffAugment, lower discriminator learning rate, 40 + 40 schedule | 54.67 |
| v3 | Lower identity loss, removed outer skip, EMA weights | 55.90 |
| v4 | Same skip removal with identity weight 0.5 | 54.16 |

The original run performed best. Removing the outer skip connection appears to have hurt the model because both v3 and v4 performed worse. The v4 run also showed that FID moved several points between checkpoints, so some of the difference between runs may be normal GAN variation.

## 10. Limitations and next steps

The main limitations are the single random seed, the small number of Monet paintings, and the noisy 300-image FID evaluation. The domain sizes are also very different: there are 300 Monet paintings and 7,038 photos. This makes it easier for the Monet discriminator to memorize the training paintings.

My next steps would be to keep the original UNet architecture, test the identity weight separately, and repeat the best setting with three seeds. I would also complete the human audit and submit the generated images to Kaggle so that the missing required results can be added.
