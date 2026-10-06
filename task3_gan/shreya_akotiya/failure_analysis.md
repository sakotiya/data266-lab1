# Task 3 - CycleGAN Failure Analysis

**Student:** Shreya Akotiya  
**Run:** `t3_shreya_unet_128_20261002_023102`, epoch 80  
**Model:** UNet generators with PatchGAN discriminators

I selected the failure cases using the measured LPIPS and cycle-L1 values, then inspected the images manually. The evaluation used the first 300 sorted images in each direction, following the instructor's evaluation method.
All 30 candidates are listed in `outputs/failure_candidates.csv`, and the images (source, translation,
cycle reconstruction) are in `outputs/plots/failure_candidates_B2A.jpg` and `outputs/plots/failure_candidates_A2B.jpg`.

## 1. Main patterns in the results

The two translation directions had different weaknesses.

- **Photo to Monet:** The outputs were varied, but not all of them looked like real Monet paintings. This matches the higher recall of 0.730 and lower precision of 0.540.
- **Monet to photo:** The outputs looked more photographic, but they were less varied. This matches the higher precision of 0.730 and lower recall of 0.447.
- **Content preservation:** Monet-to-photo changed the images less, with LPIPS 0.244 and content cosine 0.861. Photo-to-Monet changed the images more, with LPIPS 0.304 and content cosine 0.821.

The model had particular trouble with dark skies, smooth backgrounds, and fine details such as leaves, text, and roof patterns.

## 2. Cycle-consistency finding

The mean cycle errors were low in both directions:

| Direction | Mean cycle L1 | Worst cycle L1 |
|---|---:|---:|
| Photo to Monet to photo | 0.0216 | 0.0455 (`08b790bca7.jpg`) |
| Monet to photo to Monet | 0.0223 | 0.0448 (`a619072f82.jpg`) |

These values show that the model can reconstruct the source image fairly closely. However, a low cycle error does not guarantee that the translation itself looks good. For example, several dark night photos became tiled or blotchy, but the reconstruction still restored the original night scene. This suggests that the generator may be hiding information in subtle image patterns that are recovered during the reverse translation.

## 3. Training stability

The run did not produce NaN or Inf values. The generator loss was lowest around epoch 5. After that, the discriminator became stronger: the average discriminator loss decreased from about 0.23 to 0.08, while the Photo-to-Monet adversarial loss increased from about 0.40 to 0.76. This may explain why later outputs still contained artifacts even though the cycle loss remained low.

## 4. Individual failure cases

| # | Sample | Direction | What I observed | Likely reason |
|---:|---|---|---|---|
| 1 | `09fc404e31.jpg` | Photo to Monet | A dark, starry sky became a blotchy blue-grey field with a repeated tiled pattern at the top. The cycle image restored the original sky. | Low-contrast night images are difficult for InstanceNorm and the decoder. The model may also be hiding the source information in the texture. |
| 2 | `07054731ab.jpg` | Photo to Monet | City lights at dusk produced the same tiled band and a mottled sky, although the lights were kept. | The generator learned a repeated texture for dark regions instead of representing the actual scene. |
| 3 | `08341635fa.jpg` | Photo to Monet | A dark corridor became grey-green, while the bright doorways stayed visible. | The bright areas provided stronger information than the dark background. |
| 4 | `02ded12bbd.jpg` | Photo to Monet | A dull sunset became a flat grey haze and lost the red horizon in the translated image. | PatchGAN rewards local Monet-like texture, but it does not strongly protect global colour arrangement. |
| 5 | `063ab57d41.jpg` | Photo to Monet | An office building and road sign stayed mostly photographic with only a light colour change. | The scene was unlike typical Monet paintings, so a small change was an easier solution. Identity loss also encourages small changes. |
| 6 | `04b8bfdb1c.jpg` | Photo to Monet | Cows on grass remained clearly photographic, although the image became slightly softer and lighter. | The generator made only a small style change because the original content was easy to preserve. |
| 7 | `0962094f25.jpg` | Photo to Monet | A banana plant under a glass roof barely changed, but its cycle error was relatively high. The watermark text was copied through. | Small details such as leaves, roof lines, and text are difficult to reconstruct exactly. |
| 8 | `08b790bca7.jpg` | Photo to Monet | A rope bridge received convincing brush texture, but fine leaf detail was lost during reconstruction. | Foliage contains high-frequency detail that conflicts with the painted texture. |
| 9 | `b1ea5d5a7d.jpg` | Monet to photo | A golden sunset painting became almost black, although the reconstruction recovered the gold colour. | The generator mapped soft painted light to a darker, higher-contrast photo style. |
| 10 | `6a03aea8be.jpg` | Monet to photo | A blue-green painting became a dark field with bright horizontal streaks at the top. | The streaks appear to be an artifact from the transposed-convolution decoder in a flat region. |
| 11 | `2cca56415e.jpg` | Monet to photo | A haystack sunset became a strong orange-red flare and the field became very dark. | The photo domain contains high-contrast sunsets, so the generator exaggerated the contrast. |
| 12 | `a619072f82.jpg` | Monet to photo | A coastal painting developed a rainbow-like horizontal band across the sky, and the band remained in the reconstruction. | The repeated artifact is produced by the generator rather than by the source image. |
| 13 | `676a5a4c2e.jpg` | Monet to photo | A dense hillside painting was returned almost unchanged and still looked painted. | The brushwork already contains local texture that the discriminator can accept as photo-like. |
| 14 | `10c555c1b1.jpg` | Monet to photo | A bridge with boats kept much of its painted appearance and also showed horizontal sky streaks. | This combines the weak style change from case 13 with the sky artifact from cases 10 and 12. |

## 5. Shared problems with the other model

Several difficult images also appeared in Zoheb's ResNet-9 failure list, including the dark Parliament scene, the coastal image, the building, and the sunset. This suggests that these images are difficult for CycleGAN generally, not only for my UNet. The tiled pattern in the night photos and the horizontal streaks in the sky were more specific to my model.

## 6. Human audit

The blinded audit is complete: 30 samples (15 per direction), two raters, scores 1-5 for style,
content and artifacts (5 = best), using the rubric in `outputs/RATING_GUIDE.md`.

| Mean score | Monet to photo | Photo to Monet |
|---|---:|---:|
| Style | 3.80 | 4.00 |
| Content | 4.23 | 3.83 |
| Artifacts (5 = none) | 4.23 | 4.20 |

Inter-rater agreement: Cohen's kappa **0.500** (moderate), 67.8% exact agreement, 84.4% within one
point. Details: `outputs/human_audit_summary.json`.

The audit agrees with the failure cases above. Photo to Monet scores higher on style but lower on
content, which fits the night photos that turn into tiled patterns. Artifacts score around 4.2 in
both directions, because the tiling and the sky streaks appear in only some images.

## 7. Main conclusion

The model learned to change colour and texture and usually kept the main scene structure. Its biggest problems were repeated patterns in dark images, horizontal streaks, exaggerated contrast, and translations that changed too little. The low cycle error shows that reconstruction works, but it does not prove that the generated style is visually correct. A larger set of Monet paintings, more seeds, and a completed human audit would give a stronger evaluation.
