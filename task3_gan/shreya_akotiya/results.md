# Task 3 — CycleGAN style transfer — results (shreya_akotiya)

## What I built

A CycleGAN model to transform photographs into Monet-style paintings using unpaired image-to-image translation.

## Architecture and why

**Generator: UNet with skip connections**
- 8 encoder blocks, 7 decoder blocks with skip connections
- Skip connections preserve spatial details during style transfer
- 41.8M parameters per generator
- Chose UNet over ResNet-9 (Zoheb's approach) because skip connections help maintain image structure while changing style

**Discriminator: PatchGAN**
- 3 convolutional layers
- 31x31 patch output for local texture discrimination
- 662K parameters per discriminator

**Total parameters:** 85M

## Hyperparameters and why

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Image size | 256px | Higher resolution for better FID; standard for CycleGAN |
| Epochs | 80 | Sufficient for convergence without overfitting |
| LR decay | Starts epoch 60, 20 epochs | Stabilizes training in later epochs |
| Learning rate | 2e-4 | Standard for Adam with GANs |
| Batch size | 1 | Standard for CycleGAN (instance normalization) |
| Lambda cycle | 10.0 | Strong cycle consistency constraint |
| Lambda identity | 0.5 | Preserves color distribution |
| Replay buffer | 50 | Stabilizes discriminator training |

## Metrics

| Metric | Value |
|--------|-------|
| FID (Photo→Monet) | 79.15 |
| MiFID | 0.40 |
| **Kaggle Score** | **39.78** |
| KID (Photo→Monet) | 0.0110 |
| LPIPS | 0.316 |
| Cycle L1 (A→B→A) | 0.042 |
| Cycle L1 (B→A→B) | 0.044 |

**Comparison with teammate (Zoheb - ResNet-9):**
| Metric | Shreya (UNet) | Zoheb (ResNet-9) |
|--------|---------------|------------------|
| FID | 79.15 | 101.03 |
| MiFID | 0.40 | 0.41 |
| Score | 39.78 | 50.72 |

Evidence: loss curves in `outputs/plots/`, training history in `outputs/train_history.csv`

| Result | Checkpoint | Raw log |
|--------|------------|---------|
| FID 79.15 | t3_shreya_unet_256_epoch080.pt | t3_shreya_unet_128_20261002_023102.log |

## Hardware disclosure

| Item | Value |
|------|-------|
| GPU | NVIDIA T4 (Colab) |
| RAM | ~12 GB |
| Training time | 9.66 hours |
| Where it ran | Google Colab |

## What I would try next

1. **More epochs** - Could train to 150 epochs with extended LR decay
2. **Attention mechanisms** - Add self-attention layers to capture global style patterns
3. **Progressive training** - Start at 128px, then fine-tune at 256px
4. **Perceptual loss** - Add VGG-based perceptual loss for better style transfer
