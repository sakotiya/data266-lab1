"""CycleGAN networks. Every size comes from the `model:` block of configs/cyclegan_baseline.yaml.

Generator: ResNet generator (Johnson et al.; the CycleGAN paper's 256px generator).
Discriminator: 70x70 PatchGAN (Isola et al.), scores overlapping patches, not whole images.
Instance norm throughout, weights ~ N(0, init_gain).
"""
from __future__ import annotations

import torch.nn as nn


def init_weights(net: nn.Module, gain: float) -> nn.Module:
    def init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(init)
    return net


class ResBlock(nn.Module):
    """x + (pad, conv3, IN, ReLU, pad, conv3, IN)(x). Reflection padding avoids border artifacts."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim))

    def forward(self, x):
        return x + self.body(x)


class ResnetGenerator(nn.Module):
    """7x7 conv -> 2 stride-2 downsamples -> N residual blocks -> 2 transposed-conv upsamples
    -> 7x7 conv -> tanh. Output in [-1, 1], same size as the input."""

    def __init__(self, filters: int, blocks: int) -> None:
        super().__init__()
        f = filters
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, f, 7), nn.InstanceNorm2d(f), nn.ReLU(True)]
        for i in range(2):                                      # f -> 2f -> 4f
            c = f * 2 ** i
            layers += [nn.Conv2d(c, c * 2, 3, 2, 1), nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
        layers += [ResBlock(f * 4) for _ in range(blocks)]
        for i in range(2):                                      # 4f -> 2f -> f
            c = f * 2 ** (2 - i)
            layers += [nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(f, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """`layers` stride-2 4x4 convs (channels doubling, capped at 8x), then a stride-1 conv, then a
    1-channel map of real/fake scores. layers=3 gives a 70x70 receptive field."""

    def __init__(self, filters: int, layers: int) -> None:
        super().__init__()
        f = filters
        seq = [nn.Conv2d(3, f, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(f * prev, f * mult, 4, 2, 1), nn.InstanceNorm2d(f * mult),
                    nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** layers, 8)
        seq += [nn.Conv2d(f * prev, f * mult, 4, 1, 1), nn.InstanceNorm2d(f * mult),
                nn.LeakyReLU(0.2, True), nn.Conv2d(f * mult, 1, 4, 1, 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def build_networks(cfg: dict) -> dict:
    """G_AB: Monet -> photo, G_BA: photo -> Monet, D_A judges Monet, D_B judges photos."""
    m = cfg["model"]
    if m["generator"] != "resnet" or m["discriminator"] != "patchgan" or m["norm"] != "instance":
        raise NotImplementedError("only resnet / patchgan / instance are implemented")
    make_g = lambda: ResnetGenerator(m["gen_filters"], m["gen_blocks"])
    make_d = lambda: PatchDiscriminator(m["disc_filters"], m["disc_layers"])
    return {name: init_weights(net, m["init_gain"])
            for name, net in (("g_ab", make_g()), ("g_ba", make_g()), ("d_a", make_d()), ("d_b", make_d()))}
