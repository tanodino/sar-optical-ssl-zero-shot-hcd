"""
GPU augmentation pipelines for SAR and optical images.

The pipelines follow the official DeCUR code for SSL4EO-S12 (`src/pretrain/pretrain_mm.py`,
`train_transforms_s1` / `train_transforms_s2c`):

    SAR     : RandomResizedCrop(scale 0.2-1) -> grayscale p=0.2 -> Gaussian blur p=0.5 -> h/v flips
    optical : RandomResizedCrop(scale 0.2-1) -> brightness+contrast p=0.8 -> grayscale p=0.2
              -> Gaussian blur p=0.5 -> solarize p=0.2 -> h/v flips

Properties:
* No clipping to [0, 1]: every operation works for any value range (data are normalised by preprocess_corpus.py).
* Every random decision (apply or not, parameters) is drawn **per sample**, not once per batch.

Value-range-dependent operations
--------------------------------
* Brightness is multiplicative in physical units (DeCUR: factor U(0.6, 1.4) on reflectance). In normalised
  units the physical zero is at `zero_point = -mean / std` per channel, so brightness is applied as
  x' = zero_point + f * (x - zero_point). With the default zero_point = 0 it is a plain multiplication.
  The true values come from the normalisation statistics (`norm_stats.json`).
* Solarize (DeCUR: threshold 0.5 on [0, 1] data, x' = 1 - x above it) is applied **per channel** in normalised
  units: threshold_c = normalised middle of the clipping range of band c, and
  x' = max_c - x with max_c = lo_c + hi_c (normalised clip limits), i.e. a reflection about the middle, so the
  brightest clipped value maps to the darkest, exactly as 1 - x on [0, 1]. Values come from `norm_stats.json`
  (key `optical_augmentation`). Without them (unit/smoke tests) solarize is disabled.

Usage: `aug = Augmenter.sar(img_size)` or `Augmenter.optical(img_size)`, then `x_aug = aug(x)` on a
(B, C, H, W) tensor already on the GPU. The module has no parameters.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bernoulli(batch: int, p: float, device) -> torch.Tensor:
    """(B,) boolean tensor, True with probability p independently for each sample."""
    return torch.rand(batch, device=device) < p


def random_resized_crop(x: torch.Tensor, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)) -> torch.Tensor:
    """Per-sample RandomResizedCrop, output has the same size as the input.

    For each image an area fraction s ~ U(scale) and a log-uniform aspect ratio r ~ ratio are drawn; the crop
    (clipped to the image) is resampled to the full size with bilinear interpolation (affine grid sampling).
    """
    B, _, H, W = x.shape
    dev = x.device
    area = torch.empty(B, device=dev).uniform_(*scale)
    log_r = torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    r = torch.exp(log_r)
    # Crop width / height as fractions of the image side (w*h = area, w/h = r), clipped to <= 1.
    w = torch.sqrt(area * r).clamp(max=1.0)
    h = torch.sqrt(area / r).clamp(max=1.0)
    # Crop centre, in normalised coordinates [-1, 1], such that the crop stays inside the image.
    cx = (torch.rand(B, device=dev) * 2 - 1) * (1 - w)
    cy = (torch.rand(B, device=dev) * 2 - 1) * (1 - h)
    theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 2] = w, cx
    theta[:, 1, 1], theta[:, 1, 2] = h, cy
    grid = F.affine_grid(theta, size=list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def random_flips(x: torch.Tensor) -> torch.Tensor:
    """Independent horizontal and vertical flips, each with probability 0.5 per sample."""
    B = x.shape[0]
    hflip = _bernoulli(B, 0.5, x.device).view(B, 1, 1, 1)
    x = torch.where(hflip, x.flip(-1), x)
    vflip = _bernoulli(B, 0.5, x.device).view(B, 1, 1, 1)
    return torch.where(vflip, x.flip(-2), x)


def random_grayscale(x: torch.Tensor, p: float) -> torch.Tensor:
    """With probability p per sample, replace every channel by the mean over channels (any number of bands)."""
    B = x.shape[0]
    apply = _bernoulli(B, p, x.device).view(B, 1, 1, 1)
    return torch.where(apply, x.mean(dim=1, keepdim=True).expand_as(x), x)


def random_gaussian_blur(x: torch.Tensor, p: float, sigma=(0.1, 2.0), kernel_size: int = 9) -> torch.Tensor:
    """With probability p per sample, blur with an isotropic Gaussian of sigma ~ U(sigma).

    One separable kernel per sample (shared by its channels). kernel_size = 9 covers +-2 sigma for the largest
    sigma (2.0); borders use reflection padding.
    """
    B, C, H, W = x.shape
    sig = torch.empty(B, device=x.device).uniform_(*sigma)
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
    k1d = torch.exp(-(t[None, :] ** 2) / (2 * sig[:, None].to(x.dtype) ** 2))     # (B, k)
    k1d = k1d / k1d.sum(dim=1, keepdim=True)
    # Grouped convolution: treat (B*C) as channels so each sample uses its own kernel.
    kh = k1d.repeat_interleave(C, dim=0).view(B * C, 1, kernel_size, 1)
    kw = k1d.repeat_interleave(C, dim=0).view(B * C, 1, 1, kernel_size)
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (0, 0, half, half), mode="reflect"), kh, groups=B * C)
    y = F.conv2d(F.pad(y, (half, half, 0, 0), mode="reflect"), kw, groups=B * C)
    y = y.view(B, C, H, W)
    apply = _bernoulli(B, p, x.device).view(B, 1, 1, 1)
    return torch.where(apply, y, x)


def random_brightness_contrast(x: torch.Tensor, p: float, strength: float = 0.4,
                               zero_point: Optional[torch.Tensor] = None) -> torch.Tensor:
    """With probability p per sample, apply brightness then contrast (DeCUR RandomBrightness/RandomContrast(0.4)).

    brightness: x' = z0 + f_b * (x - z0), f_b ~ U(1 - strength, 1 + strength), z0 = physical zero (see module doc)
    contrast:   x'' = m + f_c * (x' - m), f_c ~ U(1 - strength, 1 + strength), m = per-sample, per-channel mean
    """
    B, C = x.shape[:2]
    z0 = torch.zeros(1, C, 1, 1, device=x.device, dtype=x.dtype) if zero_point is None \
        else zero_point.to(x.device, x.dtype).view(1, C, 1, 1)
    f_b = torch.empty(B, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(1 - strength, 1 + strength)
    f_c = torch.empty(B, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(1 - strength, 1 + strength)
    y = z0 + f_b * (x - z0)
    m = y.mean(dim=(2, 3), keepdim=True)
    y = m + f_c * (y - m)
    apply = _bernoulli(B, p, x.device).view(B, 1, 1, 1)
    return torch.where(apply, y, x)


def random_solarize(x: torch.Tensor, p: float, threshold, max_value) -> torch.Tensor:
    """With probability p per sample, invert values above the threshold: x' = max_value - x where x >= threshold.

    `threshold` and `max_value` are scalars or per-channel sequences/tensors of length C (one value per band);
    one random decision per sample, applied to all channels with their own threshold.
    """
    B, C = x.shape[:2]
    t = torch.as_tensor(threshold, device=x.device, dtype=x.dtype)
    m = torch.as_tensor(max_value, device=x.device, dtype=x.dtype)
    if t.ndim:                                         # per-channel values -> broadcast as (1, C, 1, 1)
        assert t.numel() == C and m.numel() == C, f"expected {C} per-channel values"
        t, m = t.view(1, C, 1, 1), m.view(1, C, 1, 1)
    y = torch.where(x >= t, m - x, x)
    apply = _bernoulli(B, p, x.device).view(B, 1, 1, 1)
    return torch.where(apply, y, x)


class Augmenter(nn.Module):
    """Augmentation pipeline for one modality; call on a (B, C, H, W) batch, returns a new augmented batch.

    Build it with `Augmenter.sar(...)` or `Augmenter.optical(...)` so that the pipeline matches the paper.
    """

    def __init__(self, brightness_contrast_p: float = 0.0, grayscale_p: float = 0.2, blur_p: float = 0.5,
                 solarize_p: float = 0.0, solarize_threshold: Optional[Sequence[float]] = None,
                 solarize_max: Optional[Sequence[float]] = None, zero_point: Optional[Sequence[float]] = None):
        super().__init__()
        self.brightness_contrast_p = brightness_contrast_p
        self.grayscale_p = grayscale_p
        self.blur_p = blur_p
        self.solarize_p = solarize_p
        self.solarize_threshold = solarize_threshold
        self.solarize_max = solarize_max
        self.register_buffer("zero_point", None if zero_point is None else torch.tensor(zero_point),
                             persistent=False)

    @classmethod
    def sar(cls) -> "Augmenter":
        """SAR pipeline: crop, grayscale (p 0.2), blur (p 0.5), flips. No radiometric jitter (as in DeCUR)."""
        return cls(brightness_contrast_p=0.0, grayscale_p=0.2, blur_p=0.5, solarize_p=0.0)

    @classmethod
    def optical(cls, zero_point: Optional[Sequence[float]] = None,
                solarize_threshold: Optional[Sequence[float]] = None,
                solarize_max: Optional[Sequence[float]] = None) -> "Augmenter":
        """Optical pipeline: crop, brightness/contrast (p 0.8), grayscale (p 0.2), blur (p 0.5), solarize
        (p 0.2, only if thresholds are given; per-band values, see module doc), flips."""
        if (solarize_threshold is None) != (solarize_max is None):
            raise ValueError("solarize_threshold and solarize_max must be given together")
        return cls(brightness_contrast_p=0.8, grayscale_p=0.2, blur_p=0.5,
                   solarize_p=0.2 if solarize_threshold is not None else 0.0,
                   solarize_threshold=solarize_threshold, solarize_max=solarize_max, zero_point=zero_point)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = random_resized_crop(x)
        if self.brightness_contrast_p > 0:
            x = random_brightness_contrast(x, self.brightness_contrast_p, zero_point=self.zero_point)
        x = random_grayscale(x, self.grayscale_p)
        x = random_gaussian_blur(x, self.blur_p)
        if self.solarize_p > 0:
            x = random_solarize(x, self.solarize_p, self.solarize_threshold, self.solarize_max)
        return random_flips(x)
