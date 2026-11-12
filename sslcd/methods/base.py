"""
Common base class of all pre-training methods.

Every method owns two encoders, `enc_sar` and `enc_opt` (saved at the end of pre-training),
plus its own heads (projectors, decoders, ...), and implements

    forward(sar, opt) -> (loss, logs)

where `sar` (B, C_sar, H, W) and `opt` (B, C_opt, H, W) are co-registered batches already on the device,
`loss` is the scalar to minimise and `logs` a dict {name: scalar tensor} of loss components to record.

`global_repr` states the method's native global representation, used by the collapse check and later by
feature extraction mode B: "cls" (CLS token) or "avg" (average of the patch tokens).
The token choice is always explicit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..vit import ViTEncoder


@dataclass
class Arch:
    """Encoder architecture shared by both modalities. Default = ViT-Small with 8x8 patches."""
    img_size: int = 64
    patch_size: int = 8
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    grad_checkpointing: bool = True


class SSLMethod(nn.Module):
    name: str = "base"
    global_repr: str = "cls"          # "cls" or "avg"

    def __init__(self, arch: Arch, sar_chans: int, opt_chans: int):
        super().__init__()
        self.arch = arch
        kw = dict(img_size=arch.img_size, patch_size=arch.patch_size, embed_dim=arch.embed_dim,
                  depth=arch.depth, num_heads=arch.num_heads, dropout=0.0,           # encoder dropout 0
                  grad_checkpointing=arch.grad_checkpointing)
        self.enc_sar = ViTEncoder(in_chans=sar_chans, **kw)
        self.enc_opt = ViTEncoder(in_chans=opt_chans, **kw)
        self.num_patches = self.enc_sar.num_patches

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        raise NotImplementedError

    @torch.no_grad()
    def global_features(self, encoder: ViTEncoder, x: torch.Tensor) -> torch.Tensor:
        """Native global representation of full images: (B, C, H, W) -> (B, D)."""
        tokens = encoder(x)
        return tokens[:, 0] if self.global_repr == "cls" else tokens[:, 1:].mean(dim=1)


def mlp_projector(in_dim: int, sizes, final_bn: bool, bias: bool = True) -> nn.Sequential:
    """MLP projector: [Linear -> BN -> ReLU] for every hidden size, then a final Linear (+ BN if final_bn).

    sizes = (hidden_1, ..., out). Examples: SimCLR-C / CroCoMAE CLS head (512, 128) with final_bn=True;
    DeCUR / Barlow Twins (8192, 8192, 8192) with final_bn=False and bias=False (official DeCUR code).
    """
    layers, d = [], in_dim
    for h in sizes[:-1]:
        layers += [nn.Linear(d, h, bias=bias), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
        d = h
    layers.append(nn.Linear(d, sizes[-1], bias=bias))
    if final_bn:
        layers.append(nn.BatchNorm1d(sizes[-1]))
    return nn.Sequential(*layers)
