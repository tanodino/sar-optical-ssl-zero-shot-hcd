"""
DeCUR — Wang et al., "Decoupling Common and Unique Representations for Multimodal Self-supervised Learning",
ECCV 2024 (arXiv 2309.05300); official code github.com/zhu-xlab/DeCUR, `src/pretrain/models/decur.py`,
`pretrain_mm.py`. Adapted to ViT-Small encoders (the paper uses ResNet-50).

* Each modality: two augmented views (shared augmentation pipelines) -> encoder -> CLS token -> its OWN projector
  (8192-8192-8192: [Linear, BN, ReLU] x 2 + Linear, no bias) -> batch standardisation (BN without affine).
* Intra-modal Barlow Twins between the two views of the same modality: L_M1 (SAR), L_M2 (optical).
* Cross-modal terms between view 1 of SAR and view 1 of optical (as in the official code):
  common block (first 87.5 % = 7168 dims) aligned, unique block (last 1024 dims) decorrelated.
* Total loss as in the official code: (L_M1 + L_M2 + (L_com + L_uni) / 2) / 3, lambda = 0.0051 for all terms.

Native global representation: CLS token; the projector output (common part) is the aligned vector.
Note: the official code shares the projector object between modalities; the paper says each branch has its
own projector; we follow the paper (separate projectors).
"""
from __future__ import annotations

import torch

from ..augment import Augmenter
from ..losses import barlow_twins_loss, decur_cross_losses
from .base import Arch, SSLMethod, mlp_projector


class DeCUR(SSLMethod):
    name = "decur"
    global_repr = "cls"

    def __init__(self, arch: Arch, sar_chans: int, opt_chans: int, aug_sar: Augmenter, aug_opt: Augmenter,
                 projector=(8192, 8192, 8192), common_frac: float = 0.875, lambd: float = 0.0051):
        super().__init__(arch, sar_chans, opt_chans)
        self.aug_sar, self.aug_opt = aug_sar, aug_opt
        self.lambd = lambd
        self.dim_common = int(round(projector[-1] * common_frac))     # 7168 for 8192 and 87.5 %
        self.proj_sar = mlp_projector(arch.embed_dim, projector, final_bn=False, bias=False)
        self.proj_opt = mlp_projector(arch.embed_dim, projector, final_bn=False, bias=False)

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        # Two independent augmented views per modality.
        z_sar1 = self.proj_sar(self.enc_sar(self.aug_sar(sar))[:, 0])
        z_sar2 = self.proj_sar(self.enc_sar(self.aug_sar(sar))[:, 0])
        z_opt1 = self.proj_opt(self.enc_opt(self.aug_opt(opt))[:, 0])
        z_opt2 = self.proj_opt(self.enc_opt(self.aug_opt(opt))[:, 0])

        # Intra-modal Barlow Twins (the standardisation is done inside the loss).
        loss_m1 = barlow_twins_loss(z_sar1, z_sar2, self.lambd)
        loss_m2 = barlow_twins_loss(z_opt1, z_opt2, self.lambd)
        # Cross-modal common / unique terms on view 1 of each modality.
        loss_com, loss_uni = decur_cross_losses(z_sar1, z_opt1, self.dim_common, self.lambd)

        loss = (loss_m1 + loss_m2 + 0.5 * (loss_com + loss_uni)) / 3.0
        logs = dict(loss_m1_sar=loss_m1.detach(), loss_m2_opt=loss_m2.detach(),
                    loss_common=loss_com.detach(), loss_unique=loss_uni.detach())
        return loss, logs
