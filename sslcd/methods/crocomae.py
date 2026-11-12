"""
CroCoMAE — Recchia et al., arXiv:2611.14124 (2026).

Three losses, unweighted sum:
1. L_MAE — per-modality masked reconstruction, mask ratio 0.75, the SAME mask for both modalities,
   per-patch normalised targets, 2-layer 128-d decoders. Averaged over the two modalities.
2. L_shd — cross-modal InfoNCE between co-registered *visible* patch tokens, restricted to the shared subspace
   = first `shared_pct` % of the token dimensions (default 50 %). All other visible patches of the batch, of both
   modalities, are negatives. tau = 1.
3. L_cls — intra-modal SimCLR on the CLS token: original image vs augmented view (shared augmentation pipelines), through a
   2-layer projector (512-128, BN), tau = 1; averaged over the two modalities.

Native global representation: CLS token. Dense features: final-layer patch tokens; shared = [..., :K_shd],
unshared = [..., K_shd:].
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..augment import Augmenter
from ..losses import mae_loss, nt_xent
from ..vit import MAEDecoder, patchify, random_masking
from .base import Arch, SSLMethod, mlp_projector


class CroCoMAE(SSLMethod):
    name = "crocomae"
    global_repr = "cls"

    def __init__(self, arch: Arch, sar_chans: int, opt_chans: int, aug_sar: Augmenter, aug_opt: Augmenter,
                 mask_ratio: float = 0.75, shared_pct: float = 50.0, temperature: float = 1.0,
                 decoder_dim: int = 128, decoder_depth: int = 2, decoder_heads: int = 8,
                 cls_proj=(512, 128)):
        super().__init__(arch, sar_chans, opt_chans)
        self.aug_sar, self.aug_opt = aug_sar, aug_opt
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.k_shared = int(arch.embed_dim * shared_pct / 100)   # number of shared dims (192 for 384 and 50 %)
        dec = dict(num_patches=self.num_patches, embed_dim=arch.embed_dim, decoder_dim=decoder_dim,
                   depth=decoder_depth, num_heads=decoder_heads, patch_size=arch.patch_size)
        self.dec_sar = MAEDecoder(in_chans=sar_chans, **dec)
        self.dec_opt = MAEDecoder(in_chans=opt_chans, **dec)
        # CLS projection heads (discarded after pre-training).
        self.proj_sar = mlp_projector(arch.embed_dim, cls_proj, final_bn=True)
        self.proj_opt = mlp_projector(arch.embed_dim, cls_proj, final_bn=True)

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        B, p = sar.shape[0], self.arch.patch_size

        # ---- patch level: one mask shared by both modalities (positives must be co-registered patches) ----
        ids_keep, ids_restore, mask = random_masking(B, self.num_patches, self.mask_ratio, sar.device)
        lat_sar = self.enc_sar.forward_masked(sar, ids_keep)          # (B, 1 + K, D)
        lat_opt = self.enc_opt.forward_masked(opt, ids_keep)

        # L_MAE: average of the two per-modality reconstruction losses.
        loss_mae_sar = mae_loss(self.dec_sar(lat_sar, ids_restore), patchify(sar, p), mask)
        loss_mae_opt = mae_loss(self.dec_opt(lat_opt, ids_restore), patchify(opt, p), mask)
        loss_mae = 0.5 * (loss_mae_sar + loss_mae_opt)

        # L_shd: visible patch tokens (CLS dropped), shared subspace only, stacked over the batch (N = B*K).
        D = lat_sar.shape[-1]
        shd_sar = lat_sar[:, 1:, :self.k_shared].reshape(-1, self.k_shared)
        shd_opt = lat_opt[:, 1:, :self.k_shared].reshape(-1, self.k_shared)
        loss_shd = nt_xent(shd_sar, shd_opt, self.temperature)

        # ---- image level: CLS of the original image vs CLS of an augmented view, per modality ----
        cls = {}
        for m, x, enc, aug, proj in (("sar", sar, self.enc_sar, self.aug_sar, self.proj_sar),
                                     ("opt", opt, self.enc_opt, self.aug_opt, self.proj_opt)):
            h_orig = proj(enc(x)[:, 0])           # (B, 128)
            h_aug = proj(enc(aug(x))[:, 0])
            cls[m] = nt_xent(h_orig, h_aug, self.temperature)
        loss_cls = 0.5 * (cls["sar"] + cls["opt"])

        loss = loss_mae + loss_shd + loss_cls

        # Diagnostics on the current batch: mean cosine of co-registered tokens in each subspace.
        with torch.no_grad():
            cos = nn.functional.cosine_similarity
            un_sar = lat_sar[:, 1:, self.k_shared:].reshape(-1, D - self.k_shared)
            un_opt = lat_opt[:, 1:, self.k_shared:].reshape(-1, D - self.k_shared)
            logs = dict(loss_mae=loss_mae, loss_mae_sar=loss_mae_sar, loss_mae_opt=loss_mae_opt,
                        loss_shd=loss_shd, loss_cls=loss_cls,
                        cos_shared=cos(shd_sar.float(), shd_opt.float()).mean(),
                        cos_unshared=cos(un_sar.float(), un_opt.float()).mean())
        return loss, logs
