"""
CROMA — Fuller, Millard, Green, NeurIPS 2023; official code github.com/antofuller/CROMA (`pretrain_croma.py`).

ViT-Small adaptation used in the paper, trained with the shared engine:

* Two ViT-S encoders (the same backbone as every method), learned absolute position embeddings, no ALiBi.
* Masking 75 %, **independent** masks per modality (the official code also draws independent masks).
* Contrastive: mean of the visible patch tokens -> FFN (LN, Linear 4x, GELU, Linear) -> Linear projection ->
  l2 -> symmetric CLIP loss with learnable temperature (init 0.07, no clamp).
* Cross-modal fusion encoder (depth = encoder depth / 2): self-attention on the SAR tokens (queries),
  cross-attention to the optical tokens, FFN. Modality 1 = SAR, as in CROMA (radar tokens are the queries).
* 1-layer decoder (256-d, fixed 2D sin-cos positions) reconstructing BOTH modalities from the joint tokens;
  target normalised per patch over the concatenation of both modalities; loss on the masked patches of each
  modality; reconstruction loss = sum of the two.
* Total loss = contrastive + reconstruction (equal weights). No augmentation.
* The CLS token stays in the encoder sequence (shared backbone) but no loss uses it.

Encoder dropout 0, as for every method. Native global representation: average pooling.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import clip_loss
from ..vit import patchify, random_masking
from .base import Arch, SSLMethod


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Fixed 2D sin-cos position embedding (MAE / CROMA decoder), shape (grid_size**2, embed_dim), no CLS slot.

    Algorithm as in MAE (He et al., CVPR 2022, `util/pos_embed.py`) and the CROMA official code; independent implementation."""
    def get_1d(dim, pos):
        omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
        omega = 1.0 / 10000 ** omega
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)
    return np.concatenate([get_1d(embed_dim // 2, grid[0]), get_1d(embed_dim // 2, grid[1])], axis=1)


class FFN(nn.Module):
    """Pre-norm feed-forward block (LN -> Linear 4x -> GELU -> Linear)."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(self.norm(x))


class Attention(nn.Module):
    """Pre-norm multi-head attention (self or cross). As in the official CROMA code, cross-attention applies
    the SAME LayerNorm to the queries and to the context."""

    def __init__(self, dim: int, heads: int, cross: bool = False):
        super().__init__()
        self.heads, self.cross = heads, cross
        self.input_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        x = self.input_norm(x)
        context = self.input_norm(context) if self.cross else x
        B, N, D = x.shape
        h = self.heads
        q = self.to_q(x).view(B, N, h, D // h).transpose(1, 2)              # (B, h, N, D/h)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(B, context.shape[1], h, D // h).transpose(1, 2)
        v = v.view(B, context.shape[1], h, D // h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out(out.transpose(1, 2).reshape(B, N, D))


class CrossModalEncoder(nn.Module):
    """CROMA joint encoder: each layer = self-attention on the queries, cross-attention to the context, FFN."""

    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads), Attention(dim, heads, cross=True),
                                                    FFN(dim)]) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        for self_attn, cross_attn, ffn in self.layers:
            x = x + self_attn(x)
            x = x + cross_attn(x, context)
            x = x + ffn(x)
        return self.norm(x)


class SimpleTransformer(nn.Module):
    """Plain pre-norm transformer used by the CROMA decoder."""

    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads), FFN(dim)]) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ffn in self.layers:
            x = x + attn(x)
            x = x + ffn(x)
        return self.norm(x)


class CROMADecoder(nn.Module):
    """Decoder reconstructing both modalities from the joint tokens (placed at the SAR visible positions).

    forward(joint (B, K, D), ids_restore_sar (B, P)) -> (B, P, p*p*(C_sar + C_opt))
    """

    def __init__(self, num_patches, encoder_dim, decoder_dim, decoder_depth, decoder_heads,
                 patch_size, sar_chans, opt_chans):
        super().__init__()
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        pos = get_2d_sincos_pos_embed(decoder_dim, int(num_patches ** 0.5))
        self.register_buffer("pos_embed", torch.from_numpy(pos).float().unsqueeze(0), persistent=False)
        self.decoder = SimpleTransformer(decoder_dim, decoder_depth, decoder_heads)
        self.pred = nn.Linear(decoder_dim, patch_size ** 2 * (sar_chans + opt_chans))

    def forward(self, joint, ids_restore):
        x = self.encoder_to_decoder(joint)
        B, L = ids_restore.shape
        x = torch.cat([x, self.mask_token.expand(B, L - x.shape[1], -1)], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))   # spatial order
        return self.pred(self.decoder(x + self.pos_embed))


class CROMA(SSLMethod):
    name = "croma"
    global_repr = "avg"

    def __init__(self, arch: Arch, sar_chans: int, opt_chans: int, mask_ratio: float = 0.75,
                 decoder_dim: int = 256, decoder_depth: int = 1, decoder_heads: int = 8,
                 init_temperature: float = 0.07):
        super().__init__(arch, sar_chans, opt_chans)
        D = arch.embed_dim
        self.mask_ratio = mask_ratio
        self.sar_chans = sar_chans
        # GAP -> FFN (CROMA's unimodal representations), then linear projections for the contrastive loss.
        self.gap_ffn_sar = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.gap_ffn_opt = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.proj_sar = nn.Linear(D, D)
        self.proj_opt = nn.Linear(D, D)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        self.cross_encoder = CrossModalEncoder(D, depth=arch.depth // 2, heads=arch.num_heads)
        self.decoder = CROMADecoder(self.num_patches, D, decoder_dim, decoder_depth, decoder_heads,
                                    arch.patch_size, sar_chans, opt_chans)

    def reconstruction_loss(self, sar, opt, pred, mask_sar, mask_opt):
        """Per-patch normalised MSE; target normalised over the concatenation of both modalities (CROMA)."""
        p = self.arch.patch_size
        t_sar, t_opt = patchify(sar, p), patchify(opt, p)
        target = torch.cat([t_sar, t_opt], dim=-1).float()
        target = (target - target.mean(dim=-1, keepdim=True)) / (target.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
        pred = pred.float()
        d = t_sar.shape[-1]
        l_sar = ((pred[..., :d] - target[..., :d]) ** 2).mean(dim=-1)
        l_opt = ((pred[..., d:] - target[..., d:]) ** 2).mean(dim=-1)
        l_sar = (l_sar * mask_sar).sum() / mask_sar.sum()     # SAR loss on SAR-masked patches
        l_opt = (l_opt * mask_opt).sum() / mask_opt.sum()     # optical loss on optical-masked patches
        return l_sar + l_opt, l_sar, l_opt

    def forward(self, sar: torch.Tensor, opt: torch.Tensor):
        B = sar.shape[0]
        # Independent masks for the two modalities.
        keep_sar, restore_sar, mask_sar = random_masking(B, self.num_patches, self.mask_ratio, sar.device)
        keep_opt, _, mask_opt = random_masking(B, self.num_patches, self.mask_ratio, opt.device)
        p_sar = self.enc_sar.forward_masked(sar, keep_sar)[:, 1:]     # drop CLS: (B, K, D)
        p_opt = self.enc_opt.forward_masked(opt, keep_opt)[:, 1:]

        # Contrastive loss on pooled unimodal representations.
        z_sar = self.proj_sar(self.gap_ffn_sar(p_sar.mean(dim=1)))
        z_opt = self.proj_opt(self.gap_ffn_opt(p_opt.mean(dim=1)))
        loss_contra = clip_loss(z_sar, z_opt, self.logit_scale)

        # Joint encoding (SAR queries attend to optical) and reconstruction of both modalities.
        joint = self.cross_encoder(p_sar, p_opt)
        pred = self.decoder(joint, restore_sar)
        loss_reco, l_sar, l_opt = self.reconstruction_loss(sar, opt, pred, mask_sar, mask_opt)

        loss = loss_contra + loss_reco
        logs = dict(loss_contra=loss_contra.detach(), loss_reco=loss_reco.detach(), loss_reco_sar=l_sar.detach(),
                    loss_reco_opt=l_opt.detach(), temperature=(1.0 / self.logit_scale.exp()).detach())
        return loss, logs
