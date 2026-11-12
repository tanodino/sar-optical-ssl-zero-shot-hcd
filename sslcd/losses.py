"""
Loss functions shared by the methods.

All losses are computed in float32 (inputs are cast), even when the forward pass runs in fp16, because
softmax / cross-correlation statistics are sensitive to precision.

Sources
-------
* NT-Xent / InfoNCE with one positive per sample: SimCLR (Chen et al., ICML 2020); CroCoMAE paper Eq. (InfoNCE).
* CLIP loss with learnable temperature: CROMA official code `pretrain_croma.py` (symmetric cross-entropy).
* Barlow Twins: Zbontar et al., ICML 2021 (cross-correlation of batch-standardised embeddings).
* DeCUR common / unique terms: Wang et al., ECCV 2024, Eqs. (2)-(3); official code `models/decur.py`.
* MAE reconstruction with per-patch normalised targets: He et al., CVPR 2022, `models_mae.py::forward_loss`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------------------
# Contrastive losses
# --------------------------------------------------------------------------------------------------------
def nt_xent(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """NT-Xent (InfoNCE) loss between two aligned sets of embeddings.

    Row i of `z_a` and row i of `z_b` form the positive pair. All the other 2N - 2 embeddings of the
    concatenated set [z_a; z_b] are negatives (SimCLR formulation; the CroCoMAE paper's InfoNCE with
    exactly one positive per sample). Embeddings are l2-normalised inside the function.

    Args:
        z_a, z_b:    (N, D) embeddings (e.g. CLS projections of two views, or co-registered patch tokens).
        temperature: tau; the logits are cosine similarities divided by tau.
    Returns:
        scalar loss, averaged over the 2N anchors.
    """
    n = z_a.shape[0]
    z = F.normalize(torch.cat([z_a, z_b], dim=0).float(), dim=1)    # (2N, D), unit norm
    logits = z @ z.t() / temperature                                  # (2N, 2N) cosine / tau
    # An anchor is never compared with itself: remove the diagonal from the softmax.
    logits.fill_diagonal_(float("-inf"))
    # The positive of anchor i (in the first half) is i + N, and vice versa.
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(logits, targets)


def clip_loss(z_a: torch.Tensor, z_b: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric CLIP loss (CROMA): cross-entropy a->b and b->a with in-batch negatives.

    Args:
        z_a, z_b:    (N, D) embeddings of the two modalities (row i of each = same location).
        logit_scale: scalar parameter storing log(1 / temperature); used as exp(logit_scale) without clamp,
                     as in the official CROMA code.
    """
    z_a = F.normalize(z_a.float(), dim=-1)
    z_b = F.normalize(z_b.float(), dim=-1)
    logits = logit_scale.exp().float() * z_a @ z_b.t()               # (N, N)
    targets = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


# --------------------------------------------------------------------------------------------------------
# Redundancy-reduction losses
# --------------------------------------------------------------------------------------------------------
def batch_standardize(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Standardise every feature over the batch: (z - mean) / sqrt(var + eps).

    This is exactly what `nn.BatchNorm1d(D, affine=False)` does in training mode (biased variance),
    which is how Barlow Twins and DeCUR normalise the projector outputs before the cross-correlation.
    """
    z = z.float()
    mean = z.mean(dim=0, keepdim=True)
    var = z.var(dim=0, unbiased=False, keepdim=True)
    return (z - mean) / torch.sqrt(var + eps)


def cross_correlation(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Empirical cross-correlation matrix C = std(z_a)^T std(z_b) / N, shape (D, D)."""
    n = z_a.shape[0]
    return batch_standardize(z_a).t() @ batch_standardize(z_b) / n


def _off_diagonal_sq_sum(c: torch.Tensor) -> torch.Tensor:
    """Sum of the squared off-diagonal entries of a square matrix."""
    return (c ** 2).sum() - (torch.diagonal(c) ** 2).sum()


def barlow_twins_loss(z_a: torch.Tensor, z_b: torch.Tensor, lambd: float = 0.0051) -> torch.Tensor:
    """Barlow Twins: diagonal of C pushed to 1 (invariance), off-diagonal pushed to 0 (redundancy reduction).

    L = sum_i (1 - C_ii)^2 + lambd * sum_{i != j} C_ij^2       (Zbontar et al., Eq. 1; DeCUR Eqs. 4-5)
    """
    c = cross_correlation(z_a, z_b)
    on_diag = ((1.0 - torch.diagonal(c)) ** 2).sum()
    return on_diag + lambd * _off_diagonal_sq_sum(c)


def decur_cross_losses(z_a: torch.Tensor, z_b: torch.Tensor, dim_common: int, lambd: float = 0.0051):
    """DeCUR cross-modal terms on the projector outputs of the two modalities.

    The first `dim_common` dimensions are the *common* part (aligned across modalities: diagonal -> 1),
    the remaining dimensions are the *unique* part (decorrelated across modalities: diagonal -> 0).
    Off-diagonal entries are pushed to 0 in both blocks. (DeCUR Eqs. 2-3; `models/decur.py::bt_loss_cross`.)

    Returns:
        (loss_common, loss_unique), two scalars.
    """
    c = cross_correlation(z_a, z_b)
    c_c = c[:dim_common, :dim_common]          # common block
    c_u = c[dim_common:, dim_common:]          # unique block
    loss_common = ((1.0 - torch.diagonal(c_c)) ** 2).sum() + lambd * _off_diagonal_sq_sum(c_c)
    loss_unique = (torch.diagonal(c_u) ** 2).sum() + lambd * _off_diagonal_sq_sum(c_u)
    return loss_common, loss_unique


# --------------------------------------------------------------------------------------------------------
# Masked reconstruction
# --------------------------------------------------------------------------------------------------------
def mae_loss(pred: torch.Tensor, target_patches: torch.Tensor, mask: torch.Tensor,
             norm_pix: bool = True) -> torch.Tensor:
    """MAE loss: mean squared error on the *masked* patches only.

    Args:
        pred:           (B, P, p*p*C) decoder prediction.
        target_patches: (B, P, p*p*C) patchified input image (see `vit.patchify`).
        mask:           (B, P) 1 = masked patch (counted in the loss), 0 = visible patch.
        norm_pix:       if True, each target patch is normalised by its own mean and std
                        (MAE `norm_pix_loss`, used by the old CroCoMAE/MAE code and by the MAE paper's results).
    """
    target = target_patches.float()
    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
    per_patch = ((pred.float() - target) ** 2).mean(dim=-1)        # (B, P) mean over the pixels of a patch
    return (per_patch * mask).sum() / mask.sum()
