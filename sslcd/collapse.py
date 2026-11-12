"""
Representation-collapse check, run on frozen encoders at the end of pre-training.

For a set of features X (N, D) we report:
* std_mean / std_min : per-dimension standard deviation across samples (mean and minimum over dimensions);
                       ~0 means the features no longer vary between inputs.
* cos_mean           : mean pairwise cosine similarity between different samples; ~1 means all inputs map to
                       (almost) the same direction.
* eff_rank           : effective rank (Roy & Vetterli, 2007) = exp(entropy of the normalised singular values of
                       the centred matrix); a collapsed model has an effective rank of a few units, a healthy
                       one a large fraction of D.
* cos_centred_mean   : mean pairwise cosine after subtracting the mean feature vector. A large constant offset shared by all samples makes
                       cos_mean ~1 although the samples still differ (DeCUR SAR dry run: cos_mean 0.985,
                       cos_centred_mean 0.02, eff_rank 52); a true collapse has small std AND small eff_rank.
* offset_to_spread   : ||mean vector|| / mean ||x_i - mean vector||; >> 1 flags such an offset.

The check is computed, per modality, on (a) the method's native global representation (CLS or average
pooling, see `SSLMethod.global_repr`) and (b) individual final-layer patch tokens.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def collapse_stats(x: torch.Tensor) -> Dict[str, float]:
    """Collapse statistics of a feature matrix x (N, D); computed in float64 on CPU."""
    x = x.detach().double().cpu()
    std = x.std(dim=0)
    n = x.shape[0]

    def mean_pairwise_cos(z: torch.Tensor) -> torch.Tensor:
        zn = F.normalize(z, dim=1)
        cos = zn @ zn.t()
        return (cos.sum() - cos.diagonal().sum()) / (n * (n - 1))       # exclude self-similarity

    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu                                                          # centred features
    cos_mean = mean_pairwise_cos(x)
    cos_centred = mean_pairwise_cos(xc)
    offset_to_spread = mu.norm() / xc.norm(dim=1).mean().clamp_min(1e-12)
    s = torch.linalg.svdvals(xc)
    p = s / s.sum()
    p = p[p > 0]
    eff_rank = torch.exp(-(p * p.log()).sum())
    return dict(std_mean=float(std.mean()), std_min=float(std.min()), cos_mean=float(cos_mean),
                cos_centred_mean=float(cos_centred), offset_to_spread=float(offset_to_spread),
                eff_rank=float(eff_rank), dim=int(x.shape[1]), n=int(n))


@torch.no_grad()
def run_collapse_check(model, dataset, device, n_images: int = 3000, batch_size: int = 256,
                       n_patch_tokens: int = 3000, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Encode `n_images` random images of `dataset` with the frozen encoders of `model` and return the stats.

    Returns {"sar_global": {...}, "opt_global": {...}, "sar_patch": {...}, "opt_patch": {...}}.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=min(n_images, len(dataset)), replace=False))
    feats = {k: [] for k in ("sar_global", "opt_global", "sar_patch", "opt_patch")}
    for i in range(0, len(idx), batch_size):
        batch = [dataset[j] for j in idx[i:i + batch_size]]
        sar = torch.stack([b[0] for b in batch]).to(device)
        opt = torch.stack([b[1] for b in batch]).to(device)
        for m, x, enc in (("sar", sar, model.enc_sar), ("opt", opt, model.enc_opt)):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                tokens = enc(x).float()                                   # (B, 1 + P, D)
            glob = tokens[:, 0] if model.global_repr == "cls" else tokens[:, 1:].mean(dim=1)
            feats[f"{m}_global"].append(glob.cpu())
            feats[f"{m}_patch"].append(tokens[:, 1:].reshape(-1, tokens.shape[-1]).cpu())
    out = {}
    for k, v in feats.items():
        x = torch.cat(v)
        if k.endswith("patch") and x.shape[0] > n_patch_tokens:          # subsample individual patch tokens
            sel = torch.from_numpy(rng.choice(x.shape[0], size=n_patch_tokens, replace=False))
            x = x[sel]
        out[k] = collapse_stats(x)
    out["global_repr"] = model.global_repr
    return out
