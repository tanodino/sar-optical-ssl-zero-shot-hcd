"""
ViT encoder, random masking and MAE decoder shared by all methods.

Design
------
* ViT-Small for every method: embedding 384, 12 layers, 6 heads, MLP ratio 4, pre-norm blocks.
* position embedding and initialisation: a *learned* absolute position embedding (CLS slot included),
  trunc-normal(0.02) initialisation of the position embedding, the CLS token and the patch-embedding
  convolution; PyTorch default initialisation for the other layers.
* encoder dropout is 0 during pre-training.
* Saved encoders are always loaded with strict key matching.
* Gradient checkpointing is applied layer by layer inside `forward_tokens` when enabled and training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# --------------------------------------------------------------------------------------------------------
# Random masking (MAE-style). Algorithm as in MAE (He et al., CVPR 2022, `models_mae.py::random_masking`);
# independent implementation.
# --------------------------------------------------------------------------------------------------------
def random_masking(batch_size: int, num_patches: int, mask_ratio: float, device, generator=None):
    """Draw an independent random mask for every image of the batch.

    Args:
        batch_size:  B.
        num_patches: P, number of patch tokens per image.
        mask_ratio:  fraction of patches that are removed (e.g. 0.75).
        device:      torch device of the returned tensors.
        generator:   optional torch.Generator (for reproducible unit tests).

    Returns:
        ids_keep:    (B, K) long, indices of the K = int(P * (1 - mask_ratio)) visible patches.
        ids_restore: (B, P) long, permutation that puts [visible tokens, mask tokens] back in spatial order.
        mask:        (B, P) float, 1 = masked (removed) patch, 0 = visible patch, in spatial order.
    """
    num_keep = int(num_patches * (1.0 - mask_ratio))
    # One uniform noise value per patch; sorting it gives a random permutation per image.
    noise = torch.rand(batch_size, num_patches, device=device, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)          # random order of the patches
    ids_restore = torch.argsort(ids_shuffle, dim=1)    # inverse permutation
    ids_keep = ids_shuffle[:, :num_keep]               # the first K patches of the random order are kept
    # Binary mask in "shuffled" order (0 for the kept ones), then un-shuffled to spatial order.
    mask = torch.ones(batch_size, num_patches, device=device)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return ids_keep, ids_restore, mask


def gather_tokens(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Select tokens along the sequence axis.

    Args:
        x:   (B, P, D) token sequence.
        ids: (B, K) indices into the P axis.
    Returns:
        (B, K, D) selected tokens.
    """
    return torch.gather(x, dim=1, index=ids.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


# --------------------------------------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and project each patch to a D-dim token.

    A Conv2d with kernel = stride = patch size is equivalent to "flatten each patch + linear layer".
    Input (B, C, H, W) -> output (B, P, D) with P = (H / patch) * (W / patch), row-major patch order.
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be a multiple of the patch size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                        # (B, D, H/p, W/p)
        return x.flatten(2).transpose(1, 2)     # (B, P, D)


class ViTEncoder(nn.Module):
    """ViT encoder for one modality (ViT-Small by default).

    Output of every forward function: token sequence (B, 1 + N, D) where index 0 is the CLS token,
    after the final LayerNorm. N = P for full images, N = K for masked images.
    """

    def __init__(self, img_size: int = 64, patch_size: int = 8, in_chans: int = 2, embed_dim: int = 384,
                 depth: int = 12, num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 grad_checkpointing: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grad_checkpointing = grad_checkpointing

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        # Learned CLS token and learned absolute position embedding (slot 0 = CLS).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Pre-norm transformer blocks (norm_first=True), GELU MLP, batch-first tensors.
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                           dim_feedforward=int(embed_dim * mlp_ratio), dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        # enable_nested_tensor=False: nested tensors are an inference fast path we do not need.
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        """Trunc-normal(0.02) initialisation of the position embedding, CLS token and patch embedding; PyTorch defaults elsewhere."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

    # ---- internal helpers -------------------------------------------------------------------------------
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embedding + position embedding of the patch slots. (B, C, H, W) -> (B, P, D)."""
        return self.patch_embed(x) + self.pos_embed[:, 1:, :]

    def _add_cls_and_encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Prepend the CLS token (with its own position embedding) and run the transformer.

        tokens: (B, N, D) patch tokens that already carry their position embedding.
        returns: (B, 1 + N, D) normalised tokens.
        """
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(tokens.shape[0], -1, -1)
        x = self.dropout(torch.cat([cls, tokens], dim=1))
        for layer in self.transformer.layers:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                # Recompute the layer in the backward pass instead of storing its activations (saves memory).
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x)

    # ---- public API -------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full (unmasked) images. (B, C, H, W) -> (B, 1 + P, D)."""
        return self._add_cls_and_encode(self._embed(x))

    def forward_masked(self, x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """Encode only the visible patches given by `ids_keep` (B, K). (B, C, H, W) -> (B, 1 + K, D).

        The position embedding is added *before* the selection, so every visible token keeps the
        position of the patch it comes from.
        """
        return self._add_cls_and_encode(gather_tokens(self._embed(x), ids_keep))


# --------------------------------------------------------------------------------------------------------
# MAE decoder (per modality)
# --------------------------------------------------------------------------------------------------------
class MAEDecoder(nn.Module):
    """Lightweight MAE decoder: reconstructs the pixels of every patch from the visible tokens.

    Used by CroCoMAE and MAE (2 layers, 128-d, 8 heads). Same structure as the old `model.py::MAEDecoder`
    (learned decoder position embedding with a CLS slot), with dropout 0.

    forward(latent (B, 1 + K, D_enc), ids_restore (B, P)) -> prediction (B, P, patch_size**2 * in_chans)
    """

    def __init__(self, num_patches: int, embed_dim: int = 384, decoder_dim: int = 128, depth: int = 2,
                 num_heads: int = 8, mlp_ratio: float = 4.0, patch_size: int = 8, in_chans: int = 2):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))
        layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=num_heads,
                                           dim_feedforward=int(decoder_dim * mlp_ratio), dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size ** 2 * in_chans)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(latent)                                    # (B, 1 + K, Dd)
        B, P = ids_restore.shape
        mask_tokens = self.mask_token.expand(B, P - (x.shape[1] - 1), -1)  # one token per removed patch
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)                 # (B, P, Dd), shuffled order
        x_ = gather_tokens(x_, ids_restore)                                # back to spatial order
        x = torch.cat([x[:, :1, :], x_], dim=1) + self.decoder_pos_embed   # re-attach CLS, add positions
        x = self.decoder_norm(self.blocks(x))
        return self.decoder_pred(x)[:, 1:, :]                              # drop CLS -> (B, P, p*p*C)


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, P, C * p * p), same patch order and channel layout as `PatchEmbed`/`F.unfold`."""
    return torch.nn.functional.unfold(imgs, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
