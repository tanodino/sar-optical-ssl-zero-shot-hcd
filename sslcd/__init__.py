"""
sslcd — clean re-implementation of the self-supervised pre-training methods used in the
zero-shot SAR-optical change-detection study.

Package layout
--------------
vit.py        ViT-Small encoder (one per modality), random masking helpers, MAE decoder.
augment.py    GPU augmentation pipelines for SAR and optical images.
losses.py     Loss functions shared by the methods (NT-Xent/InfoNCE, CLIP, Barlow Twins, DeCUR, MAE).
data.py       Memory-mapped paired SAR/optical dataset and an infinite, seeded batch iterator.
engine.py     Shared training engine (optimiser, schedule, fp16, clipping, checkpoints, logging).
collapse.py   Representation-collapse diagnostics run at the end of every pre-training.
methods/      One module per method; each contains only its model heads and its loss.

Conventions
-----------
* Modality order is always (SAR, optical). SAR images have 2 channels (VV, VH); optical images have the
  bands B2, B3, B4, B8. Both are co-registered and share the same spatial size.
* Tensor shapes are written as (B, C, H, W) for images and (B, L, D) for token sequences, where
  L = number of tokens (1 CLS token + P patch tokens, or 1 + K visible tokens when masked) and D = embedding size.
"""
