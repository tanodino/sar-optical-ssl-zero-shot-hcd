"""
Registry of the three pre-training methods evaluated in the paper: CroCoMAE, DeCUR, CROMA.

`build_method(name, arch, sar_chans, opt_chans, **overrides)` returns the model; methods that use
augmentations (CroCoMAE, DeCUR) receive the shared augmentation pipelines.
`overrides` is only meant for unit tests (e.g. small projectors) and for method-specific arguments
exposed on the command line (e.g. `shared_pct` for CroCoMAE).
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..augment import Augmenter
from .base import Arch, SSLMethod
from .crocomae import CroCoMAE
from .croma import CROMA
from .decur import DeCUR

METHODS = {cls.name: cls for cls in (CroCoMAE, DeCUR, CROMA)}
USES_AUGMENTATION = {"crocomae", "decur"}


def build_method(name: str, arch: Arch, sar_chans: int, opt_chans: int,
                 opt_zero_point: Optional[Sequence[float]] = None,
                 opt_solarize_threshold: Optional[Sequence[float]] = None,
                 opt_solarize_max: Optional[Sequence[float]] = None, **overrides) -> SSLMethod:
    """Build a method; the optical augmentation constants (per band, normalised units, from norm_stats.json,
    see norm_stats.json) are only used by the methods with augmentations."""
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; choose one of {sorted(METHODS)}")
    kwargs = dict(overrides)
    if name in USES_AUGMENTATION:
        kwargs["aug_sar"] = Augmenter.sar()
        kwargs["aug_opt"] = Augmenter.optical(zero_point=opt_zero_point, solarize_threshold=opt_solarize_threshold,
                                              solarize_max=opt_solarize_max)
    return METHODS[name](arch, sar_chans, opt_chans, **kwargs)
