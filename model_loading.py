"""
Strict loading of a pre-trained framework (DeCUR, CROMA or CroCoMAE) for zero-shot change detection.

A weights folder contains:
  model_info.json   method, architecture, number of SAR / optical channels
  config.json       pre-training configuration (as written by pretrain.py)
  model.pth         released weights: {"model": state_dict of the whole method module (encoders + heads), "step": int}
                    (a full training checkpoint `resume.pth` with the same "model" / "step" keys is also accepted)
  norm_stats.json   normalisation statistics of the pre-training corpus (also read from the path stored in config.json
                    when this file is absent)

The method module is rebuilt exactly as during pre-training and every key must match (no missing, no unexpected key).
"""
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sslcd.methods import build_method                # noqa: E402
from sslcd.methods.base import Arch                   # noqa: E402
from sslcd.engine import optical_aug_constants        # noqa: E402

ARCH_KEYS = ("img_size", "patch_size", "embed_dim", "depth", "num_heads")


def strict_load(module: nn.Module, sd: dict, what: str):
    """Load a state dict with strict key matching (prefixes added by torch.compile are removed first)."""
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"{what}: missing {missing}, unexpected {unexpected}"


def load_model(run: str, info: dict):
    """Rebuild the method module of the weights folder `run` and load its weights strictly -> (model in eval mode, step)."""
    cfg = json.load(open(os.path.join(run, "config.json")))
    ns = os.path.join(run, "norm_stats.json")
    if not os.path.exists(ns):
        ns = cfg["norm_stats"]
    arch = Arch(**{k: info["arch"][k] for k in ARCH_KEYS}, grad_checkpointing=False)
    model = build_method(info["method"], arch, info["sar_channels"], info["opt_channels"],
                         **optical_aug_constants(ns, info["opt_channels"]))
    path = os.path.join(run, "model.pth")
    if not os.path.exists(path):
        path = os.path.join(run, "resume.pth")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    strict_load(model, ck["model"], f"{info['method']} {os.path.basename(path)}")
    return model.eval(), int(ck["step"])
