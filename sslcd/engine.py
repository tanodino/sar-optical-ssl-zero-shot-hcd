"""
Shared training engine: the SAME protocol for every method.

Protocol
--------
* budget in optimisation **steps**, identical for all methods; batch size 512.
* AdamW, betas (0.9, 0.95), weight decay 0.05 on weight matrices only: no weight decay on biases,
       normalisation layers, position embeddings, CLS / mask tokens and temperature parameters.
* learning rate updated at **every step**: linear warm-up from 0 over the first 5 % of the steps to the
       peak LR (1.5e-4), then a single half-cosine decay to the minimum LR (1e-6); no restarts.
* fp16 mixed precision (autocast + GradScaler), gradient clipping at global norm 1.0, gradient checkpointing.
* fixed seeds (torch, numpy, python); fast non-deterministic kernels.
* no intermediate checkpoints: one rolling `resume.pth` (atomic overwrite every `ckpt_every` steps) and the
       final encoders `encoder_sar.pth` / `encoder_opt.pth`. A run refuses an existing, non-empty output folder
       unless `--resume` is given (no silent auto-resume).
* collapse check at the end of every run -> `collapse_check.json`.

Files written in --out_dir
--------------------------
config.json         every argument + derived values (warm-up steps, parameter counts, environment).
train_log.csv       one row every `log_every` steps: step, lr, loss, method-specific components, speed, memory.
resume.pth          rolling checkpoint (model, optimiser, scaler, step, RNG states).
encoder_sar.pth, encoder_opt.pth   final encoder state dicts (keys as in `vit.ViTEncoder`, no prefixes).
model_info.json     what is needed to rebuild the encoders (architecture, channels, global representation).
collapse_check.json collapse statistics.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch

from .collapse import run_collapse_check
from .data import PairedArrays, infinite_batches
from .methods import build_method
from .methods.base import Arch

# Parameters whose name contains one of these strings never receive weight decay.
NO_DECAY_KEYWORDS = ("pos_embed", "cls_token", "mask_token", "logit_scale")


# --------------------------------------------------------------------------------------------------------
# Optimiser and schedule
# --------------------------------------------------------------------------------------------------------
def param_groups(model: torch.nn.Module, weight_decay: float) -> List[Dict]:
    """Split the trainable parameters into a decayed group and a non-decayed group.

    No decay for: 1-D parameters (biases, LayerNorm / BatchNorm weights) and the named tokens / embeddings /
    temperature listed in NO_DECAY_KEYWORDS. Weight decay for every other (matrix / conv) weight.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name for k in NO_DECAY_KEYWORDS):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at_step(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    """Learning rate used for optimisation step `step` (0-based): linear warm-up then half-cosine.

    step < warmup: peak * (step + 1) / warmup            (first step > 0, last warm-up step = peak)
    otherwise    : min + (peak - min) * 0.5 * (1 + cos(pi * t)), t = (step - warmup) / (total - warmup) in [0, 1)
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


# --------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------
def set_seed(seed: int):
    """Fix every random generator used by the run (non-deterministic CUDA kernels are allowed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> Dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(obj, path: str):
    """Write to a temporary file then rename: a crash during saving never corrupts the previous file."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# --------------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------------
def optical_aug_constants(norm_stats_path, n_opt_channels: int) -> Dict:
    """Per-band optical augmentation constants from `norm_stats.json` (preprocess_corpus.py).

    Returns kwargs for `build_method`: opt_zero_point (brightness pivot), opt_solarize_threshold, opt_solarize_max,
    each a list with one value per optical band, in the band order of the arrays (`s2_bands`).
    Without a file (unit / smoke tests on random data) all are None: zero point 0, solarize disabled.
    """
    if not norm_stats_path:
        print("[augment] no --norm_stats: brightness zero point 0, solarize disabled (tests only)", flush=True)
        return dict(opt_zero_point=None, opt_solarize_threshold=None, opt_solarize_max=None)
    with open(norm_stats_path) as f:
        ns = json.load(f)
    bands = ns["s2_bands"]
    if len(bands) != n_opt_channels:
        raise ValueError(f"norm_stats has {len(bands)} optical bands, data has {n_opt_channels}")
    aug = ns["optical_augmentation"]
    out = dict(opt_zero_point=[aug[b]["zero_point"] for b in bands],
               opt_solarize_threshold=[aug[b]["solarize_threshold"] for b in bands],
               opt_solarize_max=[aug[b]["solarize_max"] for b in bands])
    print(f"[augment] from {norm_stats_path} (bands {bands}): {json.dumps(out)}", flush=True)
    return out


def train(args) -> Dict:
    """Run one pre-training. `args` is the argparse namespace built in `pretrain.py`. Returns a summary dict."""
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp

    # ---- optical augmentation constants are mandatory ----
    # Checked before anything is created on disk: a forgotten --norm_stats would otherwise silently disable
    # solarize and use a wrong brightness pivot. Only tests on random data may skip it, explicitly.
    if bool(args.norm_stats) == bool(args.test_without_norm_stats):
        raise ValueError("give exactly one of --norm_stats <norm_stats.json> (every real run) or "
                         "--test_without_norm_stats (unit / smoke tests on random data only)")

    # ---- output folder policy ----
    os.makedirs(args.out_dir, exist_ok=True)
    resume_path = os.path.join(args.out_dir, "resume.pth")
    if args.resume:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume given but {resume_path} does not exist")
    elif os.listdir(args.out_dir):
        raise FileExistsError(f"output folder {args.out_dir} is not empty; use a new folder or --resume")
    for f in ("encoder_sar.pth", "encoder_opt.pth"):
        if os.path.exists(os.path.join(args.out_dir, f)):
            raise FileExistsError(f"{f} already exists in {args.out_dir}: this run is finished, nothing to do")

    set_seed(args.seed)

    # ---- data ----
    dataset = PairedArrays(args.sar, args.opt, max_samples=args.max_samples)
    arch = Arch(img_size=dataset.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
                depth=args.depth, num_heads=args.num_heads, grad_checkpointing=not args.no_grad_ckpt)
    aug_consts = optical_aug_constants(args.norm_stats, dataset.opt_channels)

    # ---- model ----
    overrides = {}
    if args.method == "crocomae":
        overrides["shared_pct"] = args.shared_pct
    if args.projector_override:                         # unit tests only (small projectors)
        overrides["projector"] = tuple(int(v) for v in args.projector_override.split("-"))
    model = build_method(args.method, arch, dataset.sar_channels, dataset.opt_channels,
                         **aug_consts, **overrides).to(device)
    groups = param_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    warmup_steps = max(1, int(round(args.warmup_frac * args.steps)))

    start_step = 0
    if args.resume:
        ckpt = torch.load(resume_path, map_location="cpu")
        if ckpt["config"]["method"] != args.method or ckpt["config"]["steps"] != args.steps:
            raise ValueError("resume.pth was written by a different configuration (method or steps differ)")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"   # strict loading
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        set_rng_state(ckpt["rng"])
        start_step = ckpt["step"]
        print(f"[resume] restarting at step {start_step}", flush=True)

    # Optional torch.compile; the un-compiled `model` is kept for saving (no `_orig_mod.` prefixes).
    run_model = torch.compile(model) if args.compile else model

    # ---- configuration record ----
    config = dict(vars(args))
    config.update(arch=asdict(arch), warmup_steps=warmup_steps, device=str(device), amp=use_amp,
                  n_samples=len(dataset), sar_channels=dataset.sar_channels, opt_channels=dataset.opt_channels,
                  params_total=count_params(model), params_encoders=count_params(model.enc_sar) +
                  count_params(model.enc_opt), global_repr=model.global_repr,
                  n_params_decay=sum(p.numel() for p in groups[0]["params"]),
                  n_params_no_decay=sum(p.numel() for p in groups[1]["params"]),
                  python=platform.python_version(), torch=torch.__version__, argv=sys.argv)
    if not args.resume:
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    print(json.dumps({k: config[k] for k in ("method", "steps", "batch_size", "lr", "min_lr", "warmup_steps",
                                             "weight_decay", "seed", "n_samples", "params_total",
                                             "params_encoders", "device", "amp")}, indent=1), flush=True)

    # ---- loop ----
    batches = infinite_batches(dataset, args.batch_size, args.seed, start_batch=start_step,
                               num_workers=args.num_workers)
    log_path = os.path.join(args.out_dir, "train_log.csv")
    log_file = open(log_path, "a", newline="")
    writer = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0, last_loss = time.time(), float("nan")
    model.train()
    for step in range(start_step, args.steps):
        lr = lr_at_step(step, args.steps, warmup_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        sar, opt = next(batches)
        sar, opt = sar.to(device, non_blocking=True), opt.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, logs = run_model(sar, opt)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {loss.item()} at step {step}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                                   # clip the true (unscaled) gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        last_loss = loss.item() if (step + 1) % args.log_every == 0 or step + 1 == args.steps else last_loss

        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            elapsed = time.time() - t0
            row = dict(step=step + 1, lr=lr, loss=last_loss, grad_norm=float(grad_norm),
                       **{k: float(v) for k, v in logs.items()},
                       sec_per_step=elapsed / args.log_every if (step + 1) % args.log_every == 0 else float("nan"),
                       peak_mem_gb=torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0)
            if writer is None:
                writer = csv.DictWriter(log_file, fieldnames=list(row))
                if log_file.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            log_file.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
            t0 = time.time()

        if (step + 1) % args.ckpt_every == 0 and step + 1 < args.steps:
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             scaler=scaler.state_dict(), step=step + 1, rng=rng_state(), config=config),
                        resume_path)
    log_file.close()

    # ---- final encoders ----
    torch.save(model.enc_sar.state_dict(), os.path.join(args.out_dir, "encoder_sar.pth"))
    torch.save(model.enc_opt.state_dict(), os.path.join(args.out_dir, "encoder_opt.pth"))
    with open(os.path.join(args.out_dir, "model_info.json"), "w") as f:
        json.dump(dict(method=args.method, arch=asdict(arch), sar_channels=dataset.sar_channels,
                       opt_channels=dataset.opt_channels, global_repr=model.global_repr,
                       shared_pct=args.shared_pct if args.method == "crocomae" else None), f, indent=2)

    # ---- collapse check ----
    stats = run_collapse_check(model, dataset, device, n_images=args.collapse_images, seed=args.seed)
    with open(os.path.join(args.out_dir, "collapse_check.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[collapse check]", json.dumps(stats, indent=1), flush=True)
    return dict(final_loss=last_loss, collapse=stats)
