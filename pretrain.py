"""
Single entry point for the pre-training of every method.

Example (one GPU):

    CUDA_VISIBLE_DEVICES=2 python pretrain.py \
        --method crocomae \
        --sar /path/SAR_data_normalized.npy --opt /path/MS_data_normalized.npy \
        --out_dir pretrain/crocomae --steps 100000 --seed 0

All defaults below are the protocol of the paper.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sslcd.engine import train          # noqa: E402
from sslcd.methods import METHODS       # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # what / where
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--sar", required=True, help="SAR array (N, C_sar, H, W) float32 .npy (memory-mapped)")
    p.add_argument("--opt", required=True, help="optical array (N, C_opt, H, W) float32 .npy, same N and order")
    p.add_argument("--out_dir", required=True, help="output folder; must be new/empty unless --resume")
    p.add_argument("--resume", action="store_true", help="continue from out_dir/resume.pth")
    p.add_argument("--seed", type=int, default=0)
    # budget and optimisation
    p.add_argument("--steps", type=int, required=True, help="total optimisation steps")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1.5e-4, help="peak learning rate")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_frac", type=float, default=0.05, help="warm-up length as a fraction of --steps")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=1.0)
    # architecture (ViT-Small, 8x8 patches by default; image size is read from the data)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=6)
    # method-specific
    p.add_argument("--shared_pct", type=float, default=50.0, help="CroCoMAE: % of token dims in the shared subspace")
    p.add_argument("--norm_stats", default=None,
                   help="norm_stats.json (preprocess_corpus.py): per-band optical augmentation constants (brightness zero point, "
                        "solarize threshold/max). MANDATORY for every real run")
    p.add_argument("--test_without_norm_stats", action="store_true",
                   help="tests on random data only: run without --norm_stats (solarize off, zero point 0)")
    p.add_argument("--projector_override", default=None, help=argparse.SUPPRESS)   # unit tests only
    # engineering
    p.add_argument("--device", default=None, help="default: cuda if available")
    p.add_argument("--no_amp", action="store_true", help="disable fp16 (debug only)")
    p.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing (debug only)")
    p.add_argument("--compile", action="store_true", help="use torch.compile")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--max_samples", type=int, default=None, help="use only the first N samples (dry runs)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=2000, help="steps between rolling resume.pth saves")
    p.add_argument("--collapse_images", type=int, default=3000)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
