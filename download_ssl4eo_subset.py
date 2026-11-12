"""
Download the selected SSL4EO-S12 v1.1 shards (Hugging Face, embed2scale/SSL4EO-S12-v1.1).

Reads the shard list produced by exclude_leakage.py and downloads, for every shard, the files
train/<MODALITY>/<shard> from the HuggingFace dataset embed2scale/SSL4EO-S12-v1.1.
Files already complete in --local_dir are skipped by huggingface_hub (safe to re-run after an interruption);
nothing is deleted or overwritten.

Usage:
    python download_ssl4eo_subset.py --shards subset_v1/shards.txt \
        --local_dir data/pretrain/ssl4eo_s12_v1.1 --modalities S1GRD,S2L1C
"""
import argparse
import time

from huggingface_hub import snapshot_download

REPO = "embed2scale/SSL4EO-S12-v1.1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="text file, one shard name per line")
    ap.add_argument("--local_dir", required=True)
    ap.add_argument("--modalities", default="S1GRD,S2L1C")
    ap.add_argument("--max_workers", type=int, default=8)
    args = ap.parse_args()
    print(vars(args), flush=True)
    shards = [s.strip() for s in open(args.shards) if s.strip()]
    patterns = [f"train/{m}/{s}" for m in args.modalities.split(",") for s in shards]
    print(f"{len(shards)} shards x {len(args.modalities.split(','))} modalities = {len(patterns)} files", flush=True)
    t0 = time.time()
    snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=patterns, local_dir=args.local_dir,
                      max_workers=args.max_workers)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
