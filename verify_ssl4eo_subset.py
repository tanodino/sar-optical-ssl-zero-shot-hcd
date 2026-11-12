"""
Read-only verification of the downloaded SSL4EO-S12 v1.1 subset (after the download).

Checks, for every shard listed in shards.txt and every modality:
1. the file exists and its size equals the size published on HuggingFace (file metadata API);
2. the set of Zarr samples in the shard equals the metadata (train_metadata.parquet) for that shard, and is the
   same in S1GRD and S2L1C (S1/S2 pairing by location);
3. for every kept sample (kept_samples.csv): the acquisition times stored **in the files** equal the metadata
   (S1_time_k / S2_time_k) and are all earlier than the earliest test event (temporal separation).
Only the small `time` / `sample` entries of each Zarr zip are read (the tar members are seekable), so the check
does not decode the images.

Usage:
    python verify_ssl4eo_subset.py --root data/pretrain/ssl4eo_s12_v1.1 --subset subset_v1 \
        --earliest_test_event 2026-01-06T19:00:00 --json reports/ssl4eo_subset_v1_verification.json
"""
import argparse
import collections
import json
import os
import tarfile
import zipfile

import numpy as np
import pandas as pd
import zarr
from huggingface_hub import HfApi

REPO = "embed2scale/SSL4EO-S12-v1.1"


def read_small(zf: zipfile.ZipFile, names):
    """Open only the listed arrays of a Zarr zip (plus group metadata)."""
    keep = [n for n in zf.namelist() if n.split("/")[0] in names or n.startswith(".z")]
    return zarr.open_group(store={n: zf.read(n) for n in keep}, mode="r")


def decode_time(arr) -> np.ndarray:
    """Decode a CF-encoded Zarr time array ("<unit> since <reference>") to datetime64[ns].

    Most samples use "nanoseconds since 1970-01-01", but some use e.g. "seconds since 2019-12-25T06:02:41"
    (found by the first verification run, report ssl4eo_subset_v1_verification.json)."""
    unit, ref = arr.attrs["units"].split(" since ")
    step = {"nanoseconds": "ns", "microseconds": "us", "milliseconds": "ms", "seconds": "s", "minutes": "m",
            "hours": "h", "days": "D"}[unit.strip()]
    return (np.datetime64(pd.Timestamp(ref.strip()).to_datetime64(), "ns")
            + arr[:].astype(np.int64).astype(f"timedelta64[{step}]").astype("timedelta64[ns]"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset", required=True, help="folder written by exclude_leakage.py")
    ap.add_argument("--meta", default=None, help="default: <root>/train_metadata.parquet")
    ap.add_argument("--modalities", default="S1GRD,S2L1C")
    ap.add_argument("--earliest_test_event", required=True)
    ap.add_argument("--json", default=None, help="optional output JSON (must not exist)")
    args = ap.parse_args()
    print(vars(args), flush=True)
    mods = args.modalities.split(",")
    limit = np.datetime64(pd.Timestamp(args.earliest_test_event).to_datetime64())
    shards = [s.strip() for s in open(os.path.join(args.subset, "shards.txt")) if s.strip()]
    kept = pd.read_csv(os.path.join(args.subset, "kept_samples.csv"), dtype={"sample_id": str})
    meta = pd.read_parquet(args.meta or os.path.join(args.root, "train_metadata.parquet"))
    meta = meta[meta.tar.isin(shards)].set_index("zarr")
    kept_by_zarr = kept.set_index("zarr")

    hf_size = {s.rfilename: s.size for s in HfApi().dataset_info(REPO, files_metadata=True).siblings}
    problems = collections.defaultdict(list)
    n_time_checked = collections.Counter()
    max_time = {m: np.datetime64("1900-01-01") for m in mods}

    for k, shard in enumerate(shards):
        names = {}
        for m in mods:
            rel = f"train/{m}/{shard}"
            path = os.path.join(args.root, rel)
            if not os.path.exists(path):
                problems["missing_file"].append(rel)
                continue
            if os.path.getsize(path) != hf_size.get(rel):
                problems["size_mismatch"].append((rel, os.path.getsize(path), hf_size.get(rel)))
                continue
            tcol = "S1_time_" if m.startswith("S1") else "S2_time_"
            names[m] = set()
            with tarfile.open(path) as tf:
                for mem in tf:
                    if not mem.isfile():
                        continue
                    names[m].add(mem.name)
                    if mem.name not in kept_by_zarr.index:
                        continue
                    g = read_small(zipfile.ZipFile(tf.extractfile(mem)), {"time", "sample"})
                    t = decode_time(g["time"])                                      # (4,) datetime64[ns]
                    mt = np.array([meta.loc[mem.name, f"{tcol}{i}"] for i in range(4)], dtype="datetime64[ns]")
                    if str(g["sample"][()]) != str(meta.loc[mem.name, "sample_id"]):
                        problems["sample_id_mismatch"].append((m, mem.name))
                    # allow 1 s rounding between file and parquet
                    if np.abs((t - mt).astype("timedelta64[s]").astype(np.int64)).max() > 1:
                        problems["time_mismatch"].append((m, mem.name, str(t), str(mt)))
                    max_time[m] = max(max_time[m], t.max())
                    if (t >= limit).any():
                        problems["temporal_leak"].append((m, mem.name, str(t.max())))
                    n_time_checked[m] += 1
            if names[m] != set(meta[meta.tar == shard].index):
                problems["content_vs_metadata"].append((m, shard))
        if len(names) == len(mods) and len({frozenset(v) for v in names.values()}) != 1:
            problems["s1_s2_content_differs"].append(shard)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(shards)} shards checked", flush=True)

    rep = dict(n_shards=len(shards), modalities=mods, n_kept=len(kept),
               n_time_checked=dict(n_time_checked), max_time_in_files={m: str(v) for m, v in max_time.items()},
               earliest_test_event=args.earliest_test_event,
               problems={k: v[:50] for k, v in problems.items()},
               n_problems={k: len(v) for k, v in problems.items()}, ok=not problems)
    print(json.dumps(rep, indent=1, default=str))
    if args.json:
        with open(args.json, "x") as f:          # "x": never overwrite
            json.dump(rep, f, indent=1, default=str)


if __name__ == "__main__":
    main()
