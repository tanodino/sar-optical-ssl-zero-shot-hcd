"""
Pre-processing of the SSL4EO-S12 v1.1 pre-training subset into the training format.

Input : the downloaded shards (train/S1GRD, train/S2L1C; WebDataset tars of Zarr-zip samples, read in place)
        and the subset list `kept_samples.csv` (exclude_leakage.py): one row per location with its 2 seasons.
Output (in --out_dir, which must not exist):
    SAR_data_normalized.npy  float16 (N, 2, 64, 64)   S1 GRD VV, VH (dB)
    MS_data_normalized.npy   float16 (N, 4, 64, 64)   S2 L1C B2, B3, B4, B8 (DN = reflectance x 10000)
    tile_index.csv           row -> tar, zarr, sample_id, season, tile_r, tile_c   (row i of both arrays = same tile)
    norm_stats.json          per band: clip limits (1st/99th pct), mean/std of clipped values, sample sizes;
                             + optical zero point and solarize values for the augmentations
    config.json              arguments
Row order: kept_samples.csv order x the location's 2 seasons (ascending) x 16 tiles (row-major on the 4x4 grid).

Steps
1. Statistics on a seeded random sample of `--stats_pairs` (location, season) pairs of the selection:
   from each sampled pair, `--stats_pixels` random pixels of the central 256x256 crop per band ->
   per band lo / hi = percentiles (S2: 1st / 99th; S1: 0.1st / 99th); mean/std of the values
   clipped to [lo, hi].
2. Tiling: crop rows/cols 4..259 (4-pixel border dropped), 4x4 grid of 64x64 tiles.
3. Normalisation: x_n = (clip(x, lo, hi) - mean) / std, stored as float16.
Every pair is checked: same sample id and identical x/y pixel coordinates in S1 and S2, finite values.

Augmentation constants, per S2 band, in normalised units:
    zero_point          = (0 - mean) / std            (physical reflectance 0; brightness/contrast pivot)
    solarize_threshold  = ((lo + hi) / 2 - mean) / std (middle of the clipping range, = DeCUR's 0.5 on [0,1])
    solarize_max        = lo_n + hi_n                  (so x' = solarize_max - x reflects about the middle)

Usage (first run on 1,000 pairs, then the full run):
    python preprocess_corpus.py --root data/pretrain/ssl4eo_s12_v1.1 --subset subset_v1 \
        --out_dir data/pretrain/ssl4eo_s12_v1.1_tiles64_check1000 --max_pairs 1000 --workers 8
"""
import argparse
import json
import os
import tarfile
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import zarr

S1_BANDS = ["vv", "vh"]
S2_BANDS = ["B02", "B03", "B04", "B08"]
CROP0, CROP1, TILE = 4, 260, 64          # central 256x256 crop -> 4x4 tiles of 64x64
GRID = (CROP1 - CROP0) // TILE           # 4
TILES_PER_IMAGE = GRID * GRID            # 16


def open_zarr(tf: tarfile.TarFile, member: tarfile.TarInfo, keys) -> zarr.Group:
    """Open selected arrays of a Zarr zip stored as a tar member (the tar member is seekable: no full read)."""
    z = zipfile.ZipFile(tf.extractfile(member))
    names = [n for n in z.namelist() if n.split("/")[0] in keys or n.startswith(".z")]
    return zarr.open_group(store={n: z.read(n) for n in names}, mode="r")


def read_pair_images(tf1, tf2, mem1, mem2, zarr_name):
    """Return (s1 (4, 2, 264, 264) float32, s2 (4, 4, 264, 264) float32) for one location, with consistency checks."""
    keys = {"band", "bands", "sample", "x", "y"}
    g1, g2 = open_zarr(tf1, mem1[zarr_name], keys), open_zarr(tf2, mem2[zarr_name], keys)
    assert str(g1["sample"][()]) == str(g2["sample"][()]), zarr_name
    assert np.array_equal(g1["x"][:], g2["x"][:]) and np.array_equal(g1["y"][:], g2["y"][:]), \
        f"S1/S2 grids differ: {zarr_name}"
    b1, b2 = [str(b) for b in g1["band"][:]], [str(b) for b in g2["band"][:]]
    s1 = g1["bands"].get_orthogonal_selection((slice(None), [b1.index(b) for b in S1_BANDS])).astype(np.float32)
    s2 = g2["bands"].get_orthogonal_selection((slice(None), [b2.index(b) for b in S2_BANDS])).astype(np.float32)
    assert np.isfinite(s1).all() and np.isfinite(s2).all(), f"non-finite values: {zarr_name}"
    return s1, s2


def members(path):
    tf = tarfile.open(path)
    return tf, {m.name: m for m in tf.getmembers() if m.isfile()}


# ---------------------------------------------------------------- step 1: statistics (worker)
def stats_worker(job):
    root, tar, items, n_pix, seed = job        # items: list of (zarr, season, pair_id)
    tf1, mem1 = members(os.path.join(root, "train", "S1GRD", tar))
    tf2, mem2 = members(os.path.join(root, "train", "S2L1C", tar))
    out1, out2 = [], []
    for zname, season, pid in items:
        s1, s2 = read_pair_images(tf1, tf2, mem1, mem2, zname)
        rng = np.random.default_rng([seed, pid])                 # per-pair seed: independent of worker order
        idx = rng.choice((CROP1 - CROP0) ** 2, size=n_pix, replace=False)
        c1 = s1[season, :, CROP0:CROP1, CROP0:CROP1].reshape(len(S1_BANDS), -1)[:, idx]
        c2 = s2[season, :, CROP0:CROP1, CROP0:CROP1].reshape(len(S2_BANDS), -1)[:, idx]
        out1.append(c1)
        out2.append(c2)
    tf1.close()
    tf2.close()
    return np.concatenate(out1, 1), np.concatenate(out2, 1)


def band_stats(values: np.ndarray, names, pct_lo: float, pct_hi: float) -> dict:
    """values (C, M) -> per band clip limits and mean/std of the clipped values."""
    out = {}
    for b, name in enumerate(names):
        v = values[b].astype(np.float64)
        lo, hi = np.percentile(v, [pct_lo, pct_hi])
        vc = np.clip(v, lo, hi)
        out[name] = dict(lo=float(lo), hi=float(hi), mean=float(vc.mean()), std=float(vc.std()),
                         raw_min=float(v.min()), raw_max=float(v.max()), n=int(v.size))
    return out


# ---------------------------------------------------------------- step 2-3: tiling + normalisation (worker)
def to_tiles(img: np.ndarray) -> np.ndarray:
    """(C, 264, 264) -> (16, C, 64, 64), tiles in row-major order of the 4x4 grid."""
    c = img[:, CROP0:CROP1, CROP0:CROP1]
    C = c.shape[0]
    return c.reshape(C, GRID, TILE, GRID, TILE).transpose(1, 3, 0, 2, 4).reshape(TILES_PER_IMAGE, C, TILE, TILE)


def write_worker(job):
    root, tar, items, out_dir, n_rows, st1, st2 = job          # items: list of (zarr, seasons, first_row)
    sar = np.load(os.path.join(out_dir, "SAR_data_normalized.npy"), mmap_mode="r+")
    ms = np.load(os.path.join(out_dir, "MS_data_normalized.npy"), mmap_mode="r+")
    assert sar.shape[0] == n_rows and ms.shape[0] == n_rows
    lo1, hi1, mu1, sd1 = (np.array([st1[b][k] for b in S1_BANDS], np.float32)[:, None, None]
                          for k in ("lo", "hi", "mean", "std"))
    lo2, hi2, mu2, sd2 = (np.array([st2[b][k] for b in S2_BANDS], np.float32)[:, None, None]
                          for k in ("lo", "hi", "mean", "std"))
    tf1, mem1 = members(os.path.join(root, "train", "S1GRD", tar))
    tf2, mem2 = members(os.path.join(root, "train", "S2L1C", tar))
    for zname, seasons, row in items:
        s1, s2 = read_pair_images(tf1, tf2, mem1, mem2, zname)
        for season in seasons:
            a = (np.clip(s1[season], lo1, hi1) - mu1) / sd1          # (2, 264, 264)
            b = (np.clip(s2[season], lo2, hi2) - mu2) / sd2          # (4, 264, 264)
            sar[row:row + TILES_PER_IMAGE] = to_tiles(a).astype(np.float16)
            ms[row:row + TILES_PER_IMAGE] = to_tiles(b).astype(np.float16)
            row += TILES_PER_IMAGE
    sar.flush()
    ms.flush()
    tf1.close()
    tf2.close()
    return tar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="SSL4EO-S12 v1.1 download folder (contains train/)")
    ap.add_argument("--subset", required=True, help="folder with kept_samples.csv (exclude_leakage.py)")
    ap.add_argument("--out_dir", required=True, help="new output folder (must not exist)")
    ap.add_argument("--max_pairs", type=int, default=None, help="use only the first N (location, season) pairs")
    ap.add_argument("--stats_pairs", type=int, default=5000, help="pairs sampled for the statistics")
    ap.add_argument("--stats_pixels", type=int, default=4096, help="pixels per pair and band for the statistics")
    ap.add_argument("--pct_lo", type=float, default=1.0, help="S2 lower percentile")
    ap.add_argument("--pct_hi", type=float, default=99.0, help="S2 upper percentile")
    ap.add_argument("--s1_pct_lo", type=float, default=0.1,
                    help="S1 lower percentile (0.1, keeps dark water surfaces)")
    ap.add_argument("--s1_pct_hi", type=float, default=99.0, help="S1 upper percentile")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(vars(args), flush=True)
    os.makedirs(args.out_dir, exist_ok=False)
    with open(os.path.join(args.out_dir, "config.json"), "x") as f:
        json.dump(vars(args), f, indent=1)
    t0 = time.time()

    # ---- pairs in output order ----
    kept = pd.read_csv(os.path.join(args.subset, "kept_samples.csv"), dtype={"sample_id": str})
    pairs = [(r.tar, r.zarr, r.sample_id, int(s)) for r in kept.itertuples() for s in r.seasons.split(";")]
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    n_pairs, n_rows = len(pairs), len(pairs) * TILES_PER_IMAGE
    print(f"{n_pairs} pairs -> {n_rows} tiles", flush=True)

    # ---- step 1: statistics on a seeded sample of pairs ----
    rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(n_pairs, size=min(args.stats_pairs, n_pairs), replace=False))
    jobs = {}
    for pid in sel:
        tar, zname, _, season = pairs[pid]
        jobs.setdefault(tar, []).append((zname, season, int(pid)))
    with Pool(args.workers) as pool:
        res = pool.map(stats_worker, [(args.root, t, it, args.stats_pixels, args.seed) for t, it in sorted(jobs.items())])
    v1 = np.concatenate([r[0] for r in res], 1)
    v2 = np.concatenate([r[1] for r in res], 1)
    st1 = band_stats(v1, S1_BANDS, args.s1_pct_lo, args.s1_pct_hi)
    st2 = band_stats(v2, S2_BANDS, args.pct_lo, args.pct_hi)
    aug = {}
    for b in S2_BANDS:
        s = st2[b]
        lo_n, hi_n = (s["lo"] - s["mean"]) / s["std"], (s["hi"] - s["mean"]) / s["std"]
        aug[b] = dict(zero_point=(0.0 - s["mean"]) / s["std"],
                      solarize_threshold=((s["lo"] + s["hi"]) / 2 - s["mean"]) / s["std"],
                      solarize_max=lo_n + hi_n, lo_n=lo_n, hi_n=hi_n)
    norm = dict(s1_bands=S1_BANDS, s2_bands=S2_BANDS, s1_units="dB", s2_units="L1C DN (reflectance x 10000)",
                percentiles=dict(s1=[args.s1_pct_lo, args.s1_pct_hi], s2=[args.pct_lo, args.pct_hi]), stats_pairs=int(len(sel)), stats_pixels=args.stats_pixels,
                seed=args.seed, s1=st1, s2=st2, optical_augmentation=aug,
                opt_zero_point=[aug[b]["zero_point"] for b in S2_BANDS])
    with open(os.path.join(args.out_dir, "norm_stats.json"), "x") as f:
        json.dump(norm, f, indent=1)
    print(json.dumps(norm, indent=1), flush=True)
    print(f"statistics done in {time.time() - t0:.0f} s", flush=True)

    # ---- step 2-3: allocate outputs, tile + normalise in parallel (each shard writes its own rows) ----
    for name, C in (("SAR_data_normalized.npy", len(S1_BANDS)), ("MS_data_normalized.npy", len(S2_BANDS))):
        path = os.path.join(args.out_dir, name)
        assert not os.path.exists(path)
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n_rows, C, TILE, TILE)).flush()
    index_rows, write_jobs, row = [], {}, 0
    loc = {}
    for tar, zname, sid, season in pairs:                       # group the 2 seasons of a location
        loc.setdefault((tar, zname, sid), []).append(season)
    for (tar, zname, sid), seasons in loc.items():
        write_jobs.setdefault(tar, []).append((zname, seasons, row))
        for season in seasons:
            for k in range(TILES_PER_IMAGE):
                index_rows.append((row, tar, zname, sid, season, k // GRID, k % GRID))
                row += 1
    assert row == n_rows
    with open(os.path.join(args.out_dir, "tile_index.csv"), "x") as f:
        pd.DataFrame(index_rows, columns=["row", "tar", "zarr", "sample_id", "season", "tile_r", "tile_c"]
                     ).to_csv(f, index=False)
    with Pool(args.workers) as pool:
        for k, tar in enumerate(pool.imap_unordered(write_worker, [
                (args.root, t, it, args.out_dir, n_rows, st1, st2) for t, it in sorted(write_jobs.items())])):
            print(f"  shard {tar} written ({k + 1}/{len(write_jobs)}), {time.time() - t0:.0f} s", flush=True)
    print(f"done: {n_rows} tiles in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
