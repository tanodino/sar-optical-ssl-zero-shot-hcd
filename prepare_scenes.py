"""
Convert the released EMS Wildfires 2025 benchmark (GeoTIFFs) into the per-scene arrays read by the pipeline
(extract_features.py, evaluate_benchmark.py, caa_run.py, hcd_baselines_run.py, cdrl_run.py):

  <out>/<AOI>/opt.npy    (4, H, W) float16, Sentinel-2 B2, B3, B4, B8: clipped to [lo, hi] and standardised (norm_stats.json)
  <out>/<AOI>/sar.npy    (2, H, W) float16, Sentinel-1 VV, VH (dB): clipped and standardised
  <out>/<AOI>/label.npy  (H, W) uint8, burnt = 1
  <out>/<AOI>/valid.npy  (H, W) uint8, evaluated = 1
  <out>/<AOI>/reason.npy (H, W) uint8, bit flags (see the benchmark README)
  <out>/<AOI>/meta.json  copy of the benchmark metadata.json

Invalid pixels are set to 0 in opt / sar (after standardisation), as in the paper. The statistics come from the pre-training
corpus (norm_stats.json shipped with the benchmark). The result is identical to the arrays used in the paper.

Usage: python prepare_scenes.py --benchmark <path>/ems_wildfire_2025_v1 --out_dir data/scenes
"""
import argparse
import json
import os
import shutil

import numpy as np
import rasterio

OPT_BANDS, SAR_BANDS = ["B02", "B03", "B04", "B08"], ["vv", "vh"]


def standardise(x, stats, bands):
    """x (C, H, W) float32 -> clipped to [lo, hi] and standardised per band."""
    out = np.empty_like(x, dtype=np.float32)
    for i, b in enumerate(bands):
        st = stats[b]
        out[i] = (np.clip(x[i], st["lo"], st["hi"]) - st["mean"]) / st["std"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="folder of the released benchmark (contains aois.csv)")
    ap.add_argument("--out_dir", required=True, help="new folder (refused if it exists)")
    args = ap.parse_args()
    assert not os.path.exists(args.out_dir), args.out_dir
    ns = json.load(open(os.path.join(args.benchmark, "norm_stats.json")))
    aois = sorted(d for d in os.listdir(args.benchmark) if os.path.isfile(os.path.join(args.benchmark, d, "metadata.json")))
    for a in aois:
        src, dst = os.path.join(args.benchmark, a), os.path.join(args.out_dir, a)
        os.makedirs(dst)
        read = lambda f: rasterio.open(os.path.join(src, f)).read()   # noqa: E731
        s2, s1 = read("s2_l1c.tif").astype(np.float32), read("s1_grd_db.tif").astype(np.float32)
        label, valid, reason = (read(f)[0].astype(np.uint8) for f in ("label.tif", "valid.tif", "reason.tif"))
        v = valid.astype(bool)
        assert np.array_equal(v, reason == 0), a
        opt = np.where(v, standardise(s2, ns["s2"], OPT_BANDS), 0.0).astype(np.float16)
        sar = np.where(v, standardise(s1, ns["s1"], SAR_BANDS), 0.0).astype(np.float16)
        for name, arr in (("opt", opt), ("sar", sar), ("label", label), ("valid", valid), ("reason", reason)):
            np.save(os.path.join(dst, f"{name}.npy"), arr)
        shutil.copy(os.path.join(src, "metadata.json"), os.path.join(dst, "meta.json"))
        print(a, opt.shape, f"valid {v.mean():.3f}", f"burnt {label[v].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
