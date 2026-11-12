"""
Pre-processing of the EMS wildfires 2025 benchmark into the scene format of the pipeline (reference, masks, normalised
arrays). (The released benchmark is the output of this step; prepare_scenes.py rebuilds the arrays from it.)

Inputs per AOI: data/test/ems_wildfire_2025_v1/<AOI>/{s2.tif, s2_cloudprob.tif, s1.tif, meta.json} (download_ems_wildfire.py,
native UTM 10 m grid of the S2 image), labels data/test/ems_wildfire_2025_d/labels/<AOI>_{observedEventA,notAnalysedA}.json
(EMS GeoJSON, CRS84), AOI extent from data/test/ems_wildfire_2025_d/selected.csv, pre-training footprints
reports/pretrain_kept_footprints.csv.

Label : 1 = inside an EMS burnt-area polygon (observedEventA), 0 = not; label = 0 wherever valid = 0 (as GeoFlood).
Valid : inside the AOI polygon AND outside EMS notAnalysedA AND S2 valid (no band == 0) AND S1 valid (no band == -9999)
        AND pre-event S2 cloud probability <= --cloud_prob AND farther than --buffer_km from every kept pre-training
        footprint (leakage). Every invalid pixel gets a bit code in reason.npy:
        1 outside AOI, 2 not analysed, 4 S2 no-data, 8 S1 no-data, 16 cloud, 32 leakage zone.
Images: identical transforms to pre-training / GeoFlood (norm_stats.json): band order B02, B03, B04, B08 and vv, vh;
        clip to [lo, hi], standardise with mean / std; invalid pixels set to 0 (= band mean). S2 DN are from
        COPERNICUS/S2_HARMONIZED, which already removes the +1000 offset of baseline >= 04.00: NO offset subtracted
        (GeoFlood: 1000 subtracted because it comes from the raw COPERNICUS/S2 collection).
No resampling: the scenes are on a metric 10 m UTM grid (no east-west resampling).

Outputs (new folder): <AOI>/{opt.npy (4,H,W) f16, sar.npy (2,H,W) f16, label.npy u8, valid.npy u8, reason.npy u8,
meta.json}, config.json, summary.json.

Usage:
    python preprocess_ems_wildfire.py --raw data/test/ems_wildfire_2025_v1 \
        --sel data/test/ems_wildfire_2025_d --footprints reports/pretrain_kept_footprints.csv \
        --norm_stats data/pretrain/ssl4eo_s12_v1.1_tiles64_v1/norm_stats.json --out_dir data/test/ems_wildfire_2025_v1/scenes_v1
"""
import argparse
import csv
import json
import os

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from shapely import wkt
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shp_transform, unary_union
from shapely.validation import make_valid

OPT_BANDS, SAR_BANDS = ["B02", "B03", "B04", "B08"], ["vv", "vh"]


def burn(geoms, shape_hw, transform):
    """Rasterise polygons (pixel centre inside) -> bool (H, W)."""
    geoms = [mapping(g) for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape_hw, bool)
    return rasterize(((g, 1) for g in geoms), out_shape=shape_hw, transform=transform, fill=0, dtype="uint8").astype(bool)


def load_geojson(path, to_utm):
    if not os.path.exists(path):
        return []
    feats = json.load(open(path))["features"]
    return [make_valid(shp_transform(to_utm, shape(f["geometry"]))) for f in feats if f.get("geometry")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sel", required=True)
    ap.add_argument("--footprints", required=True)
    ap.add_argument("--norm_stats", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cloud_prob", type=int, default=40)
    ap.add_argument("--buffer_km", type=float, default=10.0)
    args = ap.parse_args()
    print(vars(args), flush=True)
    os.makedirs(args.out_dir, exist_ok=False)
    with open(os.path.join(args.out_dir, "config.json"), "x") as f:
        json.dump(vars(args), f, indent=1)
    ns = json.load(open(args.norm_stats))
    lim = lambda st, bands, k: np.array([st[b][k] for b in bands], np.float32)[:, None, None]
    lo2, hi2, mu2, sd2 = (lim(ns["s2"], OPT_BANDS, k) for k in ("lo", "hi", "mean", "std"))
    lo1, hi1, mu1, sd1 = (lim(ns["s1"], SAR_BANDS, k) for k in ("lo", "hi", "mean", "std"))
    feet = [(float(r["center_lon"]), float(r["center_lat"]), r["footprint_wkt"])
            for r in csv.DictReader(open(args.footprints))]
    rows = list(csv.DictReader(open(os.path.join(args.sel, "selected.csv"))))
    summary = {}
    for r in rows:
        tag = f"{r['code']}_AOI{int(r['aoi']):02d}"
        src_dir = os.path.join(args.raw, tag)
        with rasterio.open(os.path.join(src_dir, "s2.tif")) as s:
            s2 = s.read().astype(np.float32); crs = s.crs.to_string(); tr = s.transform; H, W = s.height, s.width
        with rasterio.open(os.path.join(src_dir, "s2_cloudprob.tif")) as s:
            cp = s.read(1)
        with rasterio.open(os.path.join(src_dir, "s1.tif")) as s:
            s1 = s.read().astype(np.float32)
        to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform

        aoi = make_valid(shp_transform(to_utm, wkt.loads(r["aoi_extent_wkt"])))
        in_aoi = burn([aoi], (H, W), tr)
        burnt = burn(load_geojson(os.path.join(args.sel, "labels", f"{tag}_observedEventA.json"), to_utm), (H, W), tr)
        not_an = burn(load_geojson(os.path.join(args.sel, "labels", f"{tag}_notAnalysedA.json"), to_utm), (H, W), tr)
        s2_nod = (s2 == 0).any(0)
        s1_nod = (s1 == -9999).any(0)
        cloud = (cp > args.cloud_prob) & (cp != 255)
        x0, y0, x1, y1 = shp_transform(Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform,
                                       box(tr.c, tr.f - 10 * H, tr.c + 10 * W, tr.f)).bounds
        near = [shp_transform(to_utm, wkt.loads(w)) for lon, lat, w in feet
                if x0 - 0.5 < lon < x1 + 0.5 and y0 - 0.5 < lat < y1 + 0.5]
        leak = burn([unary_union([g.buffer(args.buffer_km * 1000) for g in near])] if near else [], (H, W), tr)

        reason = (np.uint8(1) * ~in_aoi + np.uint8(2) * not_an + np.uint8(4) * s2_nod + np.uint8(8) * s1_nod
                  + np.uint8(16) * cloud + np.uint8(32) * leak).astype(np.uint8)
        valid = reason == 0
        label = burnt & valid
        clip_cnt = {}
        for name, x, lo, hi in (("s2", s2, lo2, hi2), ("s1", s1, lo1, hi1)):
            clip_cnt[f"{name}_below_lo"] = float(((x < lo) & valid).sum() / max(valid.sum() * x.shape[0], 1))
            clip_cnt[f"{name}_above_hi"] = float(((x > hi) & valid).sum() / max(valid.sum() * x.shape[0], 1))
        opt = np.where(valid, (np.clip(s2, lo2, hi2) - mu2) / sd2, 0.0).astype(np.float16)
        sar = np.where(valid, (np.clip(s1, lo1, hi1) - mu1) / sd1, 0.0).astype(np.float16)

        d = os.path.join(args.out_dir, tag)
        os.makedirs(d)
        for name, arr in (("opt", opt), ("sar", sar), ("label", label.astype(np.uint8)), ("valid", valid.astype(np.uint8)),
                          ("reason", reason)):
            np.save(os.path.join(d, f"{name}.npy"), arr)
        n_aoi = int(in_aoi.sum())
        meta = dict(event=tag, H=H, W=W, crs=crs, transform=list(tr)[:6], n_aoi=n_aoi, n_valid=int(valid.sum()),
                    frac_valid_of_aoi=float(valid.sum() / max(n_aoi, 1)), n_burnt=int(label.sum()),
                    burnt_frac_of_valid=float(label.sum() / max(valid.sum(), 1)),
                    burnt_px_dropped_invalid=int((burnt & ~valid).sum()),
                    invalid_in_aoi={k: float(((reason & b) > 0)[in_aoi].mean()) for k, b in
                                    (("not_analysed", 2), ("s2_nodata", 4), ("s1_nodata", 8), ("cloud", 16), ("leakage", 32))},
                    clipping=clip_cnt, opt_bands=OPT_BANDS, sar_bands=SAR_BANDS, s2_offset_subtracted=0.0,
                    s2_collection="COPERNICUS/S2_HARMONIZED", norm_stats=os.path.abspath(args.norm_stats),
                    normalised_mean_valid={b: float(opt[i][valid].astype(np.float32).mean()) for i, b in enumerate(OPT_BANDS)}
                    | {b: float(sar[i][valid].astype(np.float32).mean()) for i, b in enumerate(SAR_BANDS)},
                    n_pretrain_footprints_near=len(near), source=json.load(open(os.path.join(src_dir, "meta.json"))))
        with open(os.path.join(d, "meta.json"), "x") as f:
            json.dump(meta, f, indent=1)
        summary[tag] = {k: meta[k] for k in ("H", "W", "n_valid", "frac_valid_of_aoi", "burnt_frac_of_valid",
                                             "invalid_in_aoi", "clipping", "normalised_mean_valid")}
        iv = meta["invalid_in_aoi"]
        print(f"{tag:15s} {H:5d}x{W:<5d} valid {meta['frac_valid_of_aoi']:.3f} of AOI | burnt {100 * meta['burnt_frac_of_valid']:5.1f} % "
              f"| invalid in AOI: notan {iv['not_analysed']:.3f} s2 {iv['s2_nodata']:.3f} s1 {iv['s1_nodata']:.3f} "
              f"cloud {iv['cloud']:.3f} leak {iv['leakage']:.3f}", flush=True)
    tot_v = sum(v["n_valid"] for v in summary.values())
    tot_b = sum(v["n_valid"] * v["burnt_frac_of_valid"] for v in summary.values())
    out = dict(n_scenes=len(summary), n_valid=tot_v, burnt_frac_pooled=tot_b / max(tot_v, 1), scenes=summary)
    with open(os.path.join(args.out_dir, "summary.json"), "x") as f:
        json.dump(out, f, indent=1)
    print(f"pooled: {tot_v} valid px, burnt {100 * out['burnt_frac_pooled']:.2f} %")


if __name__ == "__main__":
    main()
