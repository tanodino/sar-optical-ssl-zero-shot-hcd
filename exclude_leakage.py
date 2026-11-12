"""
Leakage exclusion + pre-training subset selection for SSL4EO-S12 v1.1, from **metadata only**. Nothing is downloaded or deleted; the output is a list of kept samples.

Rules (the test areas of this step were those of the flood test set considered during the study, GeoFlood-275;
the EMS wildfire benchmark is separated from the corpus by masking its pixels within 10 km of a kept location)
1. Temporal separation: every S1/S2 acquisition date in the corpus must be earlier than the earliest test event
   (checked here on the metadata, and again on the downloaded files later). The script stops if it is violated.
2. Spatial exclusion: a location is removed if its footprint lies within `--buffer_km` (10 km) of the
   **bounding rectangle** of any test AOI. AOI outlines come from the Copernicus EMS public API (saved JSON answers,
   see leakage_overlap_check.py); the rectangle is taken in a local azimuthal-equidistant projection centred on the
   AOI (metres), so the buffer is a true distance.
3. Valid season: S2 cloud_cover <= `--cloud_max` (0.10) and |S1 date - S2 date| <= `--max_gap_days` (30).
4. Subset: shards are visited in a seeded random order (np.random.default_rng(seed).permutation of
   the sorted shard names); in each shard, every non-excluded location with >= `--seasons` valid seasons is kept,
   with `--seasons` (2) of its valid seasons drawn at random (seeded); whole shards are added until at least
   `--target_locations` (50,000) locations are kept (whole shards: no data downloaded for nothing).

Outputs (in --out_dir, which must not exist yet):
    kept_samples.csv   tar, zarr, sample_id, center_lon, center_lat, seasons (e.g. "0;3")
    shards.txt         shard names to download (same names in train/S1GRD and train/S2L1C)
    excluded.csv       sample_id, tar, aoi (source_id), distance_km for every spatially excluded location
    summary.json       configuration and all counts

Usage:
    python exclude_leakage.py \
        --meta data/pretrain/ssl4eo_s12_v1.1/train_metadata.parquet \
        --test_index data/test/geoflood275/raw/metadata/sample_index_test.csv \
        --ems_dir reports/emsr_activations --out_dir subset_v1
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import box, wkb, wkt
from shapely.ops import transform as shp_transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leakage_overlap_check import fetch_activation   # noqa: E402  (reuses the saved EMS answers)


def test_aois(test_index: str, ems_dir: str):
    """Return {source_id: (aoi polygon lon/lat, event_time str)} for every test event."""
    idx = pd.read_csv(test_index)
    out = {}
    for sid in sorted(idx["source_id"].unique()):
        code, num = re.match(r"(EMSR\d+)_AOI(\d+)_", sid).groups()
        act = fetch_activation(code, ems_dir)["results"]
        assert len(act) == 1, code
        aois = [a for a in act[0]["aois"] if a["number"] == int(num)]
        assert len(aois) == 1, (sid, len(aois))
        out[sid] = (wkt.loads(aois[0]["extent"]), act[0]["eventTime"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--ems_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--buffer_km", type=float, default=10.0)
    ap.add_argument("--cloud_max", type=float, default=0.10)
    ap.add_argument("--max_gap_days", type=float, default=30.0)
    ap.add_argument("--seasons", type=int, default=2)
    ap.add_argument("--target_locations", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(vars(args))
    os.makedirs(args.out_dir, exist_ok=False)          # never write into an existing folder

    meta = pd.read_parquet(args.meta)
    n_all = len(meta)
    aois = test_aois(args.test_index, args.ems_dir)

    # ---- 1. temporal separation (on metadata) ------------------------------------------------------------
    earliest_event = min(pd.Timestamp(t) for _, t in aois.values())
    tcols = [c for c in meta.columns if c.startswith(("S1_time_", "S2_time_"))]
    latest_acq = meta[tcols].max().max()
    assert latest_acq < earliest_event, f"temporal leakage: {latest_acq} >= {earliest_event}"
    print(f"temporal check OK: latest acquisition {latest_acq} < earliest test event {earliest_event}")

    # ---- 2. spatial exclusion ----------------------------------------------------------------------------
    foot = np.array([wkb.loads(g) for g in meta["geometry"]], dtype=object)
    tree = shapely.STRtree(foot)
    excl_rows, per_aoi = [], {}
    for sid, (aoi_ll, _) in aois.items():
        c = aoi_ll.centroid
        to_m = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +units=m",
                                    always_xy=True).transform
        rect_m = box(*shp_transform(to_m, aoi_ll).bounds)            # bounding rectangle in metres
        deg = (args.buffer_km + 50) / 111.0 / max(np.cos(np.radians(c.y)), 0.1)
        cand = tree.query(box(*aoi_ll.bounds).buffer(deg))           # generous lon/lat pre-filter
        n = 0
        for i in cand:
            d = rect_m.distance(shp_transform(to_m, foot[i])) / 1000.0
            if d <= args.buffer_km:
                excl_rows.append((meta["sample_id"].iloc[i], meta["tar"].iloc[i], sid, round(d, 3)))
                n += 1
        per_aoi[sid] = n
    excluded = pd.DataFrame(excl_rows, columns=["sample_id", "tar", "aoi", "distance_km"])
    excluded_ids = set(excluded["sample_id"])
    print(f"spatial exclusion: {len(excluded_ids)} locations ({len(excluded)} location-AOI pairs)")

    # ---- 3. valid seasons ---------------------------------------------------------------------------------
    valid = np.zeros((n_all, 4), dtype=bool)
    for s in range(4):
        gap = (meta[f"S1_time_{s}"] - meta[f"S2_time_{s}"]).abs().dt.total_seconds() / 86400.0
        valid[:, s] = (meta[f"cloud_cover_{s}"] <= args.cloud_max).to_numpy() & (gap <= args.max_gap_days).to_numpy()
    n_valid = valid.sum(1)
    not_excl = ~meta["sample_id"].isin(excluded_ids).to_numpy()
    eligible = not_excl & (n_valid >= args.seasons)
    print(f"eligible locations (not excluded, >= {args.seasons} valid seasons): {eligible.sum()} / {n_all}")

    # ---- 4. subset by whole shards in a seeded random order ----------------------------------------------
    shards = sorted(meta["tar"].unique())
    order = [shards[i] for i in np.random.default_rng(args.seed).permutation(len(shards))]
    season_rng = np.random.default_rng(args.seed + 1)
    by_shard = {s: np.flatnonzero((meta["tar"] == s).to_numpy()) for s in shards}
    kept, chosen_shards = [], []
    for s in order:
        if len(kept) >= args.target_locations:
            break
        chosen_shards.append(s)
        for i in by_shard[s]:                                         # metadata order inside the shard
            if not eligible[i]:
                continue
            seas = np.sort(season_rng.choice(np.flatnonzero(valid[i]), size=args.seasons, replace=False))
            kept.append((meta["tar"].iloc[i], meta["zarr"].iloc[i], meta["sample_id"].iloc[i],
                         meta["center_lon"].iloc[i], meta["center_lat"].iloc[i], ";".join(map(str, seas))))
    kept = pd.DataFrame(kept, columns=["tar", "zarr", "sample_id", "center_lon", "center_lat", "seasons"])
    in_chosen = meta["tar"].isin(chosen_shards).to_numpy()

    summary = dict(
        config=vars(args), n_locations_all=n_all,
        earliest_test_event=str(earliest_event), latest_acquisition=str(latest_acq),
        n_excluded_spatial=len(excluded_ids), excluded_per_aoi=per_aoi,
        n_valid_seasons_histogram={str(k): int((n_valid == k).sum()) for k in range(5)},
        n_eligible_all=int(eligible.sum()),
        n_shards=len(chosen_shards),
        in_chosen_shards=dict(locations=int(in_chosen.sum()),
                              excluded_spatial=int((in_chosen & ~not_excl).sum()),
                              too_few_valid_seasons=int((in_chosen & not_excl & (n_valid < args.seasons)).sum()),
                              kept=int(len(kept))),
        n_kept_locations=int(len(kept)), n_pairs=int(len(kept) * args.seasons),
        season_counts={str(k): int(v) for k, v in
                       pd.Series(";".join(kept["seasons"]).split(";")).value_counts().sort_index().items()})
    with open(os.path.join(args.out_dir, "kept_samples.csv"), "x") as f:
        kept.to_csv(f, index=False)
    with open(os.path.join(args.out_dir, "shards.txt"), "x") as f:
        f.write("\n".join(chosen_shards) + "\n")
    with open(os.path.join(args.out_dir, "excluded.csv"), "x") as f:
        excluded.to_csv(f, index=False)
    with open(os.path.join(args.out_dir, "summary.json"), "x") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
