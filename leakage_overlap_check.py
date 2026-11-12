"""
Spatial-overlap measurement between the pre-training corpus and the test areas (diagnostic only).

Context: leakage is controlled by **temporal separation**
(SSL4EO-S12 v1.1: Dec 2019 - Oct 2021; GeoFlood-275 test events: 2026). This script does NOT filter anything:
it only *measures* how many pre-training locations fall on, or near, the test areas, so that the residual
"same place, different years" overlap can be reported as a number.

Why the outlines come from Copernicus EMS: the GeoFlood-275 test GeoTIFFs carry no georeferencing
(no CRS, no GCPs, identity transform). Every test sample key starts with the
public Copernicus EMS Rapid Mapping activation code and AOI number (e.g. EMSR866_AOI06_...), and the EMS public
API returns each AOI outline as a WKT polygon in lon/lat (EPSG:4326).

Steps
1. Read the GeoFlood test sample index -> the set of (activation, AOI number) pairs and tiles per event.
2. For each activation, query https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/
   ?code=<EMSR...> and save the raw JSON answer (new file, never overwritten) for reproducibility.
3. For each test AOI: project the AOI and the SSL4EO footprints to a local azimuthal-equidistant projection
   centred on the AOI (distances in metres), then count the SSL4EO locations whose footprint
   (a) intersects the AOI, (b) lies within each buffer distance; and the fraction of the AOI area covered by
   intersecting footprints.

Usage:
    python leakage_overlap_check.py \
        --index data/test/geoflood275/raw/metadata/sample_index_test.csv \
        --ssl4eo_meta data/pretrain/ssl4eo_s12_v1.1/train_metadata.parquet \
        --raw_dir reports/emsr_activations --out reports/leakage_overlap.json
"""
import argparse
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import wkb, wkt
from shapely.ops import transform as shp_transform, unary_union

API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code={}"


def fetch_activation(code: str, raw_dir: str) -> dict:
    """Return the EMS JSON for one activation; reuse the saved copy if present (never overwrite it)."""
    path = os.path.join(raw_dir, f"{code}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    req = urllib.request.Request(API.format(code), headers={"User-Agent": "cd-study-overlap-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    with open(path, "x") as f:                       # "x": never overwrite
        json.dump(data, f, indent=1)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="GeoFlood sample_index_test.csv")
    ap.add_argument("--ssl4eo_meta", required=True, help="SSL4EO-S12 v1.1 train_metadata.parquet")
    ap.add_argument("--raw_dir", required=True, help="folder for the raw EMS JSON answers")
    ap.add_argument("--out", required=True, help="output JSON report (must not exist)")
    ap.add_argument("--buffers_km", default="1,5,10", help="buffer distances in km")
    args = ap.parse_args()
    print(vars(args))
    buffers = [float(b) for b in args.buffers_km.split(",")]
    os.makedirs(args.raw_dir, exist_ok=True)

    # 1. test events: source_id like "EMSR866_AOI06_DEL_PRODUCT" -> ("EMSR866", 6)
    idx = pd.read_csv(args.index)
    events = {}
    for sid, n in idx.groupby("source_id").size().items():
        m = re.match(r"(EMSR\d+)_AOI(\d+)_", sid)
        assert m, sid
        events[sid] = dict(code=m.group(1), aoi=int(m.group(2)), n_tiles=int(n))

    # 2. AOI outlines from EMS
    acts = {c: fetch_activation(c, args.raw_dir) for c in sorted({e["code"] for e in events.values()})}

    # SSL4EO footprints (WKB polygons in lon/lat) and their dates (for the report)
    meta = pd.read_parquet(args.ssl4eo_meta, columns=["sample_id", "geometry", "center_lon", "center_lat"])
    foot = np.array([wkb.loads(g) for g in meta["geometry"]], dtype=object)
    tree = shapely.STRtree(foot)

    report = {"buffers_km": buffers, "n_ssl4eo_locations": int(len(meta)), "events": {}}
    for sid, e in sorted(events.items()):
        res = acts[e["code"]]["results"]
        assert len(res) == 1, (e["code"], len(res))
        act = res[0]
        aois = [a for a in act["aois"] if a["number"] == e["aoi"]]
        if len(aois) != 1:
            report["events"][sid] = dict(e, error=f"{len(aois)} AOIs with number {e['aoi']}")
            print(sid, "ERROR", report["events"][sid]["error"])
            continue
        aoi_ll = wkt.loads(aois[0]["extent"])
        # local azimuthal-equidistant projection centred on the AOI: distances in metres
        c = aoi_ll.centroid
        to_m = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +units=m",
                                    always_xy=True).transform
        aoi_m = shp_transform(to_m, aoi_ll)
        # candidates: footprints within ~(max buffer + margin) in degrees, then exact test in metres
        deg = (max(buffers) + 20) / 111.0 / max(np.cos(np.radians(c.y)), 0.1)
        cand = tree.query(aoi_ll.buffer(deg))
        cand_m = [shp_transform(to_m, foot[i]) for i in cand]
        dist_km = np.array([aoi_m.distance(g) / 1000 for g in cand_m])
        inter = [g for g, d in zip(cand_m, dist_km) if d == 0]
        cover = unary_union(inter).intersection(aoi_m).area / aoi_m.area if inter else 0.0
        out = dict(e, activation_name=act["name"], event_time=act["eventTime"], aoi_name=aois[0]["name"],
                   aoi_area_km2=round(aoi_m.area / 1e6, 2),
                   n_intersecting=int((dist_km == 0).sum()),
                   frac_aoi_area_covered=round(float(cover), 5),
                   intersecting_sample_ids=[meta["sample_id"].iloc[i] for i, d in zip(cand, dist_km) if d == 0])
        for b in buffers:
            out[f"n_within_{b:g}km"] = int((dist_km <= b).sum())
        report["events"][sid] = out
        print(f"{sid:28s} {act['eventTime'][:10]} {aois[0]['name'][:22]:22s} area {out['aoi_area_km2']:9.1f} km2 "
              f"tiles {e['n_tiles']:4d} | intersect {out['n_intersecting']:3d} (cover {cover:.4f}) "
              + " ".join(f"<={b:g}km {out[f'n_within_{b:g}km']:3d}" for b in buffers))

    ev = [v for v in report["events"].values() if "error" not in v]
    report["summary"] = dict(
        n_events=len(events), n_events_ok=len(ev),
        n_events_with_intersection=sum(v["n_intersecting"] > 0 for v in ev),
        n_intersecting_total=sum(v["n_intersecting"] for v in ev),
        **{f"n_within_{b:g}km_total": sum(v[f"n_within_{b:g}km"] for v in ev) for b in buffers},
        earliest_test_event_time=min(v["event_time"] for v in ev))
    print(json.dumps(report["summary"], indent=1))
    with open(args.out, "x") as f:                   # "x": never overwrite
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
