"""
Selection of the EMS wildfires 2025 benchmark: Copernicus EMS Rapid Mapping WILDFIRE activations of 2025 (EMS wildfire
delineations, at most 2 AOIs per activation, 10 km leakage rule; labels only from EMS, imagery from Google Earth Engine
with the SSL4EO-S12 collections). Metadata + small label
files only: no imagery is downloaded here.

Steps
1. EMS API (public dashboard API): index of activations, then the record of every 2025 activation with
   category "Wildfire" and subCategory "Forest fire" (requests spaced by --sleep seconds). Raw JSON saved.
2. Per AOI, the LABEL PRODUCT = among its DEL / GRA products that have an observedEventA (burnt area) layer, the one
   whose source images are the most recent (latest monitoring); label_date = latest image acquisition time of that
   product; burnt_ha = its "Burnt area" statistic. Candidates: burnt_ha >= --min_burnt_ha.
3. Label files of the candidates (observedEventA and, if present, notAnalysedA GeoJSON, from the public EMS S3 bucket).
4. Leakage: area of the AOI and of the burnt polygons lying farther than --buffer_km from every
   kept pre-training footprint (reports/pretrain_kept_footprints.csv); those zones will be masked at evaluation.
5. Image availability in Google Earth Engine (project --gee_project), images must fully contain the AOI extent:
   pre  = COPERNICUS/S2_HARMONIZED (L1C) in [event - --pre_days, event - 1 day] whose fraction of cloudy pixels INSIDE
          the AOI (COPERNICUS/S2_CLOUD_PROBABILITY > --cloud_prob, 20 m) is <= --aoi_cloud_max (tile-level
          CLOUDY_PIXEL_PERCENTAGE <= --s2_cloud_max only as a speed pre-filter); the latest one is kept;
   post = COPERNICUS/S1_GRD, IW, VV+VH, in (label_date, label_date + --post_days]; the earliest one is kept.
6. Eligible = labels found + S2 pre + S1 post + burnt area outside the leakage buffer >= --min_burnt_ha.
   + S2 pre -> S1 post gap <= --max_gap_days (15 days).
   Selection = seeded random order of the eligible AOIs, at most --max_per_activation per activation, first --n_select.

Outputs (new folder --out_dir, must not exist): api/*.json, labels/<EMSR>_AOIxx_{observedEventA,notAnalysedA}.json,
candidates.csv (every AOI with the reason when not eligible), selected.csv, summary.json.

Usage:
    python select_ems_wildfire.py --out_dir data/test/ems_wildfire_2025 \
        --footprints reports/pretrain_kept_footprints.csv --gee_project <your-gee-project>
"""
import argparse
import csv
import datetime as dt
import json
import os
import random
import time

import requests
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union
from shapely.validation import make_valid

API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"
S3 = "https://rapidmapping-viewer.s3.eu-west-1.amazonaws.com"


def get_json(url, sleep, tries=5):
    for k in range(tries):
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            time.sleep(sleep)
            return r.json()
        time.sleep(sleep * (k + 2))                                       # throttled: back off
    raise RuntimeError(f"{url}: HTTP {r.status_code}")


def label_product(aoi):
    """Latest DEL/GRA product with an observedEventA layer -> (product, observedEventA url, notAnalysedA url, date)."""
    best = None
    for p in aoi.get("products", []):
        if p.get("type") not in ("DEL", "GRA"):
            continue
        jsons = [l.get("json") for l in p.get("layers", []) if l.get("json")]
        obs = [u for u in jsons if "observedEventA" in u]
        if not obs:
            continue
        times = [im["acquisitionTime"] for im in p.get("images", []) if im.get("acquisitionTime")]
        if not times:
            continue
        nan = [u for u in jsons if "notAnalysedA" in u]
        cand = (max(times), p, obs[0], nan[0] if nan else None)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return None
    return best[1], best[2], best[3], best[0]


def burnt_ha(product):
    s = (product.get("stats") or {}).get("Burnt area") or {}
    vals = []
    for v in s.values():                       # "total" is sometimes a string in the API (e.g. "883.2")
        try:
            vals.append(float(str(v.get("total")).replace(",", "")))
        except (TypeError, ValueError):
            pass
    return sum(vals) if vals else None


def local_metric(geom_ll):
    c = geom_ll.centroid
    fwd = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +units=m", always_xy=True).transform
    return lambda g: shp_transform(fwd, g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--footprints", required=True)
    ap.add_argument("--gee_project", required=True)
    ap.add_argument("--year", default="2025")
    ap.add_argument("--min_burnt_ha", type=float, default=500.0)
    ap.add_argument("--buffer_km", type=float, default=10.0)
    ap.add_argument("--pre_days", type=int, default=90)
    ap.add_argument("--post_days", type=int, default=14)
    ap.add_argument("--s2_cloud_max", type=float, default=80.0,
                    help="tile-level CLOUDY_PIXEL_PERCENTAGE pre-filter (speed only); the decision uses --aoi_cloud_max")
    ap.add_argument("--aoi_cloud_max", type=float, default=0.01,
                    help="max fraction of cloudy pixels INSIDE the AOI (S2_CLOUD_PROBABILITY > --cloud_prob)")
    ap.add_argument("--cloud_prob", type=int, default=40)
    ap.add_argument("--max_gap_days", type=float, default=15.0, help="max days between S2 pre and S1 post")
    ap.add_argument("--max_per_activation", type=int, default=2)
    ap.add_argument("--n_select", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--from_dir", default=None,
                    help="reuse api/*.json and labels/*.json of a previous run folder instead of downloading them")
    args = ap.parse_args()
    print(vars(args), flush=True)
    os.makedirs(args.out_dir, exist_ok=False)
    for sub in ("api", "labels"):
        os.makedirs(os.path.join(args.out_dir, sub))

    # ---- 1. EMS API ----
    src = args.from_dir
    index = (json.load(open(os.path.join(src, "api", "index.json"))) if src
             else get_json(f"{API}/public-activations-info/?limit=500", args.sleep))
    with open(os.path.join(args.out_dir, "api", "index.json"), "x") as f:
        json.dump(index, f)
    acts = [a for a in index.get("results", index) if a.get("category") == "Wildfire"
            and str(a.get("eventTime", ""))[:4] == args.year]
    print(f"{len(acts)} wildfire activations in {args.year} in the index", flush=True)
    records = []
    for a in acts:
        rec = (json.load(open(os.path.join(src, "api", f"{a['code']}.json"))) if src
               else get_json(f"{API}/public-activations/?code={a['code']}", args.sleep))
        rec = rec["results"][0] if "results" in rec else rec
        with open(os.path.join(args.out_dir, "api", f"{a['code']}.json"), "x") as f:
            json.dump(rec, f)
        records.append(rec)

    # ---- 2-3. candidates + labels ----
    rows = []
    for rec in records:
        for aoi in rec.get("aois", []):
            row = dict(code=rec["code"], aoi=int(aoi["number"]), aoi_name=aoi.get("name"),
                       country=";".join(c["name"] for c in rec.get("countries", [])), sub_category=rec.get("subCategory"),
                       event_time=rec["eventTime"], aoi_extent_wkt=aoi["extent"], reason="")
            if rec.get("subCategory") != "Forest fire":
                row["reason"] = "not forest fire"; rows.append(row); continue
            lp = label_product(aoi)
            if lp is None:
                row["reason"] = "no DEL/GRA observedEventA"; rows.append(row); continue
            prod, obs_url, nan_url, label_date = lp
            row.update(product_type=prod["type"], product_id=prod["id"], monitoring_number=prod.get("monitoringNumber"),
                       product_version=(prod.get("version") or {}).get("number"), label_date=label_date,
                       label_sensors=";".join(sorted({im.get("sensorName", "") for im in prod.get("images", [])})),
                       burnt_ha_stat=burnt_ha(prod), observed_url=obs_url, not_analysed_url=nan_url or "")
            if row["burnt_ha_stat"] is None or row["burnt_ha_stat"] < args.min_burnt_ha:
                row["reason"] = f"burnt area < {args.min_burnt_ha:.0f} ha or missing"; rows.append(row); continue
            tag = f"{rec['code']}_AOI{int(aoi['number']):02d}"
            for kind, url in (("observedEventA", obs_url), ("notAnalysedA", nan_url)):
                if not url:
                    continue
                if src:
                    content = open(os.path.join(src, "labels", f"{tag}_{kind}.json"), "rb").read()
                else:
                    url = url if url.startswith("http") else f"{S3}/{url.lstrip('/')}"
                    r = requests.get(url, timeout=120)
                    if r.status_code != 200:
                        row["reason"] = f"{kind} download HTTP {r.status_code}"
                        break
                    content = r.content
                    time.sleep(args.sleep / 2)
                with open(os.path.join(args.out_dir, "labels", f"{tag}_{kind}.json"), "xb") as f:
                    f.write(content)
            rows.append(row)
    cands = [r for r in rows if not r["reason"]]
    print(f"{len(rows)} AOIs, {len(cands)} candidates after product / area filters", flush=True)

    # ---- 4. leakage ----
    feet = []
    with open(args.footprints) as f:
        for r in csv.DictReader(f):
            feet.append((float(r["center_lon"]), float(r["center_lat"]), r["footprint_wkt"]))
    for row in cands:
        tag = f"{row['code']}_AOI{row['aoi']:02d}"
        aoi_ll = make_valid(wkt.loads(row["aoi_extent_wkt"]))
        burnt = unary_union([make_valid(shape(ft["geometry"])) for ft in
                             json.load(open(os.path.join(args.out_dir, "labels", f"{tag}_observedEventA.json")))["features"]
                             if ft.get("geometry")])
        to_m = local_metric(aoi_ll)
        aoi_m, burnt_m = make_valid(to_m(aoi_ll)), make_valid(to_m(burnt))   # EMS polygons can be self-intersecting
        x0, y0, x1, y1 = aoi_ll.bounds
        near = [wkt.loads(w) for lon, lat, w in feet if x0 - 1 < lon < x1 + 1 and y0 - 1 < lat < y1 + 1]
        excl = unary_union([to_m(g).buffer(args.buffer_km * 1000) for g in near]) if near else None
        keep_aoi = aoi_m.difference(excl) if excl is not None else aoi_m
        keep_burnt = burnt_m.intersection(keep_aoi)
        row.update(aoi_km2=round(aoi_m.area / 1e6, 2), burnt_ha_polygons=round(burnt_m.area / 1e4, 1),
                   n_pretrain_within_buffer=int(sum(1 for g in near if to_m(g).distance(aoi_m) <= args.buffer_km * 1000)),
                   aoi_kept_frac=round(keep_aoi.area / aoi_m.area, 3),
                   burnt_ha_kept=round(keep_burnt.area / 1e4, 1))
        if row["burnt_ha_kept"] < args.min_burnt_ha:
            row["reason"] = f"burnt area outside {args.buffer_km:.0f} km leakage buffer < {args.min_burnt_ha:.0f} ha"

    # ---- 5. image availability (GEE) ----
    import ee
    ee.Initialize(project=args.gee_project)
    for row in [r for r in cands if not r["reason"]]:
        geom = ee.Geometry(json.loads(json.dumps(wkt.loads(row["aoi_extent_wkt"]).__geo_interface__)))
        ev = dt.datetime.fromisoformat(row["event_time"])
        ld = dt.datetime.fromisoformat(row["label_date"])
        t0, t1 = (ev - dt.timedelta(days=args.pre_days)).isoformat(), (ev - dt.timedelta(days=1)).isoformat()
        s2raw = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geom).filterDate(t0, t1)
                 .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", args.s2_cloud_max))
                 .filter(ee.Filter.contains(leftField=".geo", rightValue=geom)))
        cprob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(geom).filterDate(t0, t1)
        joined = ee.ImageCollection(ee.Join.saveFirst("cp").apply(
            s2raw, cprob, ee.Filter.equals(leftField="system:index", rightField="system:index")))

        def aoi_cloud(im, geom=geom):
            frac = (ee.Image(im.get("cp")).select("probability").gt(args.cloud_prob)
                    .reduceRegion(ee.Reducer.mean(), geom, 20, maxPixels=1e10, bestEffort=True).get("probability"))
            return im.set("aoi_cloud", frac)
        s2 = (joined.map(aoi_cloud).filter(ee.Filter.lte("aoi_cloud", args.aoi_cloud_max))
              .sort("system:time_start", False))
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom)
              .filterDate((ld + dt.timedelta(seconds=1)).isoformat(), (ld + dt.timedelta(days=args.post_days)).isoformat())
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
              .filter(ee.Filter.contains(leftField=".geo", rightValue=geom)).sort("system:time_start"))
        n2, n1 = s2.size().getInfo(), s1.size().getInfo()
        row.update(n_s2_pre=n2, n_s1_post=n1)
        if n2:
            im = ee.Image(s2.first())
            row.update(s2_id=im.get("system:index").getInfo(),
                       s2_date=ee.Date(im.get("system:time_start")).format("YYYY-MM-dd'T'HH:mm").getInfo(),
                       s2_cloud_tile=im.get("CLOUDY_PIXEL_PERCENTAGE").getInfo(),
                       s2_cloud_aoi=im.get("aoi_cloud").getInfo())
        if n1:
            im = ee.Image(s1.first())
            row.update(s1_id=im.get("system:index").getInfo(),
                       s1_date=ee.Date(im.get("system:time_start")).format("YYYY-MM-dd'T'HH:mm").getInfo(),
                       s1_orbit=im.get("orbitProperties_pass").getInfo(), s1_platform=im.get("platform_number").getInfo())
        if n2 and n1:
            gap = (dt.datetime.fromisoformat(row["s1_date"]) - dt.datetime.fromisoformat(row["s2_date"])).total_seconds() / 86400
            row["gap_s2_s1_days"] = round(gap, 1)
        if not n2:
            row["reason"] = f"no S2 L1C pre-event image covering the AOI with AOI cloud <= {args.aoi_cloud_max}"
        elif not n1:
            row["reason"] = f"no S1 GRD IW VV+VH within {args.post_days} days after the label image covering the AOI"
        elif row["gap_s2_s1_days"] > args.max_gap_days:
            row["reason"] = f"S2 pre -> S1 post gap {row['gap_s2_s1_days']} days > {args.max_gap_days:.0f}"
        print(f"  {row['code']} AOI{row['aoi']:02d}: S2 pre {n2}, S1 post {n1}, burnt kept {row['burnt_ha_kept']} ha"
              f" {('-> ' + row['reason']) if row['reason'] else ''}", flush=True)

    # ---- 6. selection ----
    eligible = [r for r in cands if not r["reason"]]
    order = list(range(len(eligible)))
    random.Random(args.seed).shuffle(order)
    per_act, selected = {}, []
    for i in order:
        r = eligible[i]
        if per_act.get(r["code"], 0) >= args.max_per_activation:
            continue
        per_act[r["code"]] = per_act.get(r["code"], 0) + 1
        selected.append(r)
        if len(selected) == args.n_select:
            break
    for r in rows:
        r["selected"] = r in selected

    cols = sorted({k for r in rows for k in r})
    for name, data in (("candidates.csv", rows), ("selected.csv", selected)):
        with open(os.path.join(args.out_dir, name), "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    reasons = {}
    for r in rows:
        reasons[r["reason"] or "eligible"] = reasons.get(r["reason"] or "eligible", 0) + 1
    summary = dict(config=vars(args), n_activations=len(records), n_aois=len(rows), n_candidates=len(cands),
                   n_eligible=len(eligible), n_selected=len(selected), reasons=reasons,
                   selected=[f"{r['code']}_AOI{r['aoi']:02d} {r['country']} burnt_kept={r['burnt_ha_kept']} ha "
                             f"S2 {r.get('s2_date')} S1 {r.get('s1_date')} gap {r.get('gap_s2_s1_days')} d" for r in selected])
    with open(os.path.join(args.out_dir, "summary.json"), "x") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
