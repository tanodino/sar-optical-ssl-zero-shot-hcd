"""
Download of the imagery of the EMS wildfires 2025 benchmark (AOIs of selected.csv from select_ems_wildfire.py, ~0.5 GB)
from Google Earth Engine, with the collections used by SSL4EO-S12 v1.1:

  pre  : COPERNICUS/S2_HARMONIZED (Sentinel-2 L1C, DN = reflectance x 10000; the HARMONIZED collection already removes
         the +1000 offset of processing baseline >= 04.00, so NO offset must be subtracted later) bands B2, B3, B4, B8
         (native 10 m) -> s2.tif (uint16, 0 = no data); COPERNICUS/S2_CLOUD_PROBABILITY of the same image ->
         s2_cloudprob.tif (uint8 0-100, 255 = no data; native 20 m, nearest to the 10 m grid).
  post : COPERNICUS/S1_GRD (sigma0 dB: thermal noise removal, calibration, terrain correction, as processed by GEE) bands
         VV, VH -> s1.tif (float32, -9999 = no data), resampled to the S2 grid with GEE's default (nearest neighbour).

Grid: the native projection (UTM zone) of the selected S2 image, 10 m pixels aligned to its 10 m grid (no resampling of
the 10 m S2 bands), extent = AOI extent snapped outwards to that grid. Requests are tiled (--tile px) to stay under the
GEE download size limit, then mosaicked with rasterio. One folder per AOI with meta.json (ids, dates, grid).

Usage:
    python download_ems_wildfire.py --selected data/test/ems_wildfire_2025_d/selected.csv \
        --out_dir data/test/ems_wildfire_2025_v1 --gee_project <your-gee-project>
"""
import argparse
import csv
import io
import json
import math
import os
import time
import zipfile

import ee
import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import Affine
from shapely import wkt
from shapely.ops import transform as shp_transform


def fetch(image, crs, x0, y0, w, h, retries=4):
    """Download a w x h window (top-left x0, y0 in `crs`, 10 m) of `image` as a numpy array (bands, h, w)."""
    params = dict(crs=crs, crs_transform=[10, 0, x0, 0, -10, y0], dimensions=f"{w}x{h}", format="GEO_TIFF")
    for k in range(retries):
        try:
            url = image.getDownloadURL(params)
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            data = r.content
            if data[:2] == b"PK":                                        # zipped GeoTIFF(s)
                z = zipfile.ZipFile(io.BytesIO(data))
                data = z.read(sorted(n for n in z.namelist() if n.endswith(".tif"))[0])
            with rasterio.open(io.BytesIO(data)) as src:
                arr = src.read()
                assert src.width == w and src.height == h, (src.width, src.height, w, h)
                assert abs(src.transform.a - 10) < 1e-6 and abs(src.transform.c - x0) < 1e-3 and abs(src.transform.f - y0) < 1e-3
            return arr
        except Exception as e:                                           # transient GEE / network errors
            print(f"    retry {k + 1}/{retries}: {e}", flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError("download failed")


def download(image, crs, X0, Y0, W, H, tile, nbands, dtype):
    out = np.zeros((nbands, H, W), dtype=dtype)
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            h, w = min(tile, H - r0), min(tile, W - c0)
            out[:, r0:r0 + h, c0:c0 + w] = fetch(image, crs, X0 + 10 * c0, Y0 - 10 * r0, w, h)
    return out


def write(path, arr, crs, X0, Y0, nodata, descriptions):
    prof = dict(driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0], dtype=arr.dtype.name,
                crs=crs, transform=Affine(10, 0, X0, 0, -10, Y0), nodata=nodata, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)
        for i, d in enumerate(descriptions, 1):
            dst.set_band_description(i, d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gee_project", required=True)
    ap.add_argument("--tile", type=int, default=2048)
    args = ap.parse_args()
    print(vars(args), flush=True)
    os.makedirs(args.out_dir, exist_ok=False)
    ee.Initialize(project=args.gee_project)
    rows = list(csv.DictReader(open(args.selected)))
    t_all = time.time()
    for r in rows:
        tag = f"{r['code']}_AOI{int(r['aoi']):02d}"
        t0 = time.time()
        d = os.path.join(args.out_dir, tag)
        os.makedirs(d)
        s2 = ee.Image(f"COPERNICUS/S2_HARMONIZED/{r['s2_id']}")
        cp = ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{r['s2_id']}")
        s1 = ee.Image(f"COPERNICUS/S1_GRD/{r['s1_id']}")
        proj = s2.select("B2").projection().getInfo()                   # native UTM grid of the S2 image
        crs, tr = proj["crs"], proj["transform"]                         # [10, 0, x_origin, 0, -10, y_origin]
        to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
        aoi = shp_transform(to_utm, wkt.loads(r["aoi_extent_wkt"]))
        minx, miny, maxx, maxy = aoi.bounds
        ox, oy = tr[2], tr[5]                                             # snap outwards to the 10 m S2 grid
        X0 = ox + 10 * math.floor((minx - ox) / 10)
        Y0 = oy - 10 * math.floor((oy - maxy) / 10)
        W = int(math.ceil((maxx - X0) / 10))
        H = int(math.ceil((Y0 - miny) / 10))
        print(f"{tag}: {crs}, {W} x {H} px", flush=True)
        a_s2 = download(s2.select(["B2", "B3", "B4", "B8"]).toUint16().unmask(0), crs, X0, Y0, W, H, args.tile, 4, np.uint16)
        a_cp = download(cp.select("probability").toUint8().unmask(255), crs, X0, Y0, W, H, args.tile, 1, np.uint8)
        a_s1 = download(s1.select(["VV", "VH"]).toFloat().unmask(-9999), crs, X0, Y0, W, H, args.tile, 2, np.float32)
        write(os.path.join(d, "s2.tif"), a_s2, crs, X0, Y0, 0, ["B2", "B3", "B4", "B8"])
        write(os.path.join(d, "s2_cloudprob.tif"), a_cp, crs, X0, Y0, 255, ["probability"])
        write(os.path.join(d, "s1.tif"), a_s1, crs, X0, Y0, -9999, ["VV", "VH"])
        s1_info = s1.toDictionary(["orbitProperties_pass", "platform_number", "relativeOrbitNumber_start"]).getInfo()
        meta = dict(aoi=tag, country=r["country"], event_time=r["event_time"], label_date=r["label_date"],
                    label_product=dict(type=r["product_type"], id=r["product_id"], version=r["product_version"],
                                       sensors=r["label_sensors"]),
                    s2=dict(collection="COPERNICUS/S2_HARMONIZED", id=r["s2_id"], date=r["s2_date"],
                            aoi_cloud=float(r["s2_cloud_aoi"]), bands=["B2", "B3", "B4", "B8"], nodata=0,
                            offset_note="HARMONIZED: +1000 offset already removed; do NOT subtract 1000"),
                    cloudprob=dict(collection="COPERNICUS/S2_CLOUD_PROBABILITY", nodata=255),
                    s1=dict(collection="COPERNICUS/S1_GRD", id=r["s1_id"], date=r["s1_date"], bands=["VV", "VH"],
                            units="dB", nodata=-9999, resampling="GEE default (nearest neighbour)", **s1_info),
                    gap_s2_s1_days=float(r["gap_s2_s1_days"]),
                    grid=dict(crs=crs, transform=[10, 0, X0, 0, -10, Y0], width=W, height=H),
                    stats=dict(s2_nodata_frac=float((a_s2 == 0).all(0).mean()),
                               s1_nodata_frac=float((a_s1 == -9999).any(0).mean()),
                               s2_dn_percentiles={b: np.percentile(a_s2[i][a_s2[i] > 0], [1, 50, 99]).tolist()
                                                  for i, b in enumerate(["B2", "B3", "B4", "B8"])},
                               s1_db_percentiles={b: np.percentile(a_s1[i][a_s1[i] != -9999], [1, 50, 99]).tolist()
                                                  for i, b in enumerate(["VV", "VH"])}),
                    seconds=round(time.time() - t0, 1))
        with open(os.path.join(d, "meta.json"), "x") as f:
            json.dump(meta, f, indent=1)
        print(f"  done in {meta['seconds']} s; S2 no-data {meta['stats']['s2_nodata_frac']:.4f}, "
              f"S1 no-data {meta['stats']['s1_nodata_frac']:.4f}; S2 B4 DN p1/50/99 "
              f"{[round(v) for v in meta['stats']['s2_dn_percentiles']['B4']]}; S1 VV dB p1/50/99 "
              f"{[round(v, 1) for v in meta['stats']['s1_db_percentiles']['VV']]}", flush=True)
    print(f"all done in {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
