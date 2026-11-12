"""
East-west resampling helpers for scenes stored on a lat/lon grid (not needed for the EMS benchmark, which is on a UTM
10 m grid; kept because cva.py and extract_features.py import them).

Why: on a lat/lon grid pixels are ~10 m north-south but 10 m x cos(lat) east-west, while the encoders were pre-trained on
10 x 10 m UTM tiles. Each scene is resampled EAST-WEST ONLY to ~10 m pixels; change maps computed on that grid are resampled back to the
original grid, and all metrics use the original label pixels (labels / validity are never interpolated for scoring).

Geometry: W original columns of width 10*cos(lat) m -> W_new = round(W * cos(lat)) columns covering the same extent;
s = W / W_new = original pixels per new pixel (~1 / cos(lat)). lat = centroid of the EMS AOI extent.
Kernels:
  * images (forward): exact AREA AVERAGE of the original pixels (pixels = constant over their width), computed from
    the cumulative sum along x evaluated at fractional positions;
  * validity (forward): a new pixel is valid only if ALL original pixels overlapping it (positive overlap) are valid;
  * change maps (backward): LINEAR interpolation along x at the original pixel centres (edge values held constant).
"""
import json
import os
import re

import numpy as np
from shapely import wkt


def aoi_latitude(scene_id: str, ems_dir: str) -> float:
    """Latitude of the centroid of the EMS AOI extent (same rule as check_grid_anisotropy.py)."""
    m = re.match(r"(EMSR\d+)_AOI(\d+)", scene_id)
    act = json.load(open(os.path.join(ems_dir, f"{m.group(1)}.json")))
    act = act["results"][0] if "results" in act else act
    a = [x for x in act["aois"] if x["number"] == int(m.group(2))][0]
    return float(wkt.loads(a["extent"]).centroid.y)


def new_width(W: int, lat: float) -> int:
    return int(round(W * np.cos(np.radians(lat))))


def _prefix(x: np.ndarray) -> np.ndarray:
    """Cumulative integral along the last axis at integer pixel edges: F[..., k] = sum of x[..., :k], F[..., 0] = 0."""
    F = np.zeros(x.shape[:-1] + (x.shape[-1] + 1,), np.float64)
    np.cumsum(x, axis=-1, out=F[..., 1:])
    return F


def _eval_prefix(F: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of F at fractional edge positions `pos` (in original pixel units)."""
    k = np.clip(np.floor(pos).astype(np.int64), 0, F.shape[-1] - 2)
    f = pos - k
    return F[..., k] * (1 - f) + F[..., k + 1] * f


def forward_image(x: np.ndarray, W_new: int, row_block: int = 512) -> np.ndarray:
    """Area-average resampling along the last axis: (..., H, W) -> (..., H, W_new), float32."""
    W = x.shape[-1]
    s = W / W_new
    edges = np.arange(W_new + 1) * s                                  # new pixel edges in original pixel units
    out = np.empty(x.shape[:-1] + (W_new,), np.float32)
    H = x.shape[-2]
    for r0 in range(0, H, row_block):                                 # row blocks to bound memory
        F = _prefix(np.asarray(x[..., r0:r0 + row_block, :], np.float64))
        G = _eval_prefix(F, edges)
        out[..., r0:r0 + row_block, :] = (np.diff(G, axis=-1) / s).astype(np.float32)
    return out


def forward_valid(valid: np.ndarray, W_new: int, row_block: int = 512) -> np.ndarray:
    """New pixel valid iff every original pixel with positive overlap is valid: (H, W) -> (H, W_new) bool."""
    W = valid.shape[-1]
    s = W / W_new
    edges = np.arange(W_new + 1) * s
    lo = np.floor(edges[:-1] + 1e-9).astype(np.int64)                 # first overlapped original pixel
    hi = np.minimum(np.ceil(edges[1:] - 1e-9).astype(np.int64), W)    # one past the last overlapped pixel
    out = np.empty(valid.shape[:-1] + (W_new,), bool)
    for r0 in range(0, valid.shape[-2], row_block):
        inv = _prefix((~np.asarray(valid[r0:r0 + row_block], bool)).astype(np.float64))
        out[r0:r0 + row_block] = (inv[:, hi] - inv[:, lo]) == 0
    return out


def backward_map(m: np.ndarray, W: int) -> np.ndarray:
    """Linear interpolation along the last axis from W_new columns back to the original W columns, float32."""
    W_new = m.shape[-1]
    s = W / W_new
    u = (np.arange(W) + 0.5) / s - 0.5                                # original centres in new-pixel index units
    u = np.clip(u, 0, W_new - 1)
    k = np.minimum(np.floor(u).astype(np.int64), W_new - 2) if W_new > 1 else np.zeros(W, np.int64)
    f = (u - k) if W_new > 1 else np.zeros(W)
    m = np.asarray(m, np.float32)
    return (m[..., k] * (1 - f) + m[..., np.minimum(k + 1, W_new - 1)] * f).astype(np.float32)


def resample_scene(scene_dir: str, out_dir: str, lat: float):
    """Write the east-west resampled scene (opt, sar, valid; label kept only for reference, NOT used for scoring)."""
    opt = np.load(os.path.join(scene_dir, "opt.npy"), mmap_mode="r")
    sar = np.load(os.path.join(scene_dir, "sar.npy"), mmap_mode="r")
    valid = np.load(os.path.join(scene_dir, "valid.npy"), mmap_mode="r").astype(bool)
    H, W = valid.shape
    Wn = new_width(W, lat)
    v_new = forward_valid(valid, Wn)
    os.makedirs(out_dir, exist_ok=False)
    for name, x in (("opt", opt), ("sar", sar)):
        y = forward_image(x, Wn)
        y[:, ~v_new] = 0.0                                            # invalid -> 0 (= band mean), as in the originals
        np.save(os.path.join(out_dir, f"{name}.npy"), y.astype(np.float16))
    np.save(os.path.join(out_dir, "valid.npy"), v_new.astype(np.uint8))
    meta = dict(source=os.path.abspath(scene_dir), lat=lat, cos_lat=float(np.cos(np.radians(lat))), H=H, W_orig=W,
                W_new=Wn, s=W / Wn, valid_frac_orig=float(valid.mean()), valid_frac_new=float(v_new.mean()),
                kernels=dict(images="area average", valid="all overlapped original pixels valid",
                             maps_back="linear interpolation (backward_map)"),
                note="labels are NOT resampled: metrics on the original grid after backward_map")
    with open(os.path.join(out_dir, "meta.json"), "x") as f:
        json.dump(meta, f, indent=1)
    return meta
