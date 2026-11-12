"""
Change maps -> metrics. The CVA score itself (||l2n(f_opt) - l2n(f_sar)||, and its bias-corrected version "<v>_bc") is
computed in extract_features.py; this module reads the maps and computes threshold-free metrics (AUROC, AUPRC), Otsu and
EM/Bayes thresholds, thresholded metrics (F1, Kappa, OA, precision, recall, FPR), per-scene and pooled aggregation, and
handles external binary maps (e.g. the official binary maps of the dedicated methods).

Maps -> evaluation pixels
  * Evaluation pixels = valid pixels of the scene; every map must be finite there (checked).
  * Token-level maps (<tag>_<variant>_tok.npy, one score per 8x8 patch) can be upsampled to pixels with nearest (block) or
    bilinear interpolation (upsample_tokens).
  * map_to_original / binary_to_original bring maps computed on an east-west resampled grid back to the original grid
    (ew_resample.backward_map); for the EMS benchmark the maps are already on the original grid (identity).

Exact metrics from float16 histograms
  Scores are non-negative and stored in float16. Every score is counted in a 65,536-bin histogram indexed by its float16
  bit pattern, which is monotonic in the value for non-negative float16. Per scene we keep two histograms (changed /
  unchanged pixels); pooled metrics use the sum over scenes. AUROC (trapezoidal ROC, ties handled as
  sklearn.metrics.roc_auc_score) and AUPRC (average precision, as sklearn.metrics.average_precision_score) are exact for
  the float16 scores (unit-tested against sklearn).

Thresholding
  Otsu: per scene, on the valid-pixel score histogram (float16 bins); changed = score > threshold.
  EM/Bayes: 2-Gaussian mixture fitted per scene by weighted EM on the same histogram, initialised from the Otsu split,
  stopped when the relative change of the log-likelihood is < 1e-8 or after 1000 iterations; threshold where
  pi0 N(x; mu0, s0) = pi1 N(x; mu1, s1) between the two means; no crossing -> undefined for that scene (reported).
  Pooled thresholded metrics = sum of the per-scene confusion matrices (each scene with its own threshold).

Run the tests: python -m unittest tests/test_cva.py -v
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ew_resample import backward_map                                         # noqa: E402

NBINS = 1 << 16
_VALUES = np.arange(NBINS, dtype=np.uint16).view(np.float16).astype(np.float64)   # value of every float16 bit pattern
POS_FINITE = np.arange(NBINS) < 0x7C00                                       # non-negative finite float16 patterns


# ----------------------------------------------------------------------------------------------- maps
def upsample_tokens(tok: np.ndarray, H: int, W: int, mode: str = "nearest", P: int = 8) -> np.ndarray:
    """Token map (ceil(H/P), ceil(W/P)), token (i, j) = patch rows [P i, P i + P) -> (H, W) float32.
    nearest: every pixel takes its patch score; bilinear: linear interpolation between patch centres (P i + P/2), values
    beyond the outer centres held constant. NaN tokens (inputs skipped at extraction) are ignored by bilinear: the weights of
    the finite neighbours are renormalised (identical to plain bilinear when all 4 neighbours are finite); NaN only if all 4
    neighbours are NaN."""
    tok = np.asarray(tok, np.float32)
    if mode == "nearest":
        return np.repeat(np.repeat(tok, P, 0), P, 1)[:H, :W]
    assert mode == "bilinear", mode

    def axis(n_out, n_grid):
        u = np.clip((np.arange(n_out) + 0.5 - P / 2) / P, 0, n_grid - 1)
        k0 = np.minimum(np.floor(u).astype(np.int64), max(n_grid - 2, 0))
        f = (u - k0) if n_grid > 1 else np.zeros(n_out)
        return k0, np.minimum(k0 + 1, n_grid - 1), f.astype(np.float32)
    y0, y1, fy = axis(H, tok.shape[0])
    x0, x1, fx = axis(W, tok.shape[1])
    ok = np.isfinite(tok)
    val = np.where(ok, tok, 0).astype(np.float64)
    num = np.zeros((H, W)); den = np.zeros((H, W))
    for yi, wy in ((y0, 1 - fy), (y1, fy)):
        for xi, wx in ((x0, 1 - fx), (x1, fx)):
            w = wy[:, None].astype(np.float64) * wx[None].astype(np.float64) * ok[yi][:, xi]
            num += w * val[yi][:, xi]
            den += w
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out.astype(np.float32)


def scene_truth(scene_dir: str):
    """(label bool, valid bool) on the ORIGINAL grid of a scene folder."""
    return (np.load(os.path.join(scene_dir, "label.npy")) > 0,
            np.load(os.path.join(scene_dir, "valid.npy")).astype(bool))


def map_to_original(m: np.ndarray, orig_W: int, missing: np.ndarray = None) -> np.ndarray:
    """Map on the extraction grid -> original grid (identity if the widths match; else backward_map along x).
    missing (bool, extraction grid): pixels without a computed score (CAA writes 0 on invalid ew pixels). They are ignored by
    the linear interpolation (weights of the available neighbour renormalised); an original pixel whose two neighbours are both
    missing takes the value of the nearest available ew pixel (2-D Euclidean, from the ew pixel containing its centre)."""
    m = np.asarray(m, np.float32)
    if m.shape[1] == orig_W:
        return m if missing is None else np.where(missing, np.nan, m).astype(np.float32)
    if missing is None:
        return backward_map(m, orig_W)
    ok = ~missing
    num = backward_map(np.where(ok, m, 0).astype(np.float32), orig_W)
    den = backward_map(ok.astype(np.float32), orig_W)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan).astype(np.float32)
    bad = np.isnan(out)
    if bad.any() and ok.any():                                        # nearest available ew pixel (2-D, Euclidean)
        from scipy.ndimage import distance_transform_edt
        W_new = m.shape[1]
        u = np.clip(np.floor((np.arange(orig_W) + 0.5) * W_new / orig_W).astype(np.int64), 0, W_new - 1)
        _, (ri, ci) = distance_transform_edt(~ok, return_indices=True)
        rr, jj = np.nonzero(bad)
        out[rr, jj] = m[ri[rr, u[jj]], ci[rr, u[jj]]]
    return out


def binary_to_original(b: np.ndarray, orig_W: int) -> np.ndarray:
    """Binary map on the extraction grid -> original grid by NEAREST neighbour along x (the ew pixel containing the centre of
    the original pixel); identity if the widths match."""
    b = np.asarray(b)
    if b.shape[1] == orig_W:
        return b.astype(bool)
    W_new = b.shape[1]
    u = np.clip(np.floor((np.arange(orig_W) + 0.5) * W_new / orig_W).astype(np.int64), 0, W_new - 1)
    return b[:, u].astype(bool)


def caa_scene(caa_dir: str, truth_dir: str, ew_dir: str = None):
    """CAA outputs of one scene on the ORIGINAL grid, evaluation pixels only:
    returns (labels, {diff_crf, diff_raw: continuous scores}, {cm_crf, cm_raw: binary}) restricted to valid original pixels.
    ew_dir (GeoFlood): scenes_v1_ew/<scene> whose valid.npy marks the pixels where CAA computed a score."""
    label, valid = scene_truth(truth_dir)
    W = label.shape[1]
    miss = None if ew_dir is None else ~np.load(os.path.join(ew_dir, "valid.npy")).astype(bool)
    cont = {k: map_to_original(np.load(os.path.join(caa_dir, f"{k}.npy")), W, miss)[valid] for k in ("diff_crf", "diff_raw")}
    binm = {k: binary_to_original(np.load(os.path.join(caa_dir, f"{k}.npy")), W)[valid] for k in ("cm_crf", "cm_raw")}
    return label[valid], cont, binm


# ----------------------------------------------------------------------------------------------- histograms / metrics
def histograms(scores: np.ndarray, labels: np.ndarray):
    """scores (n,) finite, >= 0; labels (n,) bool -> (pos, neg) int64 counts per float16 bin."""
    s = np.asarray(scores, np.float32)
    assert np.isfinite(s).all(), "non-finite score on an evaluation pixel"
    assert (s >= 0).all(), "negative score"
    k = s.astype(np.float16).view(np.uint16)
    lab = np.asarray(labels, bool)
    return (np.bincount(k[lab], minlength=NBINS).astype(np.int64),
            np.bincount(k[~lab], minlength=NBINS).astype(np.int64))


def _desc_cum(pos, neg):
    """Cumulative TP / FP when lowering the threshold through the occupied bins (high -> low)."""
    occ = np.nonzero((pos + neg) > 0)[0][::-1]
    return np.cumsum(pos[occ]), np.cumsum(neg[occ]), occ


def auroc(pos, neg) -> float:
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    tp, fp, _ = _desc_cum(pos, neg)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def auprc(pos, neg) -> float:
    """Average precision: sum over thresholds of (R_k - R_{k-1}) * P_k (no interpolation), as sklearn."""
    P = pos.sum()
    if P == 0:
        return float("nan")
    tp, fp, _ = _desc_cum(pos, neg)
    rec = np.concatenate([[0.0], tp / P])
    prec = tp / (tp + fp)
    return float(np.sum((rec[1:] - rec[:-1]) * prec))


def otsu_bin(pos, neg) -> int:
    """Otsu on the total histogram: returns bin index t maximising the between-class variance of {<= t} vs {> t}
    (first maximum). changed = bin > t."""
    h = (pos + neg).astype(np.float64)
    occ = np.nonzero(h)[0]
    if len(occ) < 2:
        return int(occ[-1]) if len(occ) else 0
    w = h[occ]
    v = _VALUES[occ]
    w0 = np.cumsum(w)[:-1]
    m0 = np.cumsum(w * v)[:-1]
    W, M = w.sum(), (w * v).sum()
    w1, m1 = W - w0, M - m0
    between = w0 * w1 * (m0 / w0 - m1 / w1) ** 2
    return int(occ[int(np.argmax(between))])


def confusion(pos, neg, t_bin: int):
    """changed = bin > t_bin -> (tp, fp, fn, tn)."""
    hi = np.arange(NBINS) > t_bin
    tp, fp = int(pos[hi].sum()), int(neg[hi].sum())
    return tp, fp, int(pos.sum()) - tp, int(neg.sum()) - fp


def confusion_value(pos, neg, thr: float):
    """changed = score > thr (thr a real value) -> (tp, fp, fn, tn)."""
    hi = POS_FINITE & (_VALUES > thr)
    tp, fp = int(pos[hi].sum()), int(neg[hi].sum())
    return tp, fp, int(pos.sum()) - tp, int(neg.sum()) - fp


def confusion_binary(pred: np.ndarray, labels: np.ndarray):
    """External binary map (e.g. CAA cm_crf) on evaluation pixels -> (tp, fp, fn, tn)."""
    p, y = np.asarray(pred, bool), np.asarray(labels, bool)
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    return tp, fp, int((~p & y).sum()), int((~p & ~y).sum())


def gaussian_crossing(pi0, mu0, v0, pi1, mu1, v1):
    """x in (mu0, mu1) with pi0 N(x; mu0, v0) = pi1 N(x; mu1, v1) (v = variance), or None. If several roots lie between
    the means (not possible for a quadratic with both roots inside unless degenerate), the one closest to the midpoint."""
    # log pi0 - log s0 - (x-mu0)^2 / (2 v0) = log pi1 - log s1 - (x-mu1)^2 / (2 v1)  ->  a x^2 + b x + c = 0
    a = 1 / (2 * v1) - 1 / (2 * v0)
    b = mu0 / v0 - mu1 / v1
    c = (mu1 ** 2 / (2 * v1) - mu0 ** 2 / (2 * v0) + np.log(pi0) - np.log(pi1) - 0.5 * np.log(v0) + 0.5 * np.log(v1))
    if abs(a) < 1e-15 * max(abs(b), 1e-300):
        roots = [-c / b] if b != 0 else []
    else:
        disc = b * b - 4 * a * c
        roots = [] if disc < 0 else [(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]
    lo, hi = min(mu0, mu1), max(mu0, mu1)
    inside = [r for r in roots if lo < r < hi]
    return float(min(inside, key=lambda r: abs(r - (lo + hi) / 2))) if inside else None


def em_threshold(pos, neg, tol: float = 1e-8, max_iter: int = 1000) -> dict:
    """EM/Bayes threshold of one scene (2-Gaussian mixture on the histogram, Otsu initialisation). Returns dict(threshold or
    None, pi, mu, var (lists, class 0 = low scores), iterations, converged)."""
    h = (pos + neg).astype(np.float64)
    occ = np.nonzero(h)[0]
    w, x = h[occ], _VALUES[occ]
    out = dict(threshold=None, pi=None, mu=None, var=None, iterations=0, converged=False)
    if len(occ) < 2:
        return out
    t = otsu_bin(pos, neg)
    side = occ > t
    vfloor = 1e-12 + 1e-6 * np.average((x - np.average(x, weights=w)) ** 2, weights=w)   # avoids a collapsing component
    pi = np.array([w[~side].sum(), w[side].sum()]) / w.sum()
    mu = np.array([np.average(x[~side], weights=w[~side]), np.average(x[side], weights=w[side])])
    var = np.maximum(np.array([np.average((x[~side] - mu[0]) ** 2, weights=w[~side]),
                               np.average((x[side] - mu[1]) ** 2, weights=w[side])]), vfloor)
    prev = -np.inf
    for it in range(1, max_iter + 1):
        logp = np.log(pi)[:, None] - 0.5 * np.log(2 * np.pi * var)[:, None] - (x[None] - mu[:, None]) ** 2 / (2 * var[:, None])
        m = logp.max(0)
        lse = m + np.log(np.exp(logp - m).sum(0))
        ll = float((w * lse).sum())
        r = np.exp(logp - lse) * w                                     # weighted responsibilities (2, n_bins)
        nk = r.sum(1)
        if (nk <= 0).any():
            break
        pi = nk / nk.sum()
        mu = (r * x).sum(1) / nk
        var = np.maximum((r * (x[None] - mu[:, None]) ** 2).sum(1) / nk, vfloor)
        out["iterations"] = it
        if abs(ll - prev) <= tol * abs(ll):
            out["converged"] = True
            break
        prev = ll
    o = np.argsort(mu)
    pi, mu, var = pi[o], mu[o], var[o]
    out.update(pi=pi.tolist(), mu=mu.tolist(), var=var.tolist(),
               threshold=gaussian_crossing(pi[0], mu[0], var[0], pi[1], mu[1], var[1]))
    return out


def thresholded_metrics(tp, fp, fn, tn) -> dict:
    n = tp + fp + fn + tn
    oa = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else float("nan")
    kappa = (oa - pe) / (1 - pe) if n and pe != 1 else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")                  # false-positive rate
    return dict(OA=oa, precision=prec, recall=rec, F1=f1, kappa=kappa, FPR=fpr, tp=tp, fp=fp, fn=fn, tn=tn)


EM_KEYS = ("OA", "precision", "recall", "F1", "kappa", "FPR", "tp", "fp", "fn", "tn")


def evaluate_hist(pos, neg, fixed: dict = None) -> dict:
    """All metrics of one histogram pair (a scene): threshold-free, Otsu (plain keys) and EM (keys em_*).
    fixed = dict(otsu=value, em=em_threshold-dict) imposes thresholds computed elsewhere (SAR baseline: thresholds from the
    unmodified VV score, applied after the permanent-water removal); changed = score > value."""
    out = dict(AUROC=auroc(pos, neg), AUPRC=auprc(pos, neg), n_pos=int(pos.sum()), n_neg=int(neg.sum()))
    if fixed is None:
        t = otsu_bin(pos, neg)
        out["otsu_threshold"] = float(_VALUES[t]) if t < 0x7C00 else float("nan")
        out.update(thresholded_metrics(*confusion(pos, neg, t)))
        em = em_threshold(pos, neg)
    else:
        out["otsu_threshold"] = float(fixed["otsu"])
        out.update(thresholded_metrics(*confusion_value(pos, neg, fixed["otsu"])))
        em = fixed["em"]
    out.update(em_threshold=em["threshold"], em_converged=em["converged"], em_iterations=em["iterations"],
               em_pi=em["pi"], em_mu=em["mu"], em_var=em["var"])
    m = thresholded_metrics(*confusion_value(pos, neg, em["threshold"])) if em["threshold"] is not None else \
        {k: (float("nan") if k not in ("tp", "fp", "fn", "tn") else None) for k in EM_KEYS}
    out.update({f"em_{k}": v for k, v in m.items()})
    return out


def aggregate(scene_hists: dict, fixed: dict = None) -> dict:
    """scene_hists {scene: (pos, neg)} -> dict(per_scene, pooled, mean_of_scenes).
    pooled: AUROC / AUPRC on the summed histograms; thresholded metrics from the summed per-scene confusion matrices
    (per-scene Otsu thresholds). mean_of_scenes: mean of the per-scene metrics over scenes where they are defined."""
    per = {s: evaluate_hist(p, n, None if fixed is None else fixed[s]) for s, (p, n) in scene_hists.items()}
    P = sum(p for p, _ in scene_hists.values())
    N = sum(n for _, n in scene_hists.values())
    conf = [sum(per[s][k] for s in per) for k in ("tp", "fp", "fn", "tn")]
    pooled = dict(AUROC=auroc(P, N), AUPRC=auprc(P, N), n_pos=int(P.sum()), n_neg=int(N.sum()))
    pooled.update(thresholded_metrics(*conf))
    em_ok = [s for s in per if per[s]["em_threshold"] is not None]
    if em_ok:
        pooled.update({f"em_{k}": v for k, v in thresholded_metrics(
            *[sum(per[s][f"em_{k}"] for s in em_ok) for k in ("tp", "fp", "fn", "tn")]).items()})
    pooled["em_scenes_undefined"] = sorted(set(per) - set(em_ok))
    keys = ("AUROC", "AUPRC", "OA", "precision", "recall", "F1", "kappa", "FPR") + tuple(
        f"em_{k}" for k in ("OA", "precision", "recall", "F1", "kappa", "FPR"))
    mean = {k: float(np.nanmean([per[s][k] for s in per])) if any(np.isfinite(per[s][k]) for s in per) else float("nan")
            for k in keys}
    mean["n_scenes_defined"] = {k: int(sum(np.isfinite(per[s][k]) for s in per)) for k in keys}
    return dict(per_scene=per, pooled=pooled, mean_of_scenes=mean)


# ----------------------------------------------------------------------------------------------- scene-level driver
def aggregate_confusions(scene_conf: dict) -> dict:
    """{scene: (tp, fp, fn, tn)} of an external binary map -> per-scene, pooled (summed) and mean-of-scenes thresholded metrics."""
    per = {s: thresholded_metrics(*c) for s, c in scene_conf.items()}
    pooled = thresholded_metrics(*[sum(c[i] for c in scene_conf.values()) for i in range(4)])
    keys = ("OA", "precision", "recall", "F1", "kappa", "FPR")
    mean = {k: float(np.nanmean([per[s][k] for s in per])) for k in keys}
    return dict(per_scene=per, pooled=pooled, mean_of_scenes=mean)


def scene_hist(map_path: str, truth_dir: str, token: str = None):
    """Histogram pair of one map on one scene. token in {None, 'nearest', 'bilinear'}: map_path is a token map to upsample
    (on the extraction grid) before mapping to the original grid."""
    label, valid = scene_truth(truth_dir)
    m = np.load(map_path)
    if token:
        meta = os.path.join(os.path.dirname(map_path), "run.json")
        H, W = (json.load(open(meta))["H"], json.load(open(meta))["W"]) if os.path.exists(meta) else (label.shape[0], m.shape[1] * 8)
        m = upsample_tokens(m, H, W, token)
    m = map_to_original(m, label.shape[1])
    assert m.shape == label.shape, (m.shape, label.shape)
    return histograms(m[valid], label[valid])
