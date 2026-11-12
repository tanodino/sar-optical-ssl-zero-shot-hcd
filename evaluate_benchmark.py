"""
Evaluation of continuous change maps on the EMS Wildfires 2025 benchmark, identical for every method.

For each method, one continuous change map per scene (float array (H, W) saved with numpy; higher = more likely changed) is
read on the valid pixels of the scene and compared with the burnt-area reference:
  * AUROC and AUPRC per scene and pooled over all valid pixels of the scenes (exact, from float16 score histograms, as
    sklearn's roc_auc_score / average_precision_score; see cva.py), and the mean of the per-scene values;
  * average rank of the methods over the scenes (1 = best) for each measure, Friedman test, and Nemenyi critical difference
    at alpha = 0.05 (Demsar, JMLR 2006), with the pairs whose rank difference exceeds it.
Methods given with --no_rank (e.g. encoder-only variants) get metrics but take no part in ranks and tests.

Usage (the paper's comparison):
    python evaluate_benchmark.py --scenes_root data/scenes --out_dir results/benchmark \\
        --map "DeCUR=features/decur/{scene}/B_w8_proj_common.npy" \\
        --map "CROMA=features/croma/{scene}/B_w8_k10_head.npy" \\
        --map "CroCoMAE=features/crocomae/{scene}/B_w8_k10_avg_shared.npy" \\
        --map "CAA=caa/{scene}/diff_crf.npy" --map "SCCN=sccn/{scene}/diff_crf.npy" \\
        --map "cGAN=cgan/{scene}/diff_crf.npy" --map "CDRL=cdrl/{scene}/diff_raw.npy" \\
        --map "DeCUR (encoder)=features/decur/{scene}/B_w8_enc_cls.npy" --no_rank "DeCUR (encoder)"
({scene} is replaced by each scene folder name, e.g. EMSR832_AOI01.)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import friedmanchisquare, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cva  # noqa: E402

# Nemenyi critical values q_0.05 for k = 2 .. 10 methods (Demsar, JMLR 2006, Table 5a)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes_root", required=True, help="folder of the scene arrays (prepare_scenes.py)")
    ap.add_argument("--map", action="append", required=True, metavar="NAME=PATTERN",
                    help="method name and path pattern of its maps, with {scene}; repeat for each method")
    ap.add_argument("--no_rank", action="append", default=[], help="method names excluded from ranks and tests")
    ap.add_argument("--scenes", nargs="+", default=None, help="subset of scenes (default: every scene folder)")
    ap.add_argument("--out_dir", required=True, help="new folder (refused if it exists)")
    args = ap.parse_args()
    assert not os.path.exists(args.out_dir), args.out_dir
    maps = dict(m.split("=", 1) for m in args.map)
    scenes = args.scenes or sorted(d for d in os.listdir(args.scenes_root)
                                   if os.path.isfile(os.path.join(args.scenes_root, d, "label.npy")))
    hist = {m: {} for m in maps}
    for s in scenes:
        lab, valid = cva.scene_truth(os.path.join(args.scenes_root, s))
        for m, pat in maps.items():
            x = np.load(pat.format(scene=s)).astype(np.float32)
            assert x.shape == lab.shape, (m, s, x.shape, lab.shape)
            hist[m][s] = cva.histograms(x[valid], lab[valid])
        print(s, "read", flush=True)
    per, summ = [], {}
    for m in maps:
        P = sum(hist[m][s][0] for s in scenes); N = sum(hist[m][s][1] for s in scenes)
        ps = {s: dict(AUROC=cva.auroc(*hist[m][s]), AUPRC=cva.auprc(*hist[m][s])) for s in scenes}
        summ[m] = dict(AUROC_pooled=cva.auroc(P, N), AUPRC_pooled=cva.auprc(P, N),
                       AUROC_mean=float(np.mean([v["AUROC"] for v in ps.values()])),
                       AUPRC_mean=float(np.mean([v["AUPRC"] for v in ps.values()])))
        per += [dict(method=m, scene=s, **v) for s, v in ps.items()]
    ranked = [m for m in maps if m not in args.no_rank]
    tests = {}
    if len(ranked) >= 2:
        k, n = len(ranked), len(scenes)
        cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n)) if k in Q05 else float("nan")
        for meas in ("AUROC", "AUPRC"):
            X = np.array([[next(r[meas] for r in per if r["method"] == m and r["scene"] == s) for m in ranked] for s in scenes])
            R = dict(zip(ranked, np.array([rankdata(-row) for row in X]).mean(0).tolist()))
            chi2, p = friedmanchisquare(*X.T) if k >= 3 else (float("nan"), float("nan"))
            sig = [[a, b] for a in ranked for b in ranked if R[b] - R[a] > cd]
            tests[meas] = dict(avg_rank=R, friedman_chi2=float(chi2), friedman_p=float(p), N=n, k=k,
                               nemenyi_cd_005=float(cd), significantly_better=sig)
            for m in ranked:
                summ[m][f"{meas}_rank"] = R[m]
    os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, "per_scene.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scene", "AUROC", "AUPRC"]); w.writeheader(); w.writerows(per)
    with open(os.path.join(args.out_dir, "summary.json"), "x") as f:
        json.dump(dict(scenes=scenes, maps=maps, no_rank=args.no_rank, metrics=summ, tests=tests), f, indent=1)
    print(f"{'method':18s} AUROC pooled / mean / rank   AUPRC pooled / mean / rank   ({len(scenes)} scenes)")
    for m, v in summ.items():
        r1, r2 = v.get("AUROC_rank", float("nan")), v.get("AUPRC_rank", float("nan"))
        print(f"{m:18s} {v['AUROC_pooled']:.3f} / {v['AUROC_mean']:.3f} / {r1:4.2f}     "
              f"{v['AUPRC_pooled']:.3f} / {v['AUPRC_mean']:.3f} / {r2:4.2f}")
    for meas, t in tests.items():
        print(f"{meas}: Friedman p = {t['friedman_p']:.2e}, Nemenyi CD = {t['nemenyi_cd_005']:.2f}, "
              f"significant: {', '.join(f'{a} > {b}' for a, b in t['significantly_better'])}")


if __name__ == "__main__":
    main()
