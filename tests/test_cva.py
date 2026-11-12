"""
Unit tests of cva.py, CPU only.
  1. float16 bit patterns are monotonic for non-negative values (basis of the histogram metrics).
  2. AUROC / AUPRC from histograms == sklearn roc_auc_score / average_precision_score (random scores with many ties),
     and known answers (perfect, reversed, constant scores).
  3. Thresholded metrics on a known confusion matrix (hand-computed values).
  4. Otsu == brute-force between-class-variance maximisation over the unique values; separates a clean bimodal sample.
  5. aggregate: pooled AUROC / AUPRC == sklearn on the concatenation; pooled confusion = sum of per-scene confusions;
     mean of scenes skips undefined values (scene without changed pixels).
  6. upsample_tokens: nearest = block expansion; bilinear reproduces a linear ramp on interior pixels.
The shifted-inference aggregation test lives in tests/test_extract_features.py (Mode A vs brute force,
strides 8 / 4 / 2 / 1; stride 8 = block map).
Run: python -m unittest tests/test_cva.py -v   (from the project root)
"""
import os
import sys
import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cva  # noqa: E402


def f16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float64)


def gauss_mix_sample(rng, n, pi1, mu, sd):
    y = rng.random(n) < pi1
    return np.where(y, rng.normal(mu[1], sd[1], n), rng.normal(mu[0], sd[0], n)), y


class TestEM(unittest.TestCase):
    """EM/Bayes: recovers the parameters of a known 2-Gaussian mixture; threshold == crossing of the TRUE weighted densities
    (within sampling / float16 error); gaussian_crossing on closed-form cases; None when no crossing between the means."""

    def test_recovery(self):
        rng = np.random.default_rng(5)
        for pi1, mu, sd in ((0.1, (0.4, 1.0), (0.08, 0.12)), (0.3, (0.6, 0.9), (0.1, 0.1)), (0.03, (0.3, 1.2), (0.05, 0.2))):
            x, y = gauss_mix_sample(rng, 400000, pi1, mu, sd)
            x = np.abs(x)
            pos, neg = cva.histograms(x, y)
            em = cva.em_threshold(pos, neg)
            self.assertTrue(em["converged"])
            np.testing.assert_allclose(em["pi"][1], pi1, atol=0.01)
            np.testing.assert_allclose(em["mu"], mu, atol=0.01)
            np.testing.assert_allclose(np.sqrt(em["var"]), sd, atol=0.01)
            ref = cva.gaussian_crossing(1 - pi1, mu[0], sd[0] ** 2, pi1, mu[1], sd[1] ** 2)
            self.assertAlmostEqual(em["threshold"], ref, delta=0.01)
            r = cva.evaluate_hist(pos, neg)
            tp, fp, fn, tn = cva.confusion_value(pos, neg, em["threshold"])
            self.assertEqual((r["em_tp"], r["em_fp"]), (tp, fp))
            self.assertAlmostEqual(r["em_FPR"], fp / (fp + tn))

    def test_crossing(self):
        self.assertAlmostEqual(cva.gaussian_crossing(0.5, 0.0, 1.0, 0.5, 2.0, 1.0), 1.0)          # symmetric -> midpoint
        x = cva.gaussian_crossing(0.8, 0.0, 1.0, 0.2, 2.0, 1.0)                                  # equal var: closed form
        self.assertAlmostEqual(x, 1.0 + np.log(0.8 / 0.2) / 2.0)
        self.assertIsNone(cva.gaussian_crossing(0.999999, 0.0, 1.0, 0.000001, 0.1, 1.0))       # crossing far outside (0, 0.1)

    def test_binary_and_fpr(self):
        pred = np.array([1, 1, 0, 0, 1], bool); y = np.array([1, 0, 1, 0, 0], bool)
        self.assertEqual(cva.confusion_binary(pred, y), (1, 2, 1, 1))
        m = cva.thresholded_metrics(1, 2, 1, 1)
        self.assertAlmostEqual(m["FPR"], 2 / 3)
        agg = cva.aggregate_confusions({"a": (1, 2, 1, 1), "b": (3, 0, 1, 4)})
        self.assertEqual((agg["pooled"]["tp"], agg["pooled"]["fp"], agg["pooled"]["fn"], agg["pooled"]["tn"]), (4, 2, 2, 5))


class TestBackMapping(unittest.TestCase):
    """map_to_original with missing pixels (CAA zeros): equals plain backward_map where no missing neighbour is read; uses only
    available neighbours otherwise (exact on a linear ramp); nearest fallback when both neighbours are missing.
    binary_to_original: nearest = ew pixel containing the original pixel centre."""

    def test_missing(self):
        from ew_resample import backward_map
        W, Wn = 133, 100
        ramp = np.tile(np.arange(Wn, dtype=np.float32) * 0.5, (3, 1))
        miss = np.zeros((3, Wn), bool); miss[0, 40:43] = True; miss[1, :] = False; miss[2, 60] = True
        out = cva.map_to_original(np.where(miss, 0, ramp), W, miss)
        ref = backward_map(ramp, W)
        np.testing.assert_allclose(out[1], ref[1], atol=1e-5)               # nothing missing
        used = backward_map(miss.astype(np.float32), W) > 0
        np.testing.assert_allclose(out[~used], ref[~used], atol=1e-5)
        self.assertTrue(np.isfinite(out).all())
        # a pixel reading one missing neighbour takes the other neighbour's value
        j = int(np.nonzero(used[2])[0][0])
        self.assertTrue(np.isclose(out[2, j], 59 * 0.5) or np.isclose(out[2, j], 61 * 0.5))
        # pixels inside the missing run 40-42 of row 0 get the nearest available value (39 or 43)
        inner = [jj for jj in range(W) if used[0, jj] and backward_map(miss.astype(np.float32), W)[0, jj] > 0.999]
        for jj in inner:
            self.assertIn(round(float(out[0, jj]) / 0.5), (39, 40, 41, 42, 43))   # nearest available: same col, other rows
        m2 = np.zeros((3, Wn), np.float32); m2[2] = 7.0
        miss2 = np.ones((3, Wn), bool); miss2[2] = False                    # rows 0-1 entirely missing
        np.testing.assert_allclose(cva.map_to_original(m2, W, miss2), 7.0)  # 2-D nearest -> row 2

    def test_binary(self):
        b = np.zeros((1, 4), np.uint8); b[0, 1] = 1
        out = cva.binary_to_original(b, 8)                                  # each ew pixel covers 2 original pixels
        np.testing.assert_array_equal(out[0], [0, 0, 1, 1, 0, 0, 0, 0])


class TestCVA(unittest.TestCase):
    def test_monotonic_bits(self):
        v = cva._VALUES[cva.POS_FINITE]
        self.assertTrue((np.diff(v) > 0).all())

    def test_auc_vs_sklearn(self):
        rng = np.random.default_rng(0)
        for n, prev in ((5000, 0.1), (20000, 0.5), (3000, 0.02)):
            y = rng.random(n) < prev
            s = np.abs(rng.normal(0.5 + 0.3 * y, 0.3)).round(2)          # many ties
            pos, neg = cva.histograms(s, y)
            self.assertAlmostEqual(cva.auroc(pos, neg), roc_auc_score(y, f16(s)), places=10)
            self.assertAlmostEqual(cva.auprc(pos, neg), average_precision_score(y, f16(s)), places=10)

    def test_auc_known(self):
        y = np.array([0, 0, 0, 1, 1], bool)
        p, n = cva.histograms(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), y)
        self.assertEqual(cva.auroc(p, n), 1.0); self.assertEqual(cva.auprc(p, n), 1.0)
        p, n = cva.histograms(np.array([0.9, 0.8, 0.7, 0.2, 0.1]), y)
        self.assertEqual(cva.auroc(p, n), 0.0)
        p, n = cva.histograms(np.full(5, 0.5), y)
        self.assertEqual(cva.auroc(p, n), 0.5); self.assertAlmostEqual(cva.auprc(p, n), 0.4)
        p, n = cva.histograms(np.array([0.1, 0.2]), np.array([0, 0], bool))
        self.assertTrue(np.isnan(cva.auroc(p, n)) and np.isnan(cva.auprc(p, n)))

    def test_thresholded_known(self):
        m = cva.thresholded_metrics(40, 10, 20, 30)
        self.assertAlmostEqual(m["OA"], 0.7); self.assertAlmostEqual(m["precision"], 0.8)
        self.assertAlmostEqual(m["recall"], 40 / 60); self.assertAlmostEqual(m["F1"], 80 / 110)
        self.assertAlmostEqual(m["kappa"], 0.4)                          # pe = (50*60 + 50*40) / 100^2 = 0.5

    def test_otsu(self):
        rng = np.random.default_rng(1)
        s = np.concatenate([np.abs(rng.normal(0.3, 0.05, 4000)), rng.normal(1.2, 0.1, 1000)])
        y = np.r_[np.zeros(4000, bool), np.ones(1000, bool)]
        pos, neg = cva.histograms(s, y)
        t = cva.otsu_bin(pos, neg)
        v = np.unique(f16(s))                                            # brute force over the unique values
        h = np.array([(f16(s) == x).sum() for x in v], float)
        best, arg = -1, None
        for i in range(len(v) - 1):
            w0, w1 = h[:i + 1].sum(), h[i + 1:].sum()
            m0, m1 = (h[:i + 1] * v[:i + 1]).sum() / w0, (h[i + 1:] * v[i + 1:]).sum() / w1
            b = w0 * w1 * (m0 - m1) ** 2
            if b > best + 1e-9 * abs(best):
                best, arg = b, v[i]
        self.assertEqual(cva._VALUES[t], arg)
        tp, fp, fn, tn = cva.confusion(pos, neg, t)
        self.assertEqual((tp, fp, fn, tn), (1000, 0, 0, 4000))

    def test_aggregate(self):
        rng = np.random.default_rng(2)
        scenes, allS, allY = {}, [], []
        for i, prev in enumerate((0.2, 0.05, 0.0)):
            y = rng.random(3000) < prev
            s = np.abs(rng.normal(0.4 + 0.4 * y, 0.25))
            scenes[f"s{i}"] = cva.histograms(s, y)
            allS.append(f16(s)); allY.append(y)
        r = cva.aggregate(scenes)
        S, Y = np.concatenate(allS), np.concatenate(allY)
        self.assertAlmostEqual(r["pooled"]["AUROC"], roc_auc_score(Y, S), places=10)
        self.assertAlmostEqual(r["pooled"]["AUPRC"], average_precision_score(Y, S), places=10)
        for k in ("tp", "fp", "fn", "tn"):
            self.assertEqual(r["pooled"][k], sum(r["per_scene"][s][k] for s in scenes))
        self.assertTrue(np.isnan(r["per_scene"]["s2"]["AUROC"]))
        self.assertAlmostEqual(r["mean_of_scenes"]["AUROC"], np.mean([r["per_scene"][s]["AUROC"] for s in ("s0", "s1")]))
        self.assertEqual(r["mean_of_scenes"]["n_scenes_defined"]["AUROC"], 2)

    def test_upsample(self):
        tok = np.arange(12, dtype=np.float32).reshape(3, 4)
        up = cva.upsample_tokens(tok, 20, 30, "nearest")
        self.assertEqual(up.shape, (20, 30))
        np.testing.assert_array_equal(up, np.repeat(np.repeat(tok, 8, 0), 8, 1)[:20, :30])
        gy, gx = np.mgrid[0:6, 0:7]
        tok = (0.3 * (8 * gy + 4) + 0.7 * (8 * gx + 4)).astype(np.float32)   # ramp sampled at patch centres
        up = cva.upsample_tokens(tok, 48, 56, "bilinear")
        y, x = np.mgrid[0:48, 0:56] + 0.5
        sl = (slice(4, 44), slice(4, 52))
        np.testing.assert_allclose(up[sl], (0.3 * y + 0.7 * x)[sl], atol=1e-4)
        t2 = tok.copy(); t2[2, 3] = np.nan                               # NaN token: ignored, weights renormalised
        up2 = cva.upsample_tokens(t2, 48, 56, "bilinear")
        self.assertTrue(np.isfinite(up2).all())
        far = np.ones((48, 56), bool); far[8:32, 16:40] = False             # pixels not using token (2, 3) unchanged
        np.testing.assert_allclose(up2[far], up[far], atol=1e-5)
        fy, fx = (20.5 - 4) / 8 - 2, (30.5 - 4) / 8 - 3                   # pixel (20, 30): tokens rows 2-3, cols 3-4
        w = {(2, 4): (1 - fy) * fx, (3, 3): fy * (1 - fx), (3, 4): fy * fx}  # (2, 3) is NaN -> dropped, renormalised
        exp = sum(wv * tok[k] for k, wv in w.items()) / sum(w.values())
        self.assertAlmostEqual(float(up2[20, 30]), exp, places=4)
        self.assertTrue(np.isnan(cva.upsample_tokens(np.full((2, 2), np.nan, np.float32), 16, 16, "bilinear")).all())


if __name__ == "__main__":
    unittest.main()
