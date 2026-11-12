"""
Unit tests of extract_features.py, CPU only.

  1. reflect_index == numpy 'reflect' padding.
  2. Mode A shifted-inference aggregation on a synthetic scene (token score = fraction of changed pixels in the 8x8
     patch) == brute-force reference (np.pad reflect, block means, expand, crop, average over offsets) for strides
     8, 4, 2, 1; stride 8 = the block map of the unshifted grid; token map of offset (0, 0) = block map.
  3. Mode B on a linear ramp (window score = window mean): the bilinear map equals the ramp exactly on interior pixels
     (window centre placement and interpolation geometry), strides 8 and 16.
  4. Every method loads STRICTLY from its resume.pth; for 2 random inputs both modes return the expected variants with
     finite scores; full-context methods: Mode B enc_cls score == CVA of the encoder CLS computed directly; masked
     methods: identical results for identical generator seeds, different for different seeds.

Run: python -m unittest tests/test_extract_features.py -v   (from the project root)
"""
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import extract_features as ef                                           # noqa: E402

PRE = os.environ.get("HETEROCD_WEIGHTS", os.path.join(ROOT, "pretrain"))   # folder with crocomae/ decur/ croma/ (released weights)


def frac_score(xs, xo, gen):
    """Token score = mean |opt0 - sar0| over the 8x8 patch: (B, C, 64, 64) -> {'full': (B, 64)}."""
    d = (xo[:, 0] - xs[:, 0]).abs().float()
    B = d.shape[0]
    return {"full": d.view(B, 8, 8, 8, 8).mean((2, 4)).reshape(B, 64)}


def brute_mode_a(change, stride):
    H, W = change.shape
    acc, n = np.zeros((H, W)), 0
    for dy in range(0, 8, stride):
        for dx in range(0, 8, stride):
            Hp, Wp = -(-(H + dy) // 64) * 64, -(-(W + dx) // 64) * 64
            Xp = np.pad(change, ((dy, Hp - H - dy), (dx, Wp - W - dx)), mode="reflect")
            blk = Xp.reshape(Hp // 8, 8, Wp // 8, 8).mean((1, 3))
            full = np.repeat(np.repeat(blk, 8, 0), 8, 1)
            acc += full[dy:dy + H, dx:dx + W]
            n += 1
    return acc / n


class TestGeometry(unittest.TestCase):
    def test_reflect_index(self):
        n = 11
        for pad in (1, 5, 10):
            ref = np.pad(np.arange(n), (pad, pad), mode="reflect")
            got = ef.reflect_index(np.arange(-pad, n + pad), n)
            np.testing.assert_array_equal(got, ref)

    def test_mode_a_vs_bruteforce(self):
        rng = np.random.default_rng(0)
        H, W = 100, 130
        change = np.zeros((H, W), np.float32)
        change[20:47, 33:90] = 1
        change[70:75, 5:9] = 1
        change[rng.random((H, W)) < 0.02] = 1
        sar = np.zeros((2, H, W), np.float32)
        opt = np.zeros((4, H, W), np.float32)
        opt[0] = change
        valid = np.ones((H, W), bool)
        for s in (8, 4, 2, 1):
            maps, tok0, art, n, nt = ef.mode_a(frac_score, sar, opt, valid, s, 7, "cpu")
            ref = brute_mode_a(change, s)
            np.testing.assert_allclose(maps["full"].numpy(), ref, atol=1e-5, err_msg=f"stride {s}")
            if s == 8:
                blk = np.pad(change, ((0, 28), (0, 62)), mode="reflect").reshape(16, 8, 24, 8).mean((1, 3))
                np.testing.assert_allclose(tok0["full"].astype(np.float32), blk[:13, :17], atol=1e-3)
                self.assertEqual(n, 2 * 3)                                  # 128 x 192 padded -> 2 x 3 tiles

    def test_mode_b_ramp(self):
        H, W = 160, 200
        y, x = np.mgrid[0:H, 0:W].astype(np.float64)
        ramp = (0.37 * x + 0.81 * y).astype(np.float32)
        sar = np.zeros((2, H, W), np.float32)
        opt = np.zeros((4, H, W), np.float32)
        opt[0] = ramp
        mean_score = lambda xs, xo, gen: {"m": (xo[:, 0] - xs[:, 0]).double().mean((1, 2))}  # noqa: E731
        for s in (8, 16):
            maps, grid, n, nt = ef.mode_b(mean_score, sar, opt, H, W, s, 50, "cpu")
            self.assertEqual(grid["m"].shape, (-(-H // s), -(-W // s)))
            m = maps["m"].numpy()
            i0, i1 = 32 + s, None
            sl = (slice(32 + s, H - 32 - s), slice(32 + s, W - 32 - s))
            np.testing.assert_allclose(m[sl], ramp[sl], atol=2e-3, err_msg=f"stride {s}")


class TestSkip(unittest.TestCase):
    """Input skip: maps on needed pixels identical with / without skip (noise-dependent scores, batch sizes differ),
    NaN outside `need`, fewer inputs; needed_mask makes backward_map finite on valid original pixels."""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.H, self.W = 150, 210
        self.sar = rng.standard_normal((2, self.H, self.W)).astype(np.float32)
        self.opt = rng.standard_normal((4, self.H, self.W)).astype(np.float32)
        v = np.zeros((self.H, self.W), bool)
        v[10:40, 20:60] = True; v[100:103, 150:200] = True; v[149, 0] = True
        self.valid, self.need = v, ef.needed_mask(v, None, 2)

    def test_mode_a_skip(self):
        def sf(xs, xo, nz):                                      # depends on content AND on the per-input noise
            d = (xo[:, 0] - xs[:, 0]).float()
            B = d.shape[0]
            return {"full": d.view(B, 8, 8, 8, 8).mean((2, 4)).reshape(B, 64) + nz[:, 0, :]}
        for s in (8, 2):
            nf = lambda o, r, n: ef.row_noise("cpu", 3, n, 7, 0, s, 3, o[0], o[1], r)   # noqa: E731
            a = ef.mode_a(sf, self.sar, self.opt, self.valid, s, 5, "cpu", nf, self.need, skip=False)
            b = ef.mode_a(sf, self.sar, self.opt, self.valid, s, 13, "cpu", nf, self.need, skip=True)
            ma, mb = a[0]["full"].numpy(), b[0]["full"].numpy()
            np.testing.assert_array_equal(ma[self.need], mb[self.need])
            self.assertTrue(np.isnan(mb[~self.need]).all() and np.isnan(ma[~self.need]).all())
            self.assertLess(b[3], a[3]); self.assertEqual(a[4], b[4]); self.assertEqual(a[3], a[4])
            self.assertAlmostEqual(a[2]["full"]["edge_minus_centre"], b[2]["full"]["edge_minus_centre"], places=10)

    def test_mode_b_skip(self):
        sf = lambda xs, xo, nz: {"m": (xo[:, 0] - xs[:, 0]).double().mean((1, 2)) + nz[:, 0, 0]}  # noqa: E731
        for s in (8, 16):
            nf = lambda r, n: ef.row_noise("cpu", 2, n, 7, 1, s, 2, r)   # noqa: E731
            a = ef.mode_b(sf, self.sar, self.opt, self.H, self.W, s, 9, "cpu", nf, self.need, skip=False)
            b = ef.mode_b(sf, self.sar, self.opt, self.H, self.W, s, 31, "cpu", nf, self.need, skip=True)
            ma, mb = a[0]["m"].numpy(), b[0]["m"].numpy()
            np.testing.assert_array_equal(ma[self.need], mb[self.need])
            self.assertTrue(np.isnan(mb[~self.need]).all())
            self.assertLess(b[2], a[2])

    def test_need_ew(self):
        from ew_resample import backward_map, forward_valid
        rng = np.random.default_rng(2)
        H, W, Wn = 40, 133, 100
        ov = rng.random((H, W)) < 0.05                           # sparse, isolated valid original pixels
        ov[5:15, 30:70] = True
        vew = forward_valid(ov, Wn)
        need = ef.needed_mask(vew, ov, 2)
        m = np.where(need, 1.0, np.nan).astype(np.float32)       # NaN outside need, as in the extracted maps
        back = backward_map(m, W)
        self.assertTrue(np.isfinite(back[ov]).all())
        need0 = ef.needed_mask(vew, None, 0)                     # ew validity alone would not be enough
        self.assertFalse(np.isfinite(backward_map(np.where(need0, 1.0, np.nan).astype(np.float32), W)[ov]).all())


class TestMedian(unittest.TestCase):
    """estimate_median: with n_sample >= all candidates, equals the brute-force lower median over (A) all fully valid
    8x8 blocks, (B) all windows whose centre pixel is valid; sub-sampling is seeded (same seed -> same median)."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.H, self.W = 141, 203
        self.sar = rng.standard_normal((2, self.H, self.W)).astype(np.float32)
        self.opt = rng.standard_normal((4, self.H, self.W)).astype(np.float32)
        v = rng.random((self.H, self.W)) > 0.03
        v[50:90, 100:180] = False
        self.valid = v

    @staticmethod
    def lower_median(x):
        return np.sort(x, axis=0)[(len(x) - 1) // 2]

    def test_mode_a(self):
        def dfn(xs, xo, nz):                                     # token vector = block means of opt0, opt1
            B = xo.shape[0]
            return {"t": xo[:, :2].reshape(B, 2, 8, 8, 8, 8).mean((3, 5)).reshape(B, 2, 64).transpose(1, 2).double()}
        med, n_in, n_vec = ef.estimate_median(dfn, self.sar, self.opt, self.valid, "A", 2, 10 ** 6, 7, 0)
        H8, W8 = self.H // 8, self.W // 8
        blk_ok = self.valid[:H8 * 8, :W8 * 8].reshape(H8, 8, W8, 8).all((1, 3))
        means = self.opt[:2, :H8 * 8, :W8 * 8].astype(np.float64).reshape(2, H8, 8, W8, 8).mean((2, 4))
        ref = self.lower_median(means[:, blk_ok].T)
        self.assertEqual(n_vec, int(blk_ok.sum()))
        np.testing.assert_allclose(med["t"].numpy(), ref, rtol=1e-6, atol=1e-6)
        a, _, _ = ef.estimate_median(dfn, self.sar, self.opt, self.valid, "A", 2, 5, 7, 0)
        b, _, _ = ef.estimate_median(dfn, self.sar, self.opt, self.valid, "A", 2, 5, 3, 0)
        torch.testing.assert_close(a["t"], b["t"])

    def test_mode_b(self):
        dfn = lambda xs, xo, nz: {"w": torch.stack([(xo[:, 0] - xs[:, 0]).double().mean((1, 2)),  # noqa: E731
                                                    xo[:, 1].double().mean((1, 2))], 1)}
        for s in (8, 16):
            med, n_in, _ = ef.estimate_median(dfn, self.sar, self.opt, self.valid, "B", s, 10 ** 6, 50, 0)
            pad = lambda x: np.pad(x.astype(np.float64), 64, mode="reflect")   # noqa: E731
            d0, o1 = pad(self.opt[0] - self.sar[0]), pad(self.opt[1])
            vals = []
            for r in range(-(-self.H // s)):
                for c in range(-(-self.W // s)):
                    cy, cx = min(s * r + s // 2, self.H - 1), min(s * c + s // 2, self.W - 1)
                    if self.valid[cy, cx]:
                        r0, c0 = s * r + s // 2 - 32 + 64, s * c + s // 2 - 32 + 64
                        vals.append([d0[r0:r0 + 64, c0:c0 + 64].mean(), o1[r0:r0 + 64, c0:c0 + 64].mean()])
            self.assertEqual(n_in, len(vals))
            np.testing.assert_allclose(med["w"].numpy(), self.lower_median(np.array(vals)), rtol=1e-6, atol=1e-6)


@unittest.skipUnless(all(os.path.exists(os.path.join(PRE, m, "model_info.json")) for m in ("crocomae", "decur", "croma")),
                     "pre-trained weights not found (set HETEROCD_WEIGHTS)")
class TestMethods(unittest.TestCase):
    def test_all_methods(self):
        torch.manual_seed(0)
        xs, xo = torch.randn(2, 2, 64, 64), torch.randn(2, 4, 64, 64)
        for m in ("crocomae", "decur", "croma"):
            with self.subTest(method=m):
                ex = ef.Extractor(os.path.join(PRE, m), "cpu")
                self.assertEqual(ex.step, 148000)
                for mode in ("A", "B"):
                    out = ex.scores(xs, xo, mode, 2, torch.rand(2, 2, 64, generator=ef.make_gen("cpu", 0, 1)))
                    self.assertEqual(sorted(out), sorted(ex.variants(mode)))
                    for v, s in out.items():
                        self.assertEqual(tuple(s.shape), (2, 64) if mode == "A" else (2,))
                        self.assertTrue(torch.isfinite(s).all())
                        self.assertTrue(((s >= 0) & (s <= 2.0001)).all())
                    # bias correction: plain scores unchanged, v_bc = || d - med || computed directly
                    nz = torch.rand(2, 2, 64, generator=ef.make_gen("cpu", 0, 1))
                    d = ex.diffs(xs, xo, mode, 2, nz)
                    med = {v: torch.randn(x.shape[-1]) * 0.1 for v, x in d.items()}
                    o2 = ex.scores(xs, xo, mode, 2, nz, med)
                    self.assertEqual(sorted(o2), sorted(list(d) + [v + "_bc" for v in d]))
                    for v in d:
                        torch.testing.assert_close(o2[v], out[v])
                        torch.testing.assert_close(o2[v + "_bc"], (d[v] - med[v]).norm(dim=-1))
                if m in ef.MASKED:
                    nz = lambda k: torch.rand(2, 2, 64, generator=ef.make_gen("cpu", k))  # noqa: E731
                    a = ex.scores(xs, xo, "B", 2, nz(5))
                    b = ex.scores(xs, xo, "B", 2, nz(5))
                    c = ex.scores(xs, xo, "B", 2, nz(6))
                    v = ex.variants("B")[0]
                    self.assertTrue(torch.equal(a[v], b[v]))
                    self.assertFalse(torch.equal(a[v], c[v]))
                else:
                    with torch.no_grad():
                        cs, co = ex.model.enc_sar(xs)[:, 0], ex.model.enc_opt(xo)[:, 0]
                    out = ex.scores(xs, xo, "B", 1, None)
                    torch.testing.assert_close(out["enc_cls"], ef.cva(cs, co), atol=1e-5, rtol=1e-5)
                del ex


if __name__ == "__main__":
    unittest.main()
