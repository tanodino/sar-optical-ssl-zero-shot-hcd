"""
Unit tests of cdrl_run.py helpers (CDRL), CPU only.
  1. majority_labels == the code's sample_selection (scipy generic_filter + classify_patch from third_party/CDRL) on random masks
     (adaptation 6: vectorised, numerically identical).
  2. otsu_valid on fully valid data == cv2 Otsu on the whole uint8 image (the code's change_map).
  3. gather_patches == the code's ReflectionPad2d + Unfold(stride 1) patches at the same pixels.
  4. tiles_1d covers every pixel exactly once by the tile cores.
Run: python -m unittest tests/test_cdrl_run.py -v   (from the project root)
"""
import os
import sys
import unittest

import cv2
import numpy as np
import torch
from scipy.ndimage import generic_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import cdrl_run as cr  # noqa: E402
    from Fuction_Library import classify_patch  # noqa: E402  (official CDRL code, third_party/CDRL)
except ImportError:                                            # CDRL code not installed in third_party/CDRL
    cr = None


@unittest.skipIf(cr is None, "third_party/CDRL not available")
class TestCDRL(unittest.TestCase):
    def test_majority(self):
        rng = np.random.default_rng(0)
        for p in (0.1, 0.5, 0.9):
            m = (rng.random((61, 47)) < p).astype(np.float64)
            m[10:30, 5:25] = 1
            ref = generic_filter(m, classify_patch, size=7)
            np.testing.assert_array_equal(cr.majority_labels(m.astype(np.uint8), 7, 0.8), ref.astype(np.uint8))

    def test_otsu(self):
        rng = np.random.default_rng(1)
        d = np.clip(np.concatenate([rng.normal(0.2, 0.05, 3000), rng.normal(0.7, 0.1, 1000)]), 0, 1).reshape(40, 100)
        got, t = cr.otsu_valid(d, np.ones_like(d, bool))
        _, ref = cv2.threshold((torch.from_numpy(d) * 255).type(torch.uint8).numpy(), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        np.testing.assert_array_equal(got, ref > 0)

    def test_gather(self):
        rng = np.random.default_rng(2)
        C, H, W, ps = 3, 13, 17, 7
        x = rng.standard_normal((C, H, W)).astype(np.float32)
        pad = np.pad(x, ((0, 0), (3, 3), (3, 3)), mode="reflect")
        sub = torch.nn.Sequential(torch.nn.ReflectionPad2d(ps // 2), torch.nn.Unfold(kernel_size=ps, stride=1))
        ref = sub(torch.from_numpy(x)[None])[0].T.reshape(H * W, C, ps, ps).numpy()
        idx = rng.choice(H * W, 40, replace=False)
        np.testing.assert_allclose(cr.gather_patches(pad, idx, W, ps), ref[idx], rtol=0, atol=0)

    def test_tiles(self):
        for n in (1, 100, 1024, 2500):
            cov = np.zeros(n, int)
            for ws, we, s, e in cr.tiles_1d(n, 1024, 64):
                assert ws <= s < e <= we
                cov[s:e] += 1
            self.assertTrue((cov == 1).all())


if __name__ == "__main__":
    unittest.main()
