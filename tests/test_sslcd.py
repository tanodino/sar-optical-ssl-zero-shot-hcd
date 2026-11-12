"""
Unit tests of the sslcd package. CPU only, tiny random tensors.

Run from the working folder:
    python -m unittest discover -s tests -v

What is tested
--------------
* losses against straightforward reference implementations written independently in this file
  (e.g. the multi-positive NT-Xent formula of the CroCoMAE paper, explicit loops for Barlow Twins);
* masking, patch ordering, learning-rate schedule, weight-decay groups;
* augmentations: shape, no clipping of the value range, per-sample randomness;
* every method: one forward/backward pass with a tiny ViT, finite loss, gradients reach both encoders;
* the engine end to end: output files, folder protection, strict reload of the encoders, and that
  "run 6 steps" == "run 3 steps, stop, --resume to 6 steps" (same final weights on CPU).
"""
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain import build_parser                                   # noqa: E402
from sslcd import losses                                            # noqa: E402
from sslcd.collapse import collapse_stats                           # noqa: E402
from sslcd.augment import Augmenter, random_brightness_contrast, random_solarize   # noqa: E402
from sslcd.engine import lr_at_step, optical_aug_constants, param_groups, train   # noqa: E402
from sslcd.methods import METHODS, build_method                     # noqa: E402
from sslcd.methods.base import Arch                                 # noqa: E402
from sslcd.vit import PatchEmbed, ViTEncoder, patchify, random_masking   # noqa: E402

TINY = Arch(img_size=16, patch_size=4, embed_dim=32, depth=2, num_heads=2, grad_checkpointing=False)
SMALL_PROJ = {"decur": (64, 64, 64)}


# ---------------------------------------------------------------------------------------------------------
# Reference implementations (independent of the package code)
# ---------------------------------------------------------------------------------------------------------
def ref_infonce_multi_positive(u, labels, tau):
    """CroCoMAE paper formula: -1/|U| sum_i 1/|P_i| sum_{j in P_i} log( exp(u_i.u_j/tau) / sum_{k!=i} exp(u_i.u_k/tau) )."""
    u = F.normalize(u.double(), dim=1)
    n = u.shape[0]
    total = 0.0
    for i in range(n):
        denom = sum(math.exp(float(u[i] @ u[k]) / tau) for k in range(n) if k != i)
        pos = [j for j in range(n) if j != i and labels[j] == labels[i]]
        total += sum(-math.log(math.exp(float(u[i] @ u[j]) / tau) / denom) for j in pos) / len(pos)
    return total / n


def ref_barlow(za, zb, lambd):
    za = (za.double() - za.double().mean(0)) / za.double().std(0, unbiased=False)
    zb = (zb.double() - zb.double().mean(0)) / zb.double().std(0, unbiased=False)
    c = za.t() @ zb / za.shape[0]
    d = c.shape[0]
    on = sum((1 - c[i, i]) ** 2 for i in range(d))
    off = sum(c[i, j] ** 2 for i in range(d) for j in range(d) if i != j)
    return float(on + lambd * off)


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_nt_xent_matches_paper_formula(self):
        za, zb = torch.randn(6, 5), torch.randn(6, 5)
        labels = list(range(6)) * 2
        for tau in (1.0, 0.1):
            ref = ref_infonce_multi_positive(torch.cat([za, zb]), labels, tau)
            self.assertAlmostEqual(float(losses.nt_xent(za, zb, tau)), ref, places=4)

    def test_nt_xent_prefers_aligned_pairs(self):
        za = torch.randn(16, 8)
        self.assertLess(float(losses.nt_xent(za, za + 0.01 * torch.randn(16, 8), 0.1)),
                        float(losses.nt_xent(za, torch.randn(16, 8), 0.1)))

    def test_barlow_twins_matches_reference(self):
        za, zb = torch.randn(32, 6), torch.randn(32, 6)
        self.assertAlmostEqual(float(losses.barlow_twins_loss(za, zb, 0.0051)), ref_barlow(za, zb, 0.0051), places=3)

    def test_decur_common_unique(self):
        # Identical inputs: every diagonal entry of C is 1 -> common on-diag term 0, unique on-diag term = n_unique.
        z = torch.randn(256, 8)
        lc, lu = losses.decur_cross_losses(z, z.clone(), dim_common=6, lambd=0.0)
        self.assertAlmostEqual(float(lc), 0.0, places=3)
        self.assertAlmostEqual(float(lu), 2.0, places=3)

    def test_clip_loss(self):
        za, zb = torch.randn(8, 4), torch.randn(8, 4)
        scale = torch.tensor(math.log(1 / 0.07))
        logits = F.normalize(za, dim=1) @ F.normalize(zb, dim=1).t() / 0.07
        t = torch.arange(8)
        ref = 0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.t(), t))
        self.assertAlmostEqual(float(losses.clip_loss(za, zb, scale)), float(ref), places=4)

    def test_mae_loss_only_masked_patches(self):
        imgs = torch.randn(2, 3, 8, 8)
        target = patchify(imgs, 4)                                         # (2, 4, 48)
        norm_t = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6) ** 0.5
        mask = torch.tensor([[1., 0., 0., 1.], [0., 1., 0., 0.]])
        pred = norm_t.clone()
        pred[mask == 0] += 5.0                                             # errors on VISIBLE patches only
        self.assertAlmostEqual(float(losses.mae_loss(pred, target, mask)), 0.0, places=5)
        pred2 = norm_t.clone()
        pred2[0, 0] += 1.0                                                 # error of 1 on one masked patch
        self.assertAlmostEqual(float(losses.mae_loss(pred2, target, mask)), 1.0 / 3.0, places=5)


class TestVitAndSchedule(unittest.TestCase):
    def test_random_masking(self):
        g = torch.Generator().manual_seed(0)
        ids_keep, ids_restore, mask = random_masking(5, 16, 0.75, "cpu", generator=g)
        self.assertEqual(tuple(ids_keep.shape), (5, 4))
        self.assertTrue(torch.all(mask.sum(1) == 12))
        self.assertTrue(torch.all(torch.gather(mask, 1, ids_keep) == 0))   # kept patches are not masked

    def test_patchify_order_matches_patch_embed(self):
        torch.manual_seed(0)
        pe = PatchEmbed(8, 4, 3, 5)
        x = torch.randn(2, 3, 8, 8)
        w = pe.proj.weight.reshape(5, -1)                                  # (D, C*p*p), same layout as unfold
        ref = patchify(x, 4) @ w.t() + pe.proj.bias
        self.assertTrue(torch.allclose(pe(x), ref, atol=1e-5))

    def test_encoder_shapes_and_masked_positions(self):
        enc = ViTEncoder(img_size=16, patch_size=4, in_chans=2, embed_dim=32, depth=2, num_heads=2)
        x = torch.randn(3, 2, 16, 16)
        self.assertEqual(tuple(enc(x).shape), (3, 17, 32))
        ids_keep = torch.arange(16).repeat(3, 1)                           # "mask" that keeps everything
        self.assertTrue(torch.allclose(enc.eval()(x), enc.forward_masked(x, ids_keep), atol=1e-5))

    def test_lr_schedule(self):
        total, warm, peak, mn = 1000, 50, 1.5e-4, 1e-6
        lrs = [lr_at_step(s, total, warm, peak, mn) for s in range(total)]
        self.assertAlmostEqual(lrs[0], peak / warm)
        self.assertAlmostEqual(lrs[warm - 1], peak)
        self.assertAlmostEqual(lrs[warm], peak)                            # cosine starts at the peak
        self.assertTrue(all(a >= b for a, b in zip(lrs[warm:], lrs[warm + 1:])))   # no restart
        self.assertLess(abs(lrs[-1] - mn), 1e-8)

    def test_weight_decay_groups(self):
        model = build_method("croma", TINY, 2, 4)
        decay_ids = {id(p) for p in param_groups(model, 0.05)[0]["params"]}
        named = dict(model.named_parameters())
        for name in ("enc_sar.pos_embed", "enc_sar.cls_token", "logit_scale", "enc_sar.norm.weight",
                     "enc_sar.patch_embed.proj.bias", "decoder.mask_token"):
            self.assertNotIn(id(named[name]), decay_ids, name)
        for name in ("enc_sar.patch_embed.proj.weight", "proj_sar.weight"):
            self.assertIn(id(named[name]), decay_ids, name)


class TestAugment(unittest.TestCase):
    def test_no_clipping_and_shape(self):
        torch.manual_seed(0)
        x = torch.full((8, 2, 16, 16), -5.0)                               # far outside [0, 1]
        y = Augmenter.sar()(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x, atol=1e-4))                   # constant image stays constant

    def test_per_sample_randomness(self):
        torch.manual_seed(0)
        x = torch.randn(1, 4, 16, 16).repeat(32, 1, 1, 1)                  # 32 identical images
        y = Augmenter.optical()(x)
        diffs = (y - y[:1]).abs().flatten(1).max(1).values
        self.assertGreater(int((diffs > 1e-4).sum()), 25)                  # most samples differ from sample 0

    def test_brightness_zero_point(self):
        z0 = torch.tensor([-1.0, 2.0])
        x = z0.view(1, 2, 1, 1).expand(4, 2, 8, 8).clone()
        # an image equal to the physical zero is unchanged by multiplicative brightness and by contrast
        self.assertTrue(torch.allclose(random_brightness_contrast(x, 1.0, zero_point=z0), x, atol=1e-5))

    def test_solarize_per_band_hand_computed(self):
        # per-band threshold t_c and max m_c; x' = m_c - x where x >= t_c, else x (p = 1: always applied).
        # band 0: t = 0.5, m = 1.0 (= DeCUR on [0, 1]);  band 1: t = 2.0, m = 3.0 (1.0 < t unchanged; 2.0 -> 1.0; 4.0 -> -1.0)
        x = torch.tensor([[[[0.2, 0.5, 0.9]], [[1.0, 2.0, 4.0]]]])          # (1, 2, 1, 3)
        expected = torch.tensor([[[[0.2, 0.5, 0.1]], [[1.0, 1.0, -1.0]]]])
        y = random_solarize(x, 1.0, [0.5, 2.0], [1.0, 3.0])
        self.assertTrue(torch.allclose(y, expected))
        self.assertTrue(torch.equal(random_solarize(x, 0.0, [0.5, 2.0], [1.0, 3.0]), x))   # p = 0: unchanged
        # scalar threshold still works (same value for all bands)
        self.assertTrue(torch.allclose(random_solarize(x[:, :1], 1.0, 0.5, 1.0), expected[:, :1]))
        # middle-of-range reflection, as built from norm_stats: lo_n -> hi_n, hi_n -> lo_n, middle fixed
        lo_n, hi_n = -0.9, 6.8
        t, m = (lo_n + hi_n) / 2, lo_n + hi_n
        z = torch.tensor([[[[t, hi_n]]]])
        self.assertTrue(torch.allclose(random_solarize(z, 1.0, [t], [m]), torch.tensor([[[[t, lo_n]]]]), atol=1e-6))

    def test_solarize_probability_and_switch(self):
        torch.manual_seed(0)
        self.assertEqual(Augmenter.optical().solarize_p, 0.0)                         # no constants: disabled
        aug = Augmenter.optical(zero_point=[0, 0, 0, 0], solarize_threshold=[1, 1, 1, 1], solarize_max=[2, 2, 2, 2])
        self.assertEqual(aug.solarize_p, 0.2)                                          # solarize probability
        with self.assertRaises(ValueError):
            Augmenter.optical(solarize_threshold=[1, 1, 1, 1])                         # max missing
        x = torch.full((4000, 1, 1, 1), 5.0)                                           # every pixel above threshold
        frac = (random_solarize(x, 0.2, [1.0], [2.0]) != x).float().mean().item()      # per-sample decision
        self.assertAlmostEqual(frac, 0.2, delta=0.03)

    def test_optical_aug_constants_from_norm_stats(self):
        ns = {"s2_bands": ["B02", "B03"], "optical_augmentation": {
            "B02": {"zero_point": -2.0, "solarize_threshold": 3.0, "solarize_max": 6.0},
            "B03": {"zero_point": -1.5, "solarize_threshold": 2.0, "solarize_max": 4.0}}}
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "norm_stats.json")
            with open(p, "w") as f:
                json.dump(ns, f)
            c = optical_aug_constants(p, 2)
            self.assertEqual(c, dict(opt_zero_point=[-2.0, -1.5], opt_solarize_threshold=[3.0, 2.0],
                                     opt_solarize_max=[6.0, 4.0]))
            with self.assertRaises(ValueError):
                optical_aug_constants(p, 4)                                            # band count mismatch
            m = build_method("decur", TINY, 2, 2, projector=SMALL_PROJ["decur"], **c)
            self.assertEqual(m.aug_opt.solarize_threshold, [3.0, 2.0])
            self.assertEqual(m.aug_opt.solarize_p, 0.2)
        finally:
            shutil.rmtree(d)
            self.assertEqual(optical_aug_constants(None, 4)["opt_solarize_threshold"], None)


class TestCollapseStats(unittest.TestCase):
    """Collapse statistics distinguish a large constant offset from a real collapse (added 2026-11-06)."""

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.noise = torch.randn(400, 32, generator=g)
        self.offset = torch.full((1, 32), 100.0 / math.sqrt(32))         # constant vector of norm 100

    def test_healthy(self):
        s = collapse_stats(self.noise)
        self.assertLess(abs(s["cos_mean"]), 0.05)
        self.assertLess(abs(s["cos_centred_mean"]), 0.05)
        self.assertLess(s["offset_to_spread"], 0.2)
        self.assertGreater(s["eff_rank"], 20)

    def test_large_offset_is_not_collapse(self):
        s = collapse_stats(self.noise + self.offset)
        self.assertGreater(s["cos_mean"], 0.9)                           # raw cosine looks "collapsed" ...
        self.assertLess(abs(s["cos_centred_mean"]), 0.05)                 # ... centred cosine does not
        # ||mean|| = 100 + O(1); mean distance to the mean ~ sqrt(32) = 5.66 -> ratio ~ 17.7
        self.assertAlmostEqual(s["offset_to_spread"], 100.0 / math.sqrt(32), delta=1.5)
        self.assertGreater(s["eff_rank"], 20)                             # information content unchanged
        self.assertAlmostEqual(s["eff_rank"], collapse_stats(self.noise)["eff_rank"], places=6)

    def test_real_collapse(self):
        # every sample = the same vector + tiny variation along ONE direction
        x = self.offset + 1e-3 * self.noise[:, :1] * F.one_hot(torch.tensor(0), 32).float()
        s = collapse_stats(x)
        self.assertGreater(s["cos_mean"], 0.99)
        self.assertLess(s["std_mean"], 1e-3)
        self.assertLess(s["eff_rank"], 1.5)

    def test_centred_cosine_hand_computed(self):
        # 2 samples symmetric around a mean: centred vectors are opposite -> centred cosine = -1
        x = torch.tensor([[10.0, 1.0], [10.0, -1.0]])
        s = collapse_stats(x)
        self.assertAlmostEqual(s["cos_centred_mean"], -1.0, places=6)
        self.assertAlmostEqual(s["offset_to_spread"], 10.0, places=6)    # ||(10, 0)|| / mean(||(0, +-1)||)
        self.assertAlmostEqual(s["cos_mean"], (100.0 - 1.0) / 101.0, places=6)


class TestMethods(unittest.TestCase):
    def test_forward_backward_all_methods(self):
        for name in sorted(METHODS):
            with self.subTest(method=name):
                torch.manual_seed(0)
                kw = {"projector": SMALL_PROJ[name]} if name in SMALL_PROJ else {}
                model = build_method(name, TINY, 2, 4, **kw)
                sar, opt = torch.randn(8, 2, 16, 16), torch.randn(8, 4, 16, 16)
                loss, logs = model(sar, opt)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for enc in (model.enc_sar, model.enc_opt):
                    g = enc.patch_embed.proj.weight.grad
                    self.assertTrue(g is not None and g.abs().sum() > 0)
                self.assertIn(model.global_repr, ("cls", "avg"))

    def test_global_repr_choices(self):
        self.assertEqual(METHODS["croma"].global_repr, "avg")    # CROMA: never the CLS token (it has no trained CLS token)
        for n in ("crocomae", "decur"):
            self.assertEqual(METHODS[n].global_repr, "cls")


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        np.save(os.path.join(self.tmp, "SAR.npy"), rng.standard_normal((40, 2, 16, 16)).astype(np.float32))
        np.save(os.path.join(self.tmp, "MS.npy"), rng.standard_normal((40, 4, 16, 16)).astype(np.float32))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def args(self, method, out, steps=6, resume=False):
        a = ["--method", method, "--sar", os.path.join(self.tmp, "SAR.npy"), "--opt", os.path.join(self.tmp, "MS.npy"),
             "--out_dir", out, "--steps", str(steps), "--batch_size", "8", "--patch_size", "4", "--embed_dim", "32",
             "--depth", "2", "--num_heads", "2", "--device", "cpu", "--num_workers", "0", "--log_every", "2",
             "--ckpt_every", "3", "--collapse_images", "20", "--seed", "1", "--test_without_norm_stats"]
        if method == "decur":
            a += ["--projector_override", "64-64-64"]
        if resume:
            a.append("--resume")
        return build_parser().parse_args(a)

    def test_norm_stats_mandatory(self):
        out = os.path.join(self.tmp, "no_norm_stats")
        a = self.args("crocomae", out)
        a.test_without_norm_stats = False                      # neither --norm_stats nor the test flag
        with self.assertRaises(ValueError):
            train(a)
        self.assertFalse(os.path.exists(out))                  # refused before creating anything on disk
        a.test_without_norm_stats, a.norm_stats = True, "x.json"   # both given: also refused
        with self.assertRaises(ValueError):
            train(a)

    def test_end_to_end_resume_and_protection(self):
        for method in ("crocomae", "decur"):
            with self.subTest(method=method):
                out_a = os.path.join(self.tmp, f"{method}_a")
                train(self.args(method, out_a))
                for f in ("config.json", "train_log.csv", "resume.pth", "encoder_sar.pth", "encoder_opt.pth",
                          "model_info.json", "collapse_check.json"):
                    self.assertTrue(os.path.exists(os.path.join(out_a, f)), f)
                # a non-empty folder is refused without --resume
                with self.assertRaises(FileExistsError):
                    train(self.args(method, out_a))
                # strict reload of the saved encoder
                with open(os.path.join(out_a, "model_info.json")) as f:
                    info = json.load(f)
                enc = ViTEncoder(in_chans=info["sar_channels"], dropout=0.0,
                                 **{k: info["arch"][k] for k in ("img_size", "patch_size", "embed_dim", "depth",
                                                                  "num_heads")})
                missing, unexpected = enc.load_state_dict(torch.load(os.path.join(out_a, "encoder_sar.pth")),
                                                          strict=False)
                self.assertFalse(missing or unexpected)
                # resume: copy the run folder as it was after step 3 (only resume.pth + config), continue to 6
                out_b = os.path.join(self.tmp, f"{method}_b")
                os.makedirs(out_b)
                shutil.copy(os.path.join(out_a, "resume.pth"), out_b)
                shutil.copy(os.path.join(out_a, "config.json"), out_b)
                train(self.args(method, out_b, resume=True))
                wa = torch.load(os.path.join(out_a, "encoder_opt.pth"))
                wb = torch.load(os.path.join(out_b, "encoder_opt.pth"))
                for k in wa:
                    self.assertTrue(torch.allclose(wa[k], wb[k], atol=1e-6), k)


if __name__ == "__main__":
    unittest.main()
