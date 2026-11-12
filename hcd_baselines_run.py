"""
Dedicated HCD methods SCCN and cGAN, run per scene (label-free), on CPU.

Methods (reimplementations by Luppino et al. in the pinned CAA repository, NOT the original authors' code):
  * SCCN  — Liu et al., "A Deep Convolutional Coupling Network for Change Detection Based on Heterogeneous Optical and Radar
            Images", IEEE TNNLS 29(3), 2018 (doi:10.1109/TNNLS.2016.2636227); code `SCCN.py`.
  * cGAN  — Niu et al., "A Conditional Adversarial Network for Change Detection in Heterogeneous Images", IEEE GRSL 16(1),
            2019 (doi:10.1109/LGRS.2018.2868704); code `cGAN.py`.
  Repository github.com/llu025/Heterogeneous_CD at commit ef0784f, to be cloned unmodified into third_party/Heterogeneous_CD.

The official drivers SCCN.test() / cGAN.test() are replicated here (they end with an undefined `final_kappa`) with the
following adaptations:
  1. input scaling as CAA (caa_run.py: per scene / band clip to mean +- 3 std of the valid pixels, map to [-1, 1]);
  2. no-data as CAA (invalid = 0; training samples fully valid; difference-image statistics, Otsu and CRF on valid pixels only);
  3. (not used for the EMS benchmark: east-west resampled scenes are brought back to the original grid by cva.caa_scene);
  4. fixed seeds; CPU only with a thread cap;
  5. SCCN: instead of the whole image (batch 1), each epoch trains on random fully valid crops (no augmentation, as the official
     driver) totalling ~0.25 M px: one 512 x 512 crop, or k crops of the largest available size p (k = round(512^2 / p^2)) when
     no 512 crop is fully valid; pre-training 200 epochs and training 200 epochs (official GPU-branch values);
  6. cGAN: instead of all non-overlapping 5 x 5 patches, each epoch uses 10,000 fully valid patches drawn at random from the same
     non-overlapping 5 x 5 grid (250,000 px), official batch 200 (-> 50 batches), lr 1e-5, Lambda 5, 200 epochs;
  7. the unchanged-pixel prior Pu (official: recomputed from the full-image change map after epoch 10, used from epoch 12 on) is
     computed from the change map OF THE TRAINING SAMPLES ONLY, with the current model, from epoch 12 on;
     cGAN: --cgan_prior {code, unchanged} (the paper uses `code`): `code` = as the code, Pu = round(mean(change map over the patch))
     (= 1 on CHANGED patches); `unchanged` = 1 - that (as SCCN's Pu = 1 - change map);
  8. tiled inference over the whole scene; official difference image (SCCN: ||code_x - code_y|| ; cGAN: ||G(y) - A(x)||, both with
     the official clip at mean + 3 std and division by the max, statistics on valid pixels); official final filter = dense CRF
     (`decorated_gaussian_filter` -> `_dense_gaussian_filtering`, tiled as for CAA); official 8-bit Otsu change maps.
No label is read anywhere. No early stopping exists for these two methods in the code (base criterion returns False).

Outputs (in --out_dir, refused if it exists): diff_raw.npy / diff_crf.npy (float16), cm_raw.npy / cm_crf.npy (uint8), run.json
(same names as CAA, so cva.caa_scene evaluates them), tb/ (TensorBoard logs).

Usage (TensorFlow environment, CPU):
    CUDA_VISIBLE_DEVICES= python hcd_baselines_run.py --method sccn \
        --scene data/scenes/EMSR836_AOI05 --out_dir hcd_baselines/sccn/EMSR836_AOI05 --threads 16
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import caa_run as cr                                         # noqa: E402  (thread cap, data / tiling / CRF / Otsu helpers)
import numpy as np                                           # noqa: E402
import tensorflow as tf                                      # noqa: E402
from SCCN import SCCN                                        # noqa: E402  (pinned repo, on sys.path via caa_run)
from cGAN import CGAN                                        # noqa: E402
from config import get_config_SCCN, get_config_CGAN          # noqa: E402

GPU_BRANCH = dict(list_epochs=[50, 50, 50, 50], batches=10, batch_size=10, patch_size=100,
                  affinity_batch_size=500, affinity_patch_size=20, affinity_stride=5)   # official get_config GPU branch
SCCN_CROP, SCCN_MIN_CROP = 512, 64
CGAN_PATCHES_PER_EPOCH = 10000
PRIOR_FROM_EPOCH = 12                                        # official: updated at the end of epoch index 11 (> 10)


# ----------------------------------------------------------------------------------------------- shared
def norm_clip_max(d, valid):
    """Official `_domain_difference_img` normalisation (clip at mean + 3 std, divide by max), valid pixels only."""
    return cr.normalise_domain(d, valid)


def otsu_cm_on_samples(d):
    """Official `_change_map` (8-bit Otsu, change = d >= t) on a 1-D array of difference values in [0, 1]."""
    t = float(cr.threshold_otsu(tf.constant((d * 255).astype(np.int32)))) / 255.0
    return d >= t


def config_for(method, out_dir, functional_test):
    cfg = (get_config_SCCN if method == "sccn" else get_config_CGAN)("GeoFlood", debug=False)   # name only used for the log folder
    cfg.update(GPU_BRANCH)                                   # CPU would select the debug branch: set GPU values explicitly
    if method == "sccn":
        cfg.update(Lambda=0.2, preepochs=200, batches=1, batch_size=1, C_CODE=20, evaluation_frequency=5, learning_rate=1e-3)
    else:
        cfg.update(Lambda=5.0, patch_size=5, learning_rate=1e-5, batch_size=200)
    cfg.update(epochs=sum(cfg["list_epochs"]), save_images=False, logdir=os.path.join(out_dir, "tb"))
    if functional_test:
        cfg.update(epochs=14, preepochs=2)
    return cfg


# ----------------------------------------------------------------------------------------------- SCCN
class CropSampler:
    """Fully valid square crops: size 512 if any 512 crop is fully valid, else the largest p (multiple of 32, >= 64)."""

    def __init__(self, valid, rng):
        self.rng = rng
        for p in range(SCCN_CROP, SCCN_MIN_CROP - 1, -32):
            try:
                self.s = cr.ValidPatchSampler(valid, p, rng)
                self.p = p
                break
            except ValueError:
                continue
        else:
            raise ValueError("no fully valid crop of >= 64 px")
        self.k = int(round(SCCN_CROP ** 2 / self.p ** 2))       # crops per epoch, ~0.25 M px in total

    def crops(self):
        return [self.s.sample() for _ in range(self.k)]


def sccn_diff_tiled(cd, x, y, valid, core, margin):
    """||code_x - code_y||_2 per pixel (official SCCN __call__ before normalisation), tiled; then official normalisation."""
    H, W = valid.shape
    d = np.zeros((H, W), np.float32)

    @tf.function(reduce_retracing=True)
    def net(xt, yt):
        return tf.norm(cd._enc_x(xt, False) - cd._enc_y(yt, False), ord=2, axis=-1)

    for (rs, re, r0, r1) in cr.tiles(H, core, margin):
        for (cs, ce, c0, c1) in cr.tiles(W, core, margin):
            a = net(tf.constant(x[None, rs:re, cs:ce]), tf.constant(y[None, rs:re, cs:ce])).numpy()[0]
            d[r0:r1, c0:c1] = a[r0 - rs:r1 - rs, c0 - cs:c1 - cs]
    return norm_clip_max(d, valid)


def run_sccn(cfg, x, y, valid, rng, args, log):
    cx, cy, C = x.shape[-1], y.shape[-1], cfg["C_CODE"]
    spec = {"enc_X": {"input_chs": cx, "filter_spec": [C, C, C, C]}, "enc_Y": {"input_chs": cy, "filter_spec": [C, C, C, C]},
            "dec_X": {"input_chs": C, "filter_spec": [cx]}, "dec_Y": {"input_chs": C, "filter_spec": [cy]}}   # GPU branch
    cd = SCCN(spec, **cfg)
    cs = CropSampler(valid, rng)
    log.update(crop_size=cs.p, crops_per_epoch=cs.k)
    p = cs.p

    def ds_from(crops, pu=None):
        xs = np.stack([x[r:r + p, c:c + p] for r, c in crops])
        ys = np.stack([y[r:r + p, c:c + p] for r, c in crops])
        pus = np.ones(xs.shape[:3] + (1,), np.float32) if pu is None else pu
        return tf.data.Dataset.from_tensor_slices((xs, ys, pus))

    # pre-training (official: cd.pretrain(EVALUATE, preepochs, batches=1, batch_size=1)); one new crop set per epoch
    t = time.time()
    for e in range(cfg["preepochs"]):
        cd.pretrain(ds_from(cs.crops()), preepochs=1, batches=cs.k, batch_size=1)
    log["pretrain_s"] = time.time() - t
    # training (official: epochs x cd.train(TRAIN, epochs=1); Pu = 1 - change map after epoch 10)
    t = time.time()
    for e in range(cfg["epochs"]):
        crops = cs.crops()
        pu = None
        if e >= PRIOR_FROM_EPOCH:                              # adaptation 7: change map of the training crops, current model
            xs = np.stack([x[r:r + p, c:c + p] for r, c in crops]); ys = np.stack([y[r:r + p, c:c + p] for r, c in crops])
            raw = tf.norm(cd._enc_x(xs, False) - cd._enc_y(ys, False), ord=2, axis=-1).numpy()
            d = norm_clip_max(raw.reshape(-1, p), np.ones(raw.reshape(-1, p).shape, bool)).reshape(raw.shape)
            pu = (1.0 - otsu_cm_on_samples(d.ravel()).reshape(raw.shape).astype(np.float32))[..., None]
        cd.train(ds_from(crops, pu), epochs=1, batches=cs.k, batch_size=1, evaluation_dataset=None)
    log["train_s"] = time.time() - t
    return lambda: sccn_diff_tiled(cd, x, y, valid, args.tile_core, args.tile_margin), cd


# ----------------------------------------------------------------------------------------------- cGAN
def cgan_grid(valid, ps):
    """Fully valid cells of the non-overlapping ps x ps grid anchored at (0, 0): array of (row, col) top-left positions."""
    H, W = (valid.shape[0] // ps) * ps, (valid.shape[1] // ps) * ps
    ok = valid[:H, :W].reshape(H // ps, ps, W // ps, ps).all(axis=(1, 3))
    return np.argwhere(ok) * ps


def cgan_diff_pixels(cd, xp, yp):
    """Per-pixel ||G(y) - A(x)||_2 on (n, ps, ps, c) patches (official __call__ before normalisation); batches of 200."""
    out = []
    for i in range(0, len(xp), 4000):
        g = cd._gen(tf.constant(yp[i:i + 4000]), False)
        a = cd._approx(tf.constant(xp[i:i + 4000]), False)
        out.append(tf.norm(g - a, ord=2, axis=-1).numpy())
    return np.concatenate(out)


def cgan_diff_tiled(cd, x, y, valid, ps, rows=1000):
    """Generator / approximator are per-pixel dense networks: evaluate the whole scene in row blocks, pixels grouped into
    ps x ps 'patches' only to match the network's reshape; then the official normalisation."""
    H, W = valid.shape
    d = np.zeros((H, W), np.float32)
    for r0 in range(0, H, rows):
        xb, yb = x[r0:r0 + rows].reshape(-1, x.shape[-1]), y[r0:r0 + rows].reshape(-1, y.shape[-1])
        n = len(xb)
        pad = (-n) % (ps * ps)
        xb = np.concatenate([xb, np.zeros((pad, xb.shape[1]), np.float32)]).reshape(-1, ps, ps, x.shape[-1])
        yb = np.concatenate([yb, np.zeros((pad, yb.shape[1]), np.float32)]).reshape(-1, ps, ps, y.shape[-1])
        d[r0:r0 + rows] = cgan_diff_pixels(cd, xb, yb).reshape(-1)[:n].reshape(-1, W)
    return norm_clip_max(d, valid)


def run_cgan(cfg, x, y, valid, rng, args, log):
    ps, cx, cy = cfg["patch_size"], x.shape[-1], y.shape[-1]
    spec = {"Generator": {"shapes": [ps, cy], "filter_spec": [25, 100, 500, 100, cx]},
            "Approximator": {"shapes": [ps, cx], "filter_spec": [25, 100, 500, 100, cx]},
            "Discriminator": {"shapes": [ps, cx], "filter_spec": [25, 100, 200, 50, 1]}}      # GPU branch
    cd = CGAN(spec, **cfg)
    grid = cgan_grid(valid, ps)
    n = min(CGAN_PATCHES_PER_EPOCH, len(grid))
    bs = cfg["batch_size"]
    batches = n // bs + (n % bs != 0)
    log.update(valid_grid_patches=int(len(grid)), patches_per_epoch=int(n), batches_per_epoch=int(batches))
    t = time.time()
    for e in range(cfg["epochs"]):
        sel = grid[rng.choice(len(grid), n, replace=False)]
        xp = np.stack([x[r:r + ps, c:c + ps] for r, c in sel]); yp = np.stack([y[r:r + ps, c:c + ps] for r, c in sel])
        pu = np.ones(n, np.float32)
        if e >= PRIOR_FROM_EPOCH:                              # adaptation 7: change map of the sampled patches
            raw = cgan_diff_pixels(cd, xp, yp)
            d = norm_clip_max(raw.reshape(n, -1), np.ones((n, ps * ps), bool))
            cm = otsu_cm_on_samples(d.ravel()).reshape(n, -1)
            pu = np.round(cm.mean(1)).astype(np.float32)          # official: round(mean(change map over the patch))
            if args.cgan_prior == "unchanged":
                pu = 1.0 - pu
        ds = tf.data.Dataset.from_tensor_slices((xp, yp, pu))
        cd.train(ds, epochs=1, batches=batches, batch_size=bs, evaluation_dataset=None)
    log["train_s"] = time.time() - t
    return lambda: cgan_diff_tiled(cd, x, y, valid, ps), cd


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["sccn", "cgan"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tile_core", type=int, default=2048)
    ap.add_argument("--tile_margin", type=int, default=4)
    ap.add_argument("--crf_core", type=int, default=4096)
    ap.add_argument("--crf_margin", type=int, default=128)
    ap.add_argument("--cgan_prior", choices=["code", "unchanged"], default=None)
    ap.add_argument("--crop", type=int, nargs=4, default=None, help="r0 c0 h w (functional test only)")
    ap.add_argument("--functional_test", action="store_true", help="tiny budget + tiling checks; NOT a result")
    args = ap.parse_args()
    if args.method == "cgan" and args.cgan_prior is None:
        ap.error("--cgan_prior is required for cgan (the paper uses code)")
    print(json.dumps(vars(args)), flush=True)
    os.makedirs(args.out_dir)
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    assert not tf.config.list_physical_devices("GPU"), "runs on CPU in this wrapper"
    t0 = time.time()
    x, y, valid, stats = cr.load_scene(args.scene, args.crop)
    print(f"scene {x.shape[:2]}, valid {valid.mean():.3f}, load {time.time() - t0:.1f}s", flush=True)
    cfg = config_for(args.method, args.out_dir, args.functional_test)
    rng = np.random.default_rng(args.seed)
    log = {}
    diff_fn, cd = (run_sccn if args.method == "sccn" else run_cgan)(cfg, x, y, valid, rng, args, log)
    t = time.time()
    d_raw = diff_fn()
    d_raw[~valid] = 0.0
    log["final_inference_s"] = time.time() - t
    t = time.time()
    d_crf = cr.crf_filter(x, y, d_raw, args.crf_core, args.crf_margin)
    d_crf[~valid] = 0.0
    log["crf_s"] = time.time() - t
    cm_raw, thr_raw = cr.otsu_map(d_raw, valid)
    cm_crf, thr_crf = cr.otsu_map(d_crf, valid)
    for name, arr in (("diff_raw", d_raw.astype(np.float16)), ("diff_crf", d_crf.astype(np.float16)),
                      ("cm_raw", cm_raw), ("cm_crf", cm_crf)):
        np.save(os.path.join(args.out_dir, name + ".npy"), arr)
    report = dict(args=vars(args), method=args.method, repo_commit=cr.CAA_COMMIT, scaling=stats, valid_frac=float(valid.mean()),
                  config={k: v for k, v in cfg.items() if isinstance(v, (int, float, str, list, bool))},
                  epochs_run=int(cd.epoch.numpy()), otsu_threshold=dict(raw=thr_raw, crf=thr_crf),
                  change_frac=dict(raw=float(cm_raw[valid].mean()), crf=float(cm_crf[valid].mean())),
                  timings=dict(log, total_s=time.time() - t0))
    if args.functional_test:                                   # tiled == whole-image inference (our functions)
        if args.method == "sccn":
            whole = sccn_diff_tiled(cd, x, y, valid, core=10 ** 6, margin=0)
        else:
            whole = cgan_diff_tiled(cd, x, y, valid, cfg["patch_size"], rows=10 ** 6)
        whole[~valid] = 0.0
        report["test_tiled_vs_whole_maxabs"] = float(np.abs(whole - d_raw).max())
    with open(os.path.join(args.out_dir, "run.json"), "x") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in report if k.startswith("test") or k in
                      ("timings", "otsu_threshold", "change_frac", "epochs_run")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
