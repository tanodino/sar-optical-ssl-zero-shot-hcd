"""
Code-Aligned Autoencoders (CAA) as a dedicated (per-scene, label-free) HCD method, run on one scene.

Method: Luppino et al., "Code-Aligned Autoencoders for Unsupervised Change Detection in Multimodal Remote Sensing
Images", IEEE TNNLS 35(1), 2024. Official code: github.com/llu025/Heterogeneous_CD, folder Code-Aligned_Autoencoders,
**commit ef0784f**, to be cloned unmodified into third_party/Heterogeneous_CD/ (not included here). This wrapper imports the official
modules (model `Kern_AceNet`, its training step / early stopping / weight saving, `get_config_kACE`, Otsu threshold,
dense-CRF filter) and replicates the official driver `Code_Aligned_Autoencoders.test()` with the following adaptations:

  1. input scaling: per scene and band, on valid pixels, clip to [mean - 3 std, mean + 3 std], map linearly to [-1, 1]
     (signed-data analogue of the official `_clip`; the networks end with tanh);
  2. no-data: invalid pixels = 0 after scaling; training patches drawn only where the whole
     patch is valid (same random crop + rot90 + up-down flip as the official generator); difference-image statistics,
     Otsu and CRF on valid pixels only (invalid = "no change" in the CRF unary);
  3. CPU only, inference on overlapping tiles (core + margin, even offsets because of the max-pool); the difference image
     is computed exactly as in the official `_difference_img` (per domain ||x_tilde - x_hat||_2, clipped at
     mean + 3 std, divided by the max; channel-weighted average), with the global statistics taken over the stitched
     valid pixels;
  4. dense CRF (official `_dense_gaussian_filtering`) on overlapping tiles; outputs raw + CRF-filtered maps; Otsu once
     on the stitched map (official `threshold_otsu`, 8-bit histogram as in `_change_map`);
  5. no per-epoch evaluation with labels (the official loop only logs it); labels are never read here;
  6. published network spec ([50, 50, 3], code size 3) and published training settings set explicitly (the official
     config switches to a debug network without GPU); save_images=False;
  7. seeds fixed (Python / NumPy / TensorFlow);

Training (official, unchanged): 4 stages x 50 epochs, 10 batches x 10 patches of 100 x 100, Adam + ExponentialDecay
(lr 1e-4), clipnorm 1, MSE losses (cycle, cross weighted by 1 - prior, reconstruction, kernel/affinity), L2 1e-6,
early stopping on the training loss (patience 10, best weights restored at the end); stage-1 cross weight = 0, then
1 - difference image of the previous stage.

Outputs (in --out_dir, refused if it exists): diff_raw.npy / diff_crf.npy (float16, H x W, 0 on invalid),
cm_raw.npy / cm_crf.npy (uint8 Otsu change maps, 0 on invalid), run.json (config, thresholds, timings, loss history),
tb/ (TensorBoard logs) and tb/weights/ (official best-weight checkpoints), and in --functional_test mode a test report.

Usage (TensorFlow environment, CPU):
    CUDA_VISIBLE_DEVICES= python caa_run.py \
        --scene data/scenes/EMSR832_AOI01 --out_dir caa/EMSR832_AOI01 --threads 16
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # CPU only (adaptation 3)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np                                          # noqa: E402
import tensorflow as tf                                     # noqa: E402

# Thread cap must be set before TensorFlow initialises its runtime (i.e. before the official modules are imported).
_THREADS = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 16
tf.config.threading.set_intra_op_parallelism_threads(_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(2)

CAA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "Heterogeneous_CD",
                       "Code-Aligned_Autoencoders")
sys.path.insert(0, os.path.abspath(CAA_DIR))
sys.dont_write_bytecode = True                              # the official folder is read-only
from Code_Aligned_Autoencoders import Kern_AceNet           # noqa: E402
from config import get_config_kACE                          # noqa: E402
from filtering import threshold_otsu, _dense_gaussian_filtering  # noqa: E402

CAA_COMMIT = "ef0784fba5ee26653e02f8b8f92683518c628b60"

# Published (GPU-branch) settings of the official config, set explicitly (adaptation 6).
PUBLISHED = dict(list_epochs=[50, 50, 50, 50], batches=10, batch_size=10, patch_size=100,
                 affinity_batch_size=500, affinity_patch_size=20, affinity_stride=5)
C_CODE = 3


# ----------------------------------------------------------------------------------------------- data
def scale_band(x, valid):
    """Adaptation 1: clip to mean +- 3 std of the valid pixels and map linearly to [-1, 1]; invalid -> 0 (adaptation 2)."""
    v = x[valid].astype(np.float64)
    lo, hi = v.mean() - 3 * v.std(), v.mean() + 3 * v.std()
    out = 2.0 * (np.clip(x, lo, hi) - lo) / (hi - lo) - 1.0
    out[~valid] = 0.0
    return out.astype(np.float32), dict(mean=float(v.mean()), std=float(v.std()), lo=float(lo), hi=float(hi))


def load_scene(scene, crop=None):
    """Return x (H, W, 4) optical t1, y (H, W, 2) SAR t2 in [-1, 1], valid (H, W) bool, scaling stats."""
    opt = np.load(os.path.join(scene, "opt.npy"), mmap_mode="r")      # (4, H, W) float16, normalised (prepare_scenes.py)
    sar = np.load(os.path.join(scene, "sar.npy"), mmap_mode="r")      # (2, H, W)
    valid = np.load(os.path.join(scene, "valid.npy"), mmap_mode="r")  # (H, W) uint8, image no-data / AOI mask
    sl = (slice(None), slice(None)) if crop is None else (slice(crop[0], crop[0] + crop[2]),
                                                         slice(crop[1], crop[1] + crop[3]))
    valid = np.asarray(valid[sl]).astype(bool)
    stats = {"opt": [], "sar": []}
    xs, ys = [], []
    for c in range(opt.shape[0]):
        b, s = scale_band(np.asarray(opt[(c,) + sl], np.float32), valid); xs.append(b); stats["opt"].append(s)
    for c in range(sar.shape[0]):
        b, s = scale_band(np.asarray(sar[(c,) + sl], np.float32), valid); ys.append(b); stats["sar"].append(s)
    return np.stack(xs, -1), np.stack(ys, -1), valid, stats


class ValidPatchSampler:
    """Random patch_size x patch_size crops whose pixels are all valid (adaptation 2), with the official augmentation
    (random rot90 + random up-down flip). Uniform over the fully valid positions (rejection sampling)."""

    def __init__(self, valid, patch_size, rng):
        self.ps, self.rng = patch_size, rng
        ii = np.zeros((valid.shape[0] + 1, valid.shape[1] + 1), np.int64)
        ii[1:, 1:] = valid.astype(np.int64).cumsum(0).cumsum(1)      # integral image of the validity mask
        p = patch_size
        self.full = (ii[p:, p:] - ii[:-p, p:] - ii[p:, :-p] + ii[:-p, :-p]) == p * p   # top-left positions OK
        self.frac = float(self.full.mean())
        if self.frac == 0:
            raise ValueError("no fully valid patch position")

    def sample(self):
        H, W = self.full.shape
        while True:
            r, c = int(self.rng.integers(H)), int(self.rng.integers(W))
            if self.full[r, c]:
                return r, c


def training_dataset(x, y, p, sampler):
    """tf.data pipeline equivalent to the official `_training_data_generator`, restricted to valid patches."""
    ps = sampler.ps
    cx, cy = x.shape[-1], y.shape[-1]

    def gen():
        while True:
            r, c = sampler.sample()
            tmp = np.concatenate([x[r:r + ps, c:c + ps], y[r:r + ps, c:c + ps], p[r:r + ps, c:c + ps]], -1)
            tmp = np.rot90(tmp, int(sampler.rng.integers(4)))
            if sampler.rng.random() < 0.5:
                tmp = tmp[::-1]
            tmp = np.ascontiguousarray(tmp)
            yield tmp[..., :cx], tmp[..., cx:cx + cy], tmp[..., cx + cy:]

    sig = (tf.TensorSpec([ps, ps, cx], tf.float32), tf.TensorSpec([ps, ps, cy], tf.float32),
           tf.TensorSpec([ps, ps, 1], tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------------------------- inference
def tiles(n, core, margin):
    """1-D tiling: list of (window_start, window_end, core_start, core_end) with even window starts."""
    assert core % 2 == 0 and margin % 2 == 0
    out = []
    for s in range(0, n, core):
        e = min(n, s + core)
        ws, we = max(0, s - margin), min(n, e + margin)
        out.append((ws, we, s, e))
    return out


def raw_domain_norms(cd, x, y, core, margin):
    """Per-pixel ||x_tilde - x_hat||_2 and ||y_tilde - y_hat||_2 (official `_difference_img` before the global
    normalisation), computed on overlapping tiles with the official networks in inference mode."""
    H, W = x.shape[:2]
    dx = np.zeros((H, W), np.float32)
    dy = np.zeros((H, W), np.float32)

    @tf.function(reduce_retracing=True)
    def net(xt, yt):
        xc, yc = cd._enc_x(xt, False), cd._enc_y(yt, False)
        x_tilde, y_tilde = cd._dec_x(xc, False), cd._dec_y(yc, False)
        x_hat, y_hat = cd._dec_x(yc, False), cd._dec_y(xc, False)
        return tf.norm(x_tilde - x_hat, ord=2, axis=-1), tf.norm(y_tilde - y_hat, ord=2, axis=-1)

    for (rs, re, r0, r1) in tiles(H, core, margin):
        for (cs, ce, c0, c1) in tiles(W, core, margin):
            xt, yt = x[rs:re, cs:ce], y[rs:re, cs:ce]
            ph, pw = (re - rs) % 2, (ce - cs) % 2                  # odd window (scene border): reflect-pad 1 px
            if ph or pw:
                xt = np.pad(xt, ((0, ph), (0, pw), (0, 0)), mode="reflect")
                yt = np.pad(yt, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            a, b = net(tf.constant(xt[None]), tf.constant(yt[None]))
            a, b = a.numpy()[0], b.numpy()[0]
            dx[r0:r1, c0:c1] = a[r0 - rs:r1 - rs, c0 - cs:c1 - cs]
            dy[r0:r1, c0:c1] = b[r0 - rs:r1 - rs, c0 - cs:c1 - cs]
    return dx, dy


def normalise_domain(d, valid, bandwidth=3.0):
    """Official `_domain_difference_img` normalisation, statistics over valid pixels only (adaptation 2)."""
    v = d[valid].astype(np.float64)
    thr = v.mean() + bandwidth * v.std()
    d = np.minimum(d, thr)
    d = d / d[valid].max()
    d[~valid] = 0.0
    return d.astype(np.float32)


def difference_image(cd, x, y, valid, core, margin):
    """Official `_difference_img`: channel-weighted average of the two normalised domain differences."""
    dx, dy = raw_domain_norms(cd, x, y, core, margin)
    cx, cy = x.shape[-1], y.shape[-1]
    d = (cy * normalise_domain(dx, valid) + cx * normalise_domain(dy, valid)) / (cx + cy)
    d[~valid] = 0.0
    return d


def crf_filter(x, y, d, core, margin):
    """Official dense-CRF filter (`_dense_gaussian_filtering`) on overlapping tiles (adaptation 4)."""
    H, W = d.shape
    out = np.zeros_like(d)
    for (rs, re, r0, r1) in tiles(H, core, margin):
        for (cs, ce, c0, c1) in tiles(W, core, margin):
            q = _dense_gaussian_filtering(x[None, rs:re, cs:ce], y[None, rs:re, cs:ce], d[None, rs:re, cs:ce, None])
            out[r0:r1, c0:c1] = q.numpy()[0, r0 - rs:r1 - rs, c0 - cs:c1 - cs, 0]
    return out


def otsu_map(d, valid):
    """Official `_change_map` (8-bit Otsu, change = d >= threshold), histogram on valid pixels only."""
    thr = float(threshold_otsu(tf.constant((d[valid] * 255).astype(np.int32)))) / 255.0
    return ((d >= thr) & valid).astype(np.uint8), thr


# ----------------------------------------------------------------------------------------------- main
def build_model(out_dir, cx, cy, overrides):
    config = get_config_kACE("GeoFlood", debug=False)   # the name only sets the official log folder; settings unaffected
    config.update(PUBLISHED)
    config.update(crop_factor=24 / PUBLISHED["patch_size"])   # official GPU branch
    config.update(overrides)
    config.update(epochs=sum(config["list_epochs"]), save_images=False, logdir=os.path.join(out_dir, "tb"))
    if config["epochs"] > 20:
        config.update(evaluation_frequency=config["epochs"] // 10)
    spec = {"enc_X": {"input_chs": cx, "filter_spec": [50, 50, C_CODE]},
            "enc_Y": {"input_chs": cy, "filter_spec": [50, 50, C_CODE]},
            "dec_X": {"input_chs": C_CODE, "filter_spec": [50, 50, cx]},
            "dec_Y": {"input_chs": C_CODE, "filter_spec": [50, 50, cy]}}
    return Kern_AceNet(spec, **config), config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tile_core", type=int, default=2048)
    ap.add_argument("--tile_margin", type=int, default=64)
    ap.add_argument("--crf_core", type=int, default=4096)
    ap.add_argument("--crf_margin", type=int, default=128)
    ap.add_argument("--crop", type=int, nargs=4, default=None, help="r0 c0 h w (functional test only)")
    ap.add_argument("--functional_test", action="store_true",
                    help="tiny training budget + tiling checks against whole-image results; NOT a result")
    args = ap.parse_args()
    print(json.dumps(vars(args)), flush=True)
    os.makedirs(args.out_dir)                                     # refuses an existing folder

    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    assert args.threads == _THREADS
    assert not tf.config.list_physical_devices("GPU"), "CAA runs on CPU in this wrapper"

    t0 = time.time()
    x, y, valid, stats = load_scene(args.scene, args.crop)
    print(f"scene {x.shape[:2]}, valid {valid.mean():.3f}, load {time.time() - t0:.1f}s", flush=True)
    overrides = dict(list_epochs=[2, 2], batches=2) if args.functional_test else {}
    cd, config = build_model(args.out_dir, x.shape[-1], y.shape[-1], overrides)
    rng = np.random.default_rng(args.seed)
    sampler = ValidPatchSampler(valid, config["patch_size"], rng)

    # ---- training: replicates Code_Aligned_Autoencoders.test() (stages, prior update), without label evaluation
    timings, prior_thr = {}, []
    clw = np.zeros(x.shape[:2] + (1,), np.float32)                 # stage-1 cross-loss weight = 0 (official)
    t_train = time.time()
    for stage, epochs in enumerate(config["list_epochs"]):
        config.update(epochs=epochs)
        ds = training_dataset(x, y, clw, sampler)
        cd.train(ds, evaluation_dataset=None, **config)
        alpha = difference_image(cd, x, y, valid, args.tile_core, args.tile_margin)
        clw = (1.0 - alpha)[..., None]
        print(f"stage {stage + 1}/{len(config['list_epochs'])} done, epoch {int(cd.epoch.numpy())}, "
              f"{time.time() - t_train:.0f}s", flush=True)
    timings["train_s"] = time.time() - t_train
    cd.load_all_weights(cd.log_path)                               # official: restore best weights

    # ---- final maps
    t1 = time.time()
    d_raw = difference_image(cd, x, y, valid, args.tile_core, args.tile_margin)
    timings["final_inference_s"] = time.time() - t1
    t2 = time.time()
    d_crf = crf_filter(x, y, d_raw, args.crf_core, args.crf_margin)
    d_crf[~valid] = 0.0
    timings["crf_s"] = time.time() - t2
    cm_raw, thr_raw = otsu_map(d_raw, valid)
    cm_crf, thr_crf = otsu_map(d_crf, valid)
    for name, arr in (("diff_raw", d_raw.astype(np.float16)), ("diff_crf", d_crf.astype(np.float16)),
                      ("cm_raw", cm_raw), ("cm_crf", cm_crf)):
        np.save(os.path.join(args.out_dir, name + ".npy"), arr)

    report = dict(args=vars(args), caa_commit=CAA_COMMIT, scaling=stats, valid_frac=float(valid.mean()),
                  fully_valid_patch_frac=sampler.frac, config={k: v for k, v in config.items()
                                                                if isinstance(v, (int, float, str, list, bool))},
                  epochs_run=int(cd.epoch.numpy()), loss_history=[float(v) for v in cd.metrics_history["total"]],
                  otsu_threshold=dict(raw=thr_raw, crf=thr_crf), change_frac=dict(raw=float(cm_raw[valid].mean()),
                                                                                   crf=float(cm_crf[valid].mean())),
                  timings=timings)

    if args.functional_test:
        # (a) tiled network inference == official whole-image __call__ (crop must be fully valid for equality)
        ref = cd([tf.constant(x[None]), tf.constant(y[None])]).numpy()[0, ..., 0]
        report["test_tiled_vs_official_call_maxabs"] = float(np.abs(ref - d_raw).max()) if valid.all() else None
        # (b) tiled vs whole-image network inference with our function (any crop)
        d_whole = difference_image(cd, x, y, valid, core=10 ** 6, margin=0)
        report["test_tiled_vs_whole_maxabs"] = float(np.abs(d_whole - d_raw).max())
        # (c) tiled vs whole-image CRF
        c_whole = crf_filter(x, y, d_raw, core=10 ** 6, margin=0); c_whole[~valid] = 0
        diff = np.abs(c_whole - d_crf)
        report["test_crf_tiled_vs_whole"] = dict(maxabs=float(diff.max()), mean=float(diff[valid].mean()),
                                                 binary_disagreement=float(
                                                     ((c_whole >= thr_crf) != (d_crf >= thr_crf))[valid].mean()))
    with open(os.path.join(args.out_dir, "run.json"), "x") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in report if k.startswith("test") or k in
                      ("timings", "otsu_threshold", "change_frac", "epochs_run")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
