"""
CDRL as a dedicated HCD method, run per scene (label-free), on one GPU.

Method: Zhan, Lan, Zhu, Dang, Gong, "Cross-Domain Difference Representation Learning for Unsupervised Heterogeneous Change
Detection", IEEE TGRS 63, 2025 (doi:10.1109/TGRS.2025.3584073). Official code github.com/OMEGA-RS/CDRL (MIT), to be cloned unmodified into
third_party/CDRL at commit b1c755bddeda6585e200de06db468522bf6ce131; its networks (Model_Establishment.Trans_Net / CD_Net) are
imported unchanged. This wrapper replicates CDRL_Main.py with the following adaptations:
  1. input scaling as CAA: per scene / band clip to mean +- 3 std of the valid pixels, map to [-1, 1]; invalid -> 0;
  2. no-data: stage-1 training patches and stage-3 training samples fully valid; Otsu thresholds and the DI min-max on valid pixels;
  3. stage-1 inference on ~1024 x 1024 tiles (overlap --overlap px each side, centres kept), each tile processed EXACTLY as the code
     processes a whole image (its own style codes and AdaIN instance statistics); raw DI stitched, then min-max over the valid pixels
     of the scene; seam statistic reported;
  4. stage-1 training: per epoch 600 random fully valid 64 x 64 patches (~1 M-px image at the code's stride 40), random h / v flips,
     code batch 32, Adam lr 1e-4 (betas 0.5, 0.9, wd 5e-4), 2 iterations x 10 epochs, mask initialised at random (code);
  5. mask update after each iteration: Otsu (8-bit, as the code's cv2 call) of the scene-wide DI over the valid pixels;
  6. stage 2: Gaussian filter sigma 1.5, Otsu, 7 x 7 majority (>= int(0.8 * 49) = 39 pixels equal to the centre) computed with a
     uniform filter (numerically identical to the code's scipy generic_filter, unit-tested); invalid pixels excluded;
  7. stage 3 (CDNet): per epoch at most 50,000 reliable changed + as many reliable unchanged samples (code balance; all changed when
     fewer), fully valid 7 x 7 patches gathered on demand from the reflection-padded scene (identical values to the code's Unfold),
     random h / v flips, batch 256, Adam lr 1e-3, 18 epochs; prediction on ALL uncertain valid pixels;
  8. outputs: diff_raw.npy (stage-1 DI, main continuous map), diff_gauss.npy (Gaussian-filtered DI), cm_raw.npy (Otsu of the DI),
     cm_final.npy (CDRL's final binary map); 9. patch size 7, seed 282 (code's setup_seed); 10. one GPU.
No label is read anywhere.

Usage (main environment, one GPU):
    CUDA_VISIBLE_DEVICES=0 python cdrl_run.py \
        --scene data/scenes/EMSR836_AOI05 --out_dir hcd_baselines/cdrl/EMSR836_AOI05
"""
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, minimum_filter, uniform_filter

CDRL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "CDRL")
sys.path.insert(0, os.path.abspath(CDRL_DIR))
sys.dont_write_bytecode = True                                # the official folder is read-only
from Model_Establishment import Trans_Net, CD_Net            # noqa: E402

CDRL_COMMIT = "b1c755bddeda6585e200de06db468522bf6ce131"
CFG = dict(batch_size=32, patch_size=64, lr=1e-4, max_epoch=10, iter_num=2, patches_per_epoch=600,
           cd_batch_size=256, cd_patch_size=7, cd_lr=1e-3, cd_max_epoch=18, cd_cap_changed=50000, seed=282,
           majority_alpha=0.8, gauss_sigma=1.5)


# ----------------------------------------------------------------------------------------------- data
def setup_seed(seed):
    """As the code's Fuction_Library.setup_seed."""
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def scale_band(x, valid):
    """Decision 1 (as caa_run.scale_band): clip to mean +- 3 std of the valid pixels, map to [-1, 1]; invalid -> 0."""
    v = x[valid].astype(np.float64)
    lo, hi = v.mean() - 3 * v.std(), v.mean() + 3 * v.std()
    out = 2.0 * (np.clip(x, lo, hi) - lo) / (hi - lo) - 1.0
    out[~valid] = 0.0
    return out.astype(np.float32), dict(mean=float(v.mean()), std=float(v.std()), lo=float(lo), hi=float(hi))


def load_scene(scene, crop=None):
    """-> xa (Ca, H, W) optical t1, xb (Cb, H, W) SAR t2 (float32 in [-1, 1]), valid (H, W) bool, stats."""
    opt = np.load(os.path.join(scene, "opt.npy"), mmap_mode="r")
    sar = np.load(os.path.join(scene, "sar.npy"), mmap_mode="r")
    valid = np.load(os.path.join(scene, "valid.npy"), mmap_mode="r")
    sl = (slice(None), slice(None)) if crop is None else (slice(crop[0], crop[0] + crop[2]), slice(crop[1], crop[1] + crop[3]))
    valid = np.asarray(valid[sl]).astype(bool)
    stats, out = {"opt": [], "sar": []}, {"opt": [], "sar": []}
    for name, arr in (("opt", opt), ("sar", sar)):
        for c in range(arr.shape[0]):
            b, s = scale_band(np.asarray(arr[(c,) + sl], np.float32), valid)
            out[name].append(b); stats[name].append(s)
    return np.stack(out["opt"]), np.stack(out["sar"]), valid, stats


def fully_valid_topleft(valid, p):
    """Bool map of top-left positions (r, c) whose p x p window is fully valid (integral image)."""
    ii = np.zeros((valid.shape[0] + 1, valid.shape[1] + 1), np.int64)
    ii[1:, 1:] = valid.astype(np.int64).cumsum(0).cumsum(1)
    return (ii[p:, p:] - ii[:-p, p:] - ii[p:, :-p] + ii[:-p, :-p]) == p * p


def otsu_valid(d, valid):
    """The code's change_map (cv2 Otsu on uint8(d * 255), change = value > threshold), histogram on valid pixels only.
    d in [0, 1]. Returns (bool map, threshold in 0-255)."""
    u8 = (np.clip(d, 0, 1) * 255).astype(np.uint8)              # torch .type(uint8) truncates as numpy astype
    t, _ = cv2.threshold(u8[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (u8 > t) & valid, float(t)


def minmax_valid(d, valid):
    v = d[valid]
    out = (d - v.min()) / (v.max() - v.min())
    out[~valid] = 0.0
    return out.astype(np.float32)


def majority_labels(mask, size=7, alpha=0.8):
    """Vectorised equivalent of the code's sample_selection (scipy generic_filter + classify_patch, mode 'reflect'):
    label = centre value if at least int(alpha * size^2) window pixels equal the centre, else 2. mask: 0 / 1 integers."""
    m = mask.astype(np.float64)
    n = size * size
    ones = np.rint(uniform_filter(m, size=size, mode="reflect") * n)
    same = np.where(mask == 1, ones, n - ones)
    return np.where(same >= int(alpha * n), mask, 2).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- stage 1
def tiles_1d(n, core, margin):
    out = []
    for s in range(0, n, core):
        e = min(n, s + core)
        out.append((max(0, s - margin), min(n, e + margin), s, e))
    return out


@torch.no_grad()
def stage1_inference(net, xa, xb, core, margin, device, want_translations):
    """Per tile, the code's test step on the tile as a whole image: raw cross-feature DI (before min-max) and, optionally,
    x_ab / x_ba. Returns di (H, W) float32, x_ab (Cb, H, W) / x_ba (Ca, H, W) float16 or None."""
    net.eval()
    H, W = xa.shape[1:]
    di = np.zeros((H, W), np.float32)
    x_ab = np.zeros((xb.shape[0], H, W), np.float16) if want_translations else None
    x_ba = np.zeros((xa.shape[0], H, W), np.float16) if want_translations else None
    for (rs, re, r0, r1) in tiles_1d(H, core, margin):
        for (cs, ce, c0, c1) in tiles_1d(W, core, margin):
            a = torch.from_numpy(xa[None, :, rs:re, cs:ce]).to(device)
            b = torch.from_numpy(xb[None, :, rs:re, cs:ce]).to(device)
            dd = net(a, b)
            f_a = torch.cat((dd["c_a"], dd["c_a_recon"]), 1)
            f_b = torch.cat((dd["c_b_recon"], dd["c_b"]), 1)
            d = torch.sqrt(torch.sum(torch.square(f_a - f_b), 1))[0].cpu().numpy()
            sl = (slice(r0 - rs, r1 - rs), slice(c0 - cs, c1 - cs))
            di[r0:r1, c0:c1] = d[sl]
            if want_translations:
                x_ab[:, r0:r1, c0:c1] = dd["x_ab"][0].cpu().numpy()[(slice(None),) + sl].astype(np.float16)
                x_ba[:, r0:r1, c0:c1] = dd["x_ba"][0].cpu().numpy()[(slice(None),) + sl].astype(np.float16)
    return di, x_ab, x_ba


def seam_ratio(d, valid, core):
    """Mean |d(i) - d(i+1)| across tile boundaries (rows and columns at multiples of `core`) / the same statistic over all other
    adjacent pairs (valid pixels only). ~1 = no seam."""
    dv = np.abs(np.diff(d, axis=0)); vv = valid[1:] & valid[:-1]
    dh = np.abs(np.diff(d, axis=1)); vh = valid[:, 1:] & valid[:, :-1]
    rb = np.zeros(dv.shape[0], bool); rb[core - 1::core] = True
    cb = np.zeros(dh.shape[1], bool); cb[core - 1::core] = True
    on = np.concatenate([dv[rb][vv[rb]], dh[:, cb][vh[:, cb]]])
    off = np.concatenate([dv[~rb][vv[~rb]], dh[:, ~cb][vh[:, ~cb]]])
    return float(on.mean() / off.mean()) if len(on) and len(off) else None, int(len(on))


def stage1(xa, xb, valid, args, device, rng, log):
    cfg = CFG
    net = Trans_Net(xa.shape[0], xb.shape[0]).to(device)
    opt = torch.optim.Adam(net.parameters_list, lr=cfg["lr"], betas=(0.5, 0.9), weight_decay=5e-4)
    H, W = valid.shape
    p = cfg["patch_size"]
    pos = np.argwhere(fully_valid_topleft(valid, p))
    log["stage1_valid_positions"] = int(len(pos))
    mask = rng.integers(0, 2, size=(H, W)).astype(np.float32)          # code: np.random.randint(0, 2, (H, W))
    iters, epochs = (1, 1) if args.functional_test else (cfg["iter_num"], cfg["max_epoch"])
    hist = []
    for k in range(iters):
        for ep in range(epochs):
            net.train()
            sel = pos[rng.choice(len(pos), size=cfg["patches_per_epoch"], replace=len(pos) < cfg["patches_per_epoch"])]
            order = rng.permutation(len(sel))                           # code: DataLoader(shuffle=True)
            tot = 0.0
            for i in range(0, len(sel), cfg["batch_size"]):
                q = sel[order[i:i + cfg["batch_size"]]]
                pa = np.stack([xa[:, r:r + p, c:c + p] for r, c in q])
                pb = np.stack([xb[:, r:r + p, c:c + p] for r, c in q])
                pm = np.stack([mask[None, r:r + p, c:c + p] for r, c in q])
                cat = np.concatenate([pa, pb, pm], 1)
                if True:                                                # code: RandomHorizontalFlip + RandomVerticalFlip per patch
                    hf = rng.random(len(cat)) < 0.5; vf = rng.random(len(cat)) < 0.5
                    cat[hf] = cat[hf][..., ::-1]; cat[vf] = cat[vf][..., ::-1, :]
                cat = torch.from_numpy(np.ascontiguousarray(cat)).to(device)
                a, b, m = cat[:, :xa.shape[0]], cat[:, xa.shape[0]:xa.shape[0] + xb.shape[0]], cat[:, -1:]
                opt.zero_grad()
                loss = net.update(a, b, m)
                loss.backward()
                opt.step()
                tot += loss.item()
            hist.append(tot / max(1, -(-len(sel) // cfg["batch_size"])))
        last = k == iters - 1
        di_raw, x_ab, x_ba = stage1_inference(net, xa, xb, args.tile, args.overlap, device, want_translations=last)
        di = minmax_valid(di_raw, valid)
        cm, thr = otsu_valid(di, valid)
        mask = cm.astype(np.float32)
        log.setdefault("stage1_otsu", []).append(thr)
    log["stage1_loss"] = hist
    return net, di, x_ab, x_ba


# ----------------------------------------------------------------------------------------------- stage 3
def gather_patches(arrs_pad, idx, W, ps):
    """arrs_pad: (C, H + ps - 1, W + ps - 1) reflection-padded stack; idx: flat pixel indices -> (n, C, ps, ps)."""
    r, c = np.divmod(idx, W)
    off = np.arange(ps)
    return arrs_pad[:, r[:, None, None] + off[None, :, None], c[:, None, None] + off[None, None, :]].transpose(1, 0, 2, 3)


def stage3(net, xa, xb, x_ab, x_ba, pre_map, valid, args, device, rng, log):
    cfg = CFG
    ps = cfg["cd_patch_size"]
    H, W = valid.shape
    ca, cb = xa.shape[0], xb.shape[0]
    stack = np.concatenate([xa, xb, x_ab.astype(np.float32), x_ba.astype(np.float32)], 0)
    pad = ps // 2
    stack_pad = np.pad(stack, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")      # code: ReflectionPad2d(ps // 2)
    del stack
    full = minimum_filter(valid, size=ps, mode="reflect")              # adaptation 2: fully valid 7 x 7 window
    flat = pre_map.ravel()
    ch_idx = np.flatnonzero((flat == 1) & full.ravel())
    un_idx = np.flatnonzero((flat == 0) & full.ravel())
    log.update(reliable_changed=int(len(ch_idx)), reliable_unchanged=int(len(un_idx)))
    cd = CD_Net(ps * ps * 16 * 2, net.con_encoder_params).to(device)
    opt = torch.optim.Adam(cd.parameters_list_classifier, lr=cfg["cd_lr"], betas=(0.5, 0.9), weight_decay=5e-4)
    enc_a, enc_b = net.gen_a.enc_content, net.gen_b.enc_content
    epochs = 1 if args.functional_test else cfg["cd_max_epoch"]
    hist = []

    def split(t):
        return t[:, :ca], t[:, ca:ca + cb], t[:, ca + cb:ca + 2 * cb], t[:, ca + 2 * cb:]   # x_a, x_b, x_ab, x_ba

    for ep in range(epochs):
        cd.train()
        n_c = min(len(ch_idx), cfg["cd_cap_changed"])
        cs = rng.choice(ch_idx, n_c, replace=False) if len(ch_idx) > n_c else ch_idx
        us = rng.choice(un_idx, min(n_c, len(un_idx)), replace=False) if len(un_idx) > n_c else un_idx
        idx = np.concatenate([cs, us]); lab = np.concatenate([np.ones(len(cs)), np.zeros(len(us))]).astype(np.int64)
        perm = rng.permutation(len(idx)); idx, lab = idx[perm], lab[perm]
        tot, nb = 0.0, 0
        for i in range(0, len(idx), cfg["cd_batch_size"]):
            pt = gather_patches(stack_pad, idx[i:i + cfg["cd_batch_size"]], W, ps)
            hf = rng.random(len(pt)) < 0.5; vf = rng.random(len(pt)) < 0.5
            pt[hf] = pt[hf][..., ::-1]; pt[vf] = pt[vf][..., ::-1, :]
            t = torch.from_numpy(np.ascontiguousarray(pt)).to(device)
            y = torch.from_numpy(lab[i:i + cfg["cd_batch_size"]]).to(device)
            opt.zero_grad()
            loss, _ = cd.update(enc_a, enc_b, *split(t), y)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        hist.append(tot / max(nb, 1))
    log["stage3_loss"] = hist
    # prediction on all uncertain valid pixels
    unc = np.flatnonzero((flat == 2) & valid.ravel())
    pred = np.zeros(len(unc), np.uint8)
    cd.eval()
    with torch.no_grad():
        for i in range(0, len(unc), 8192):
            t = torch.from_numpy(gather_patches(stack_pad, unc[i:i + 8192], W, ps)).to(device)
            _, p = cd(enc_a, enc_b, *split(t))
            pred[i:i + 8192] = p.cpu().numpy().astype(np.uint8)
    log["uncertain_pixels"] = int(len(unc))
    out = np.where(flat == 1, 1, 0).astype(np.uint8)
    out[unc] = pred
    return (out.reshape(H, W) & valid).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--crop", type=int, nargs=4, default=None, help="r0 c0 h w (functional test only)")
    ap.add_argument("--functional_test", action="store_true", help="1 epoch per stage; NOT a result")
    args = ap.parse_args()
    print(json.dumps(vars(args)), flush=True)
    os.makedirs(args.out_dir)
    setup_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    assert torch.cuda.is_available(), "CDRL runs on one GPU (adaptation 10)"
    device = torch.device("cuda:0")
    t0 = time.time()
    xa, xb, valid, stats = load_scene(args.scene, args.crop)
    log = dict(load_s=time.time() - t0)
    print(f"scene {valid.shape}, valid {valid.mean():.3f}", flush=True)
    t = time.time()
    net, di, x_ab, x_ba = stage1(xa, xb, valid, args, device, rng, log)
    log["stage1_s"] = time.time() - t
    log["seam_ratio"], log["seam_pairs"] = seam_ratio(di, valid, args.tile)
    cm_raw, thr_raw = otsu_valid(di, valid)
    t = time.time()
    dg = gaussian_filter(di, CFG["gauss_sigma"]).astype(np.float32)
    dg[~valid] = 0.0
    mask2, thr2 = otsu_valid(dg, valid)
    pre_map = majority_labels(mask2.astype(np.uint8), CFG["cd_patch_size"], CFG["majority_alpha"])
    pre_map[~valid] = 255
    log["stage2_s"] = time.time() - t
    t = time.time()
    cm_final = stage3(net, xa, xb, x_ab, x_ba, pre_map, valid, args, device, rng, log)
    log["stage3_s"] = time.time() - t
    for name, arr in (("diff_raw", di.astype(np.float16)), ("diff_gauss", dg.astype(np.float16)),
                      ("cm_raw", cm_raw.astype(np.uint8)), ("cm_final", cm_final)):
        np.save(os.path.join(args.out_dir, name + ".npy"), arr)
    log["total_s"] = time.time() - t0
    log["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    report = dict(args=vars(args), cdrl_commit=CDRL_COMMIT, config=CFG, scaling=stats, valid_frac=float(valid.mean()),
                  otsu=dict(di=thr_raw, gauss=thr2), change_frac=dict(raw=float(cm_raw[valid].mean()),
                                                                     final=float(cm_final[valid].mean())), log=log)
    with open(os.path.join(args.out_dir, "run.json"), "x") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("otsu", "change_frac", "log")}, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
