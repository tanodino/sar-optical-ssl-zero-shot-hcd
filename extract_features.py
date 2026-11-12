"""
Zero-shot change maps of a pre-trained framework (DeCUR, CROMA or CroCoMAE) on one scene (pre-event optical t1,
post-event SAR t2, same grid). For every feature variant of the framework, the pixel-level CVA change score is
    score = || l2n(f_opt) - l2n(f_sar) ||          (CVA magnitude on l2-normalised features, a function of the cosine distance)
computed in one or two modes:

  Mode B (sliding-window global vector; the protocol of the paper). 64x64 windows on a grid of stride b_stride; window
      (r, c) covers rows [b*r + b/2 - 32, b*r + b/2 + 32) (centre = centre of the b x b pixel block (r, c)); reflection
      padding at the borders. One score per window, placed at the window centre, then bilinear interpolation to every pixel.
      Variants (aligned = vector on which the cross-modal loss is applied; encoder = encoder-only counterpart):
        crocomae      avg_shared (mean of the patch tokens, shared subspace [:192]; aligned), avg_full
        decur         enc_cls (encoder), proj_common (projector output, common part [:7168]; aligned)
        croma         enc_avg (mean of the patch tokens; encoder), head (proj(gap_ffn(mean of the tokens)); aligned)

  Mode A (dense patch tokens, shifted inference; supplementary). For every offset (dy, dx) in range(0, 8, a_stride)^2 the
      scene is shifted (reflection padding), cut into non-overlapping 64x64 tiles, each tile is encoded, the score of every
      8x8 patch fills its 64 pixels, the shift is undone and the maps of all offsets are averaged. Variants: encoder patch
      tokens ("full"); CroCoMAE also "shared" [:192] and "unshared" [192:]. The token-level map of offset (0, 0) and a
      tile-edge statistic (mean score per position in the 8x8 token grid) are stored too.

Context-16 inference: CroCoMAE and CROMA were pre-trained with 16 visible tokens out of 64. Every 64x64 input is encoded
K times (--ctx_ks), each time with a random split of its 64 tokens into 4 groups of 16 (identical for SAR and optical),
each group encoded on its own; the tokens are put back at their positions and averaged over the K splits. The CROMA head
is applied per group (as in training) and averaged. The noise of the splits is drawn per row of inputs from a generator
seeded from (--seed, mode, stride, K, [offset], row), so the split of an input depends neither on --batch nor on skipped
inputs. DeCUR: one full forward pass per input.

Median bias correction (optional; --bc_samples 0 disables it). d = l2n(f_opt) - l2n(f_sar). A pre-pass on --bc_samples
seeded random inputs estimates m = coordinate-wise median of d (Mode B: windows whose centre pixel is valid; Mode A: valid
tokens of offset-(0, 0) tiles); the main pass also writes "v_bc" = || d - m || next to each variant v.

Weights: the framework is rebuilt exactly as in pre-training and its weights are loaded strictly (model_loading.py).

No-data: invalid pixels are 0 (= band mean) in the scene arrays; metrics use valid pixels only. Inputs whose scores cannot
reach any needed (valid) pixel, dilated by --need_margin px, are not encoded (Mode A: tiles without a needed pixel;
Mode B: windows without a needed pixel in the 3 x 3 block neighbourhood of their centre). Map values on needed pixels are
identical with and without this skip (unit test); other pixels are NaN. --no_skip encodes every input.
(The east-west resampled grid handled through meta.json "W_orig" / "source" is not used for the EMS benchmark.)

Outputs (in <out_dir>/<scene>/, refused if the folder exists):
    B_w{b}[_k{K}]_{variant}.npy        float16 (H, W)   Mode B pixel map, window stride b
    B_w{b}[_k{K}]_{variant}_grid.npy   float16 (ceil(H/b), ceil(W/b))  window-level scores
    A_s{a}[_k{K}]_{variant}.npy        float16 (H, W)   Mode A pixel map, shift stride a
    A_s{a}[_k{K}]_{variant}_tok.npy    float16 (ceil(H/8), ceil(W/8))  token-level map of offset (0, 0)
    <tag>_{variant}_median.npy         float32 (D,)  median change vector of the bias correction
    run.json  configuration, model info, timings, number of encoder inputs, peak GPU memory, tile-edge statistics

Usage (paper setting, one GPU):
    python extract_features.py --run <weights>/croma --scene data/scenes/EMSR832_AOI01 --out_dir features/croma \
        --modes B --b_strides 8 --ctx_ks 10 --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loading import load_model                                       # noqa: E402  (strict loading)
from ew_resample import forward_valid                                       # noqa: E402

MASKED = ("crocomae", "croma")       # trained on 16 visible tokens of 64 -> context-16 inference
T, P = 64, 8                                # input tile / window size and patch size (pixels)
NTOK, CTX = (T // P) ** 2, 16               # 64 tokens per input, groups of 16


# ----------------------------------------------------------------------------------------------- geometry helpers
def reflect_index(i: np.ndarray, n: int) -> np.ndarray:
    """Indices of a reflection-padded axis (as numpy 'reflect' / torch 'reflect': edge not repeated). |pad| < n."""
    i = np.abs(i)
    i = np.where(i >= n, 2 * (n - 1) - i, i)
    assert (i >= 0).all() and (i < n).all(), "padding larger than the scene"
    return i


def bilinear_from_grid(G: torch.Tensor, H: int, W: int, s: int) -> torch.Tensor:
    """Window-score grid (ny, nx), centre of cell (r, c) at continuous position (s*r + s/2, s*c + s/2) (pixel k has its
    centre at k + 0.5) -> (H, W) map by separable linear interpolation; values beyond the outer centres held constant."""
    def axis(n_out, n_grid):
        u = ((torch.arange(n_out, dtype=torch.float64, device=G.device) + 0.5) - s / 2) / s   # grid-index units
        u = u.clamp(0, n_grid - 1)
        k0 = torch.clamp(u.floor().long(), 0, max(n_grid - 2, 0))
        f = (u - k0).float() if n_grid > 1 else torch.zeros(n_out, device=G.device)
        k1 = torch.clamp(k0 + 1, max=n_grid - 1)
        return k0, k1, f
    y0, y1, fy = axis(H, G.shape[0])
    x0, x1, fx = axis(W, G.shape[1])
    Gy = G[y0] * (1 - fy)[:, None] + G[y1] * fy[:, None]                   # (H, nx)
    return Gy[:, x0] * (1 - fx)[None] + Gy[:, x1] * fx[None]               # (H, W)


def gather(arr: np.ndarray, R: np.ndarray, C: np.ndarray) -> torch.Tensor:
    """arr (Ch, H, W), row indices R (B, 64), column indices C (B, 64) -> inputs (B, Ch, 64, 64) (one fancy index)."""
    return torch.from_numpy(np.ascontiguousarray(arr[:, R[:, :, None], C[:, None, :]].transpose(1, 0, 2, 3)))


def cva(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CVA magnitude on l2-normalised features (last dim)."""
    return (F.normalize(a.float(), dim=-1) - F.normalize(b.float(), dim=-1)).norm(dim=-1)


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation of a 2-D bool mask by a (2r+1) x (2r+1) square (separable running max)."""
    out = m.copy()
    for ax in (0, 1):
        src = out.copy()
        for d in range(1, r + 1):
            if ax == 0:
                out[d:] |= src[:-d]; out[:-d] |= src[d:]
            else:
                out[:, d:] |= src[:, :-d]; out[:, :-d] |= src[:, d:]
    return out


def needed_mask(valid: np.ndarray, orig_valid: np.ndarray = None, margin: int = 2) -> np.ndarray:
    """Pixels whose map value is used downstream: valid pixels, plus (ew-resampled scene) every ew pixel overlapping a
    valid original pixel (= NOT all overlapped original pixels invalid), dilated by `margin`."""
    need = valid.astype(bool).copy()
    if orig_valid is not None:
        assert orig_valid.shape[0] == valid.shape[0]
        need |= ~forward_valid(~orig_valid.astype(bool), valid.shape[1])
    return dilate(need, margin) if margin > 0 else need


def batches(keep: np.ndarray, row_noise, batch: int):
    """Row-major iteration over the kept inputs of a (n_rows, n_cols) grid in batches of <= `batch`.
    row_noise(r) -> (n_cols, K, 64) noise tensor of row r, or None. Yields (rows, cols, noise or None)."""
    br, bc, bn, n = [], [], [], 0
    for r in range(keep.shape[0]):
        cols = np.nonzero(keep[r])[0]
        if len(cols) == 0:
            continue
        nz = row_noise(r)
        if nz is not None:
            nz = nz[torch.from_numpy(cols).to(nz.device)]
        i = 0
        while i < len(cols):
            take = min(batch - n, len(cols) - i)
            br.append(np.full(take, r)); bc.append(cols[i:i + take])
            if nz is not None:
                bn.append(nz[i:i + take])
            n += take; i += take
            if n == batch:
                yield np.concatenate(br), np.concatenate(bc), (torch.cat(bn) if bn else None)
                br, bc, bn, n = [], [], [], 0
    if n:
        yield np.concatenate(br), np.concatenate(bc), (torch.cat(bn) if bn else None)


# ----------------------------------------------------------------------------------------------- encoders
class Extractor:
    """Wraps one method (loaded from resume.pth) and returns, for a batch of 64x64 inputs, the CVA scores of every
    variant: tokens (Mode A, (B, 64) per variant) or global vectors (Mode B, (B,) per variant)."""

    def __init__(self, run: str, device: str):
        info = json.load(open(os.path.join(run, "model_info.json")))
        self.method = info["method"]
        self.model, self.step = load_model(run, info)
        self.model = self.model.to(device).eval()
        self.info, self.device = info, device
        self.k_shared = int(info["arch"]["embed_dim"] * info["shared_pct"] / 100) if info.get("shared_pct") else None
        self.amp = device.startswith("cuda")

    def variants(self, mode: str):
        m = self.method
        if mode == "A":
            return ["full", "shared", "unshared"] if m == "crocomae" else ["full"]
        return {"crocomae": ["avg_shared", "avg_full"], "decur": ["enc_cls", "proj_common"],
                "croma": ["enc_avg", "head"]}[m]

    def _encode(self, xs, xo, ctx_k, noise, want_head):
        """-> tokens_sar, tokens_opt (B, 64, D), cls_sar, cls_opt (B, D) or None, head_sar, head_opt or None."""
        mdl = self.model
        if self.method not in MASKED:
            ts, to = mdl.enc_sar(xs), mdl.enc_opt(xo)
            return ts[:, 1:], to[:, 1:], ts[:, 0], to[:, 0], None, None
        B, G = xs.shape[0], NTOK // CTX
        tok_s = tok_o = head_s = head_o = 0.0
        ar = torch.arange(B, device=xs.device)
        assert noise is not None and noise.shape == (B, ctx_k, NTOK), None if noise is None else noise.shape
        noise = noise.to(xs.device)
        for k in range(ctx_k):
            perm = torch.argsort(noise[:, k], dim=1)                          # one split per input (same SAR / opt)
            ids = perm.view(B, G, CTX).transpose(0, 1).reshape(G * B, CTX)                     # groups stacked on batch
            a = mdl.enc_sar.forward_masked(xs.repeat(G, 1, 1, 1), ids)                          # (G*B, 1 + 16, D)
            b = mdl.enc_opt.forward_masked(xo.repeat(G, 1, 1, 1), ids)
            D = a.shape[-1]
            out_s = torch.empty(B, NTOK, D, device=xs.device, dtype=a.dtype)
            out_o = torch.empty_like(out_s)
            for g in range(G):                                                 # tokens back to their positions
                sl = slice(g * B, (g + 1) * B)
                out_s[ar[:, None], ids[sl]] = a[sl, 1:]
                out_o[ar[:, None], ids[sl]] = b[sl, 1:]
            tok_s = tok_s + out_s.float() / ctx_k
            tok_o = tok_o + out_o.float() / ctx_k
            if want_head:                                                      # CROMA head per group, as in training
                hs = mdl.proj_sar(mdl.gap_ffn_sar(a[:, 1:].mean(1))).float().view(G, B, -1).mean(0)
                ho = mdl.proj_opt(mdl.gap_ffn_opt(b[:, 1:].mean(1))).float().view(G, B, -1).mean(0)
                head_s, head_o = head_s + hs / ctx_k, head_o + ho / ctx_k
        return tok_s, tok_o, None, None, (head_s if want_head else None), (head_o if want_head else None)

    @torch.no_grad()
    def pairs(self, xs, xo, mode, ctx_k, noise):
        """{variant: (f_sar, f_opt)} feature pairs, float32: Mode A (B, 64, D), Mode B (B, D).
        noise: (B, ctx_k, 64) uniform noise defining the context-16 splits (masked methods), ignored otherwise."""
        xs = xs.to(self.device, non_blocking=True).float()
        xo = xo.to(self.device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            want_head = mode == "B" and self.method == "croma"
            ts, to, cs, co, hs, ho = self._encode(xs, xo, ctx_k, noise, want_head)
            ts, to = ts.float(), to.float()
            out, m, k = {}, self.method, self.k_shared
            if mode == "A":
                out["full"] = (ts, to)
                if m == "crocomae":
                    out["shared"], out["unshared"] = (ts[..., :k], to[..., :k]), (ts[..., k:], to[..., k:])
                return out
            if m == "crocomae":
                avs, avo = ts.mean(1), to.mean(1)
                return {"avg_shared": (avs[:, :k], avo[:, :k]), "avg_full": (avs, avo)}
            if m == "croma":
                out["enc_avg"] = (ts.mean(1), to.mean(1))
                if m == "croma":
                    out["head"] = (hs.float(), ho.float())
                return out
            out["enc_cls"] = (cs.float(), co.float())
            zs, zo = self.model.proj_sar(cs).float(), self.model.proj_opt(co).float()
            if m == "decur":
                dc = self.model.dim_common
                out["proj_common"] = (zs[:, :dc], zo[:, :dc])
            else:
                out["proj"] = (zs, zo)
            return out

    @torch.no_grad()
    def diffs(self, xs, xo, mode, ctx_k, noise):
        """{variant: d = l2n(f_opt) - l2n(f_sar)} (float32); ||d|| = plain CVA score."""
        return {v: F.normalize(b, dim=-1) - F.normalize(a, dim=-1) for v, (a, b) in self.pairs(xs, xo, mode, ctx_k, noise).items()}

    @torch.no_grad()
    def scores(self, xs, xo, mode, ctx_k, noise, med=None):
        """{variant: plain CVA score} (Mode A (B, 64), Mode B (B,)); with med = {variant: (D,) median vector} also
        {variant + "_bc": || d - med ||}."""
        out = {}
        for v, d in self.diffs(xs, xo, mode, ctx_k, noise).items():
            out[v] = d.norm(dim=-1)
            if med is not None:
                out[v + "_bc"] = (d - med[v].to(d.device)).norm(dim=-1)
        return out


def estimate_median(diff_fn, sar, opt, valid, mode, stride, n_sample, batch, seed, noise_fn=None):
    """Bias-correction pre-pass. diff_fn(xs, xo, noise) -> {variant: d}; noise_fn(n) -> (n, K, 64) or None.
    Mode A: offset-(0, 0) 64x64 tiles (reflection padding) with >= 1 valid token (all 64 px valid); the median uses
    their valid tokens. Mode B: windows of the stride grid whose centre pixel is valid. n_sample inputs drawn without
    replacement (numpy rng seeded by (seed, 2, mode, stride)). Returns ({variant: (D,) lower median}, n_inputs, n_vectors)."""
    H, W = valid.shape
    rng = np.random.default_rng([seed, 2, 0 if mode == "A" else 1, stride])
    if mode == "A":
        nty, ntx = -(-H // T), -(-W // T)
        ri, ci = reflect_index(np.arange(nty * T), H), reflect_index(np.arange(ntx * T), W)
        vp = np.zeros((nty * T, ntx * T), bool)
        vp[:H, :W] = valid
        tv = vp.reshape(nty * T // P, P, ntx * T // P, P).all(axis=(1, 3))             # token validity
        tv = tv.reshape(nty, T // P, ntx, T // P).transpose(0, 2, 1, 3).reshape(nty, ntx, NTOK)
        cand = np.argwhere(tv.any(-1))
    else:
        ny, nx = -(-H // stride), -(-W // stride)
        off = np.arange(T) - T // 2 + stride // 2
        rows_all = reflect_index(stride * np.arange(ny)[:, None] + off, H)
        cols_all = reflect_index(stride * np.arange(nx)[:, None] + off, W)
        cy, cx = np.minimum(stride * np.arange(ny) + stride // 2, H - 1), np.minimum(stride * np.arange(nx) + stride // 2, W - 1)
        cand = np.argwhere(valid[cy][:, cx])
    assert len(cand), "no valid input for the bias-correction median"
    pick = cand[np.sort(rng.choice(len(cand), size=min(n_sample, len(cand)), replace=False))]
    noise = noise_fn(len(pick)) if noise_fn else None
    vecs = {}
    for i in range(0, len(pick), batch):
        q = pick[i:i + batch]
        if mode == "A":
            R, C = ri[q[:, :1] * T + np.arange(T)], ci[q[:, 1:] * T + np.arange(T)]
        else:
            R, C = rows_all[q[:, 0]], cols_all[q[:, 1]]
        d = diff_fn(gather(sar, R, C), gather(opt, R, C), None if noise is None else noise[i:i + batch])
        for v, x in d.items():
            if mode == "A":
                x = x[torch.from_numpy(tv[q[:, 0], q[:, 1]]).to(x.device)]               # valid tokens only
            vecs.setdefault(v, []).append(x.float())
    med = {v: torch.cat(x).median(dim=0).values for v, x in vecs.items()}
    return med, int(len(pick)), int(sum(len(x) for x in vecs[next(iter(vecs))]))


# ----------------------------------------------------------------------------------------------- Mode A
def mode_a(score_fn, sar: np.ndarray, opt: np.ndarray, valid: np.ndarray, stride: int, batch: int, device: str,
           noise_fn=None, need: np.ndarray = None, skip: bool = True):
    """Shifted inference. score_fn(xs, xo, noise) -> {variant: (B, 64) token scores} for inputs (B, C, 64, 64).
    noise_fn(offset, row, n_cols) -> (n_cols, K, 64) noise or None. need: (H, W) bool (None = all pixels); with skip,
    tiles without a needed scene pixel are not encoded. Returns ({variant: (H, W) float32 map, NaN outside need},
    {variant: token map of offset (0, 0)}, artifact stats, n_inputs, n_inputs_without_skip)."""
    H, W = valid.shape
    need = np.ones((H, W), bool) if need is None else need
    n_total = 0
    acc, tok0, pos_sum, pos_cnt, n_inputs = {}, {}, {}, np.zeros((T // P, T // P)), 0
    offsets = [(dy, dx) for dy in range(0, P, stride) for dx in range(0, P, stride)]
    for dy, dx in offsets:
        Hp, Wp = -(-(H + dy) // T) * T, -(-(W + dx) // T) * T            # padded size, multiple of 64
        ri = reflect_index(np.arange(Hp) - dy, H)                         # padded row -> scene row
        ci = reflect_index(np.arange(Wp) - dx, W)
        rin = (np.arange(Hp) - dy >= 0) & (np.arange(Hp) - dy < H)        # padded pixel inside the scene
        cin = (np.arange(Wp) - dx >= 0) & (np.arange(Wp) - dx < W)
        nty, ntx = Hp // T, Wp // T
        tok = {}                                                          # token score grids (Hp/8, Wp/8), NaN = skipped
        if skip:                                                          # tile kept iff it holds a needed scene pixel
            npad = np.zeros((Hp, Wp), bool)
            npad[dy:dy + H, dx:dx + W] = need
            keep = npad.reshape(nty, T, ntx, T).any(axis=(1, 3))
        else:
            keep = np.ones((nty, ntx), bool)
        n_total += nty * ntx
        rnoise = (lambda r: noise_fn((dy, dx), r, ntx)) if noise_fn else (lambda r: None)
        for qr, qc, nz in batches(keep, rnoise, batch):
            R = ri[qr[:, None] * T + np.arange(T)]                       # (B, 64) scene rows of each tile
            C = ci[qc[:, None] * T + np.arange(T)]
            sc = score_fn(gather(sar, R, C), gather(opt, R, C), nz)
            rr, cc = torch.from_numpy(qr).to(device), torch.from_numpy(qc).to(device)
            for v, s in sc.items():
                if v not in tok:
                    tok[v] = torch.full((Hp // P, Wp // P), float("nan"), device=device)
                tok[v].view(nty, T // P, ntx, T // P)[rr, :, cc, :] = s.float().view(len(qr), T // P, T // P)
            n_inputs += len(qr)

        # token validity: all 64 pixels inside the scene and valid (artifact statistics only)
        vp = valid[ri][:, ci].astype(bool) & rin[:, None] & cin[None, :]
        tv = torch.from_numpy(vp.reshape(Hp // P, P, Wp // P, P).all(axis=(1, 3))).to(device)
        iy = torch.from_numpy((np.arange(H) + dy) // P).to(device)          # scene pixel -> token row
        ix = torch.from_numpy((np.arange(W) + dx) // P).to(device)
        for v, g in tok.items():
            if v not in acc:
                acc[v] = torch.zeros(H, W, device=device)
                pos_sum[v] = np.zeros((T // P, T // P))
            acc[v] += g[iy][:, ix]
            if (dy, dx) == (0, 0):
                tok0[v] = g[:-(-H // P), :-(-W // P)].half().cpu().numpy()
            gs = torch.where(tv, g, torch.zeros_like(g)).view(nty, 8, ntx, 8).sum((0, 2)).cpu().numpy()  # per position
            pos_sum[v] += gs
        pos_cnt += tv.view(nty, 8, ntx, 8).sum((0, 2)).cpu().numpy() if tok else 0
    maps = {v: (a / len(offsets)) for v, a in acc.items()}
    nd = torch.from_numpy(need).to(device)
    for v, m in maps.items():
        assert torch.isfinite(m[nd]).all(), f"non-finite score on a needed pixel ({v})"
        m[~nd] = float("nan")
    art = {}
    for v in pos_sum:
        pm = pos_sum[v] / np.maximum(pos_cnt, 1)
        edge = np.zeros((8, 8), bool); edge[[0, -1], :] = True; edge[:, [0, -1]] = True
        centre = np.zeros((8, 8), bool); centre[2:6, 2:6] = True
        art[v] = dict(pos_mean=pm.round(5).tolist(),
                      edge_mean=float((pos_sum[v][edge]).sum() / max(pos_cnt[edge].sum(), 1)),
                      centre_mean=float((pos_sum[v][centre]).sum() / max(pos_cnt[centre].sum(), 1)),
                      n_valid_tokens=int(pos_cnt.sum()))
        art[v]["edge_minus_centre"] = art[v]["edge_mean"] - art[v]["centre_mean"]
    return maps, tok0, art, n_inputs, n_total


# ----------------------------------------------------------------------------------------------- Mode B
def mode_b(score_fn, sar: np.ndarray, opt: np.ndarray, H: int, W: int, stride: int, batch: int, device: str,
           noise_fn=None, need: np.ndarray = None, skip: bool = True):
    """Sliding windows. score_fn(xs, xo, noise) -> {variant: (B,) window scores}; noise_fn(row, n_cols) -> (n_cols, K, 64)
    or None. With skip, window (r, c) is encoded iff a needed pixel lies in rows [s(r-1), s(r+2)) x cols [s(c-1), s(c+2))
    (superset of the pixels whose bilinear value uses it). Returns ({variant: (H, W) map, NaN outside need},
    {variant: (ny, nx) grid, NaN = skipped}, n_inputs, n_inputs_without_skip)."""
    assert stride % 2 == 0 and stride <= T, stride
    ny, nx = -(-H // stride), -(-W // stride)
    off = np.arange(T) - T // 2 + stride // 2                              # window pixels relative to s*r
    need = np.ones((H, W), bool) if need is None else need
    grid = {}
    rows_all = reflect_index(stride * np.arange(ny)[:, None] + off, H)    # (ny, 64)
    cols_all = reflect_index(stride * np.arange(nx)[:, None] + off, W)    # (nx, 64)
    if skip:
        npad = np.zeros((ny * stride, nx * stride), bool)
        npad[:H, :W] = need
        blk = npad.reshape(ny, stride, nx, stride).any(axis=(1, 3))      # s x s block (r, c) holds a needed pixel
        keep = dilate(blk, 1)                                             # blocks r-1..r+1, c-1..c+1
    else:
        keep = np.ones((ny, nx), bool)
    rnoise = (lambda r: noise_fn(r, nx)) if noise_fn else (lambda r: None)
    n = 0
    for qr, qc, nz in batches(keep, rnoise, batch):
        sc = score_fn(gather(sar, rows_all[qr], cols_all[qc]), gather(opt, rows_all[qr], cols_all[qc]), nz)
        rr, cc = torch.from_numpy(qr).to(device), torch.from_numpy(qc).to(device)
        for v, s in sc.items():
            if v not in grid:
                grid[v] = torch.full((ny, nx), float("nan"), device=device)
            grid[v][rr, cc] = s.float()
        n += len(qr)
    maps = {v: bilinear_from_grid(g, H, W, stride) for v, g in grid.items()}
    nd = torch.from_numpy(need).to(device)
    for v, m in maps.items():
        assert torch.isfinite(m[nd]).all(), f"non-finite score on a needed pixel ({v})"
        m[~nd] = float("nan")
    return maps, {v: g.half().cpu().numpy() for v, g in grid.items()}, n, ny * nx


# ----------------------------------------------------------------------------------------------- main
def make_gen(device, *key):
    g = torch.Generator(device=device)
    g.manual_seed(int(np.random.SeedSequence([int(k) for k in key]).generate_state(1)[0]))
    return g


def row_noise(device, K, n_cols, *key):
    """Uniform noise (n_cols, K, 64) of one row of inputs, from a generator seeded by `key` (defines the splits)."""
    return torch.rand(n_cols, K, NTOK, generator=make_gen(device, *key), device=device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="pre-training folder (model_info.json, config.json, resume.pth)")
    ap.add_argument("--scene", required=True, help="scene folder (opt.npy (4,H,W), sar.npy (2,H,W), valid.npy)")
    ap.add_argument("--out_dir", required=True, help="outputs in <out_dir>/<scene name>/ (refused if it exists)")
    ap.add_argument("--modes", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--a_strides", type=int, nargs="+", default=[2], help="Mode A shift strides (1, 2, 4, 8)")
    ap.add_argument("--b_strides", type=int, nargs="+", default=[16], help="Mode B window strides (even, <= 64)")
    ap.add_argument("--ctx_ks", type=int, nargs="+", default=[10], help="K random splits (masked methods only)")
    ap.add_argument("--batch", type=int, default=256, help="64x64 inputs per forward pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--need_margin", type=int, default=2, help="dilation (px) of the needed-pixel mask")
    ap.add_argument("--no_skip", action="store_true", help="encode every input (default: skip inputs without needed px)")
    ap.add_argument("--bc_samples", type=int, default=2000, help="inputs of the bias-correction pre-pass (0 = no _bc maps)")
    args = ap.parse_args()
    print(vars(args), flush=True)
    torch.set_num_threads(args.threads)
    scene = os.path.basename(os.path.normpath(args.scene))
    od = os.path.join(args.out_dir, scene)
    os.makedirs(od, exist_ok=False)

    ex = Extractor(args.run, args.device)
    print(f"method {ex.method}, resume step {ex.step}, variants A {ex.variants('A')} B {ex.variants('B')}", flush=True)
    sar = np.ascontiguousarray(np.load(os.path.join(args.scene, "sar.npy")))
    opt = np.ascontiguousarray(np.load(os.path.join(args.scene, "opt.npy")))
    valid = np.load(os.path.join(args.scene, "valid.npy")).astype(bool)
    H, W = valid.shape
    assert sar.shape[1:] == (H, W) and opt.shape[1:] == (H, W)
    smeta = json.load(open(os.path.join(args.scene, "meta.json"))) if os.path.exists(os.path.join(args.scene, "meta.json")) else {}
    orig_valid = None
    if "W_orig" in smeta:                                                 # GeoFlood ew scene: original validity
        orig_valid = np.load(os.path.join(smeta["source"], "valid.npy")).astype(bool)
        assert orig_valid.shape == (H, smeta["W_orig"]) and smeta["W_new"] == W
    need = needed_mask(valid, orig_valid, args.need_margin)
    print(f"needed pixels {need.mean():.4f} of the scene (valid {valid.mean():.4f}), skip {not args.no_skip}", flush=True)
    ks = args.ctx_ks if ex.method in MASKED else [1]
    rep = dict(config=vars(args), method=ex.method, resume_step=ex.step, model_info=ex.info, scene=scene, H=H, W=W,
               valid_frac=float(valid.mean()), need_frac=float(need.mean()), skip=not args.no_skip,
               masked_context=ex.method in MASKED, runs=[])
    if os.path.exists(os.path.join(args.scene, "meta.json")):
        rep["scene_meta"] = json.load(open(os.path.join(args.scene, "meta.json")))

    def save(name, arr):
        np.save(os.path.join(od, name + ".npy"), arr)

    for K in ks:
        ktag = f"_k{K}" if ex.method in MASKED else ""
        for mode in args.modes:
            for s in (args.a_strides if mode == "A" else args.b_strides):
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                msk = ex.method in MASKED
                med, bc = None, None
                if args.bc_samples > 0:                                       # bias-correction pre-pass
                    tb = time.time()
                    bnz = (lambda n_: torch.rand(n_, K, NTOK, generator=make_gen(args.device, args.seed, 2 + (mode == "B"), s, K),
                                                 device=args.device)) if msk else None   # noqa: E731
                    med, bn, bv = estimate_median(lambda xs, xo, nz: ex.diffs(xs, xo, mode, K, nz), sar, opt, valid, mode, s,
                                                  args.bc_samples, args.batch, args.seed, bnz)
                    bc = dict(seconds=round(time.time() - tb, 2), n_inputs=bn, n_vectors=bv,
                              median_norm={v: float(m.norm()) for v, m in med.items()})
                fn = lambda xs, xo, nz: ex.scores(xs, xo, mode, K, nz, med)   # noqa: E731
                if mode == "A":
                    nfn = (lambda o, r, n_: row_noise(args.device, K, n_, args.seed, 0, s, K, o[0], o[1], r)) if msk else None  # noqa: E731
                    maps, tok0, art, n, n_tot = mode_a(fn, sar, opt, valid, s, args.batch, args.device, nfn, need,
                                                       not args.no_skip)
                    tag = f"A_s{s}{ktag}"
                    for v in (med or {}):
                        save(f"{tag}_{v}_median", med[v].cpu().numpy().astype(np.float32))
                    for v in maps:
                        save(f"{tag}_{v}", maps[v].half().cpu().numpy()); save(f"{tag}_{v}_tok", tok0[v])
                else:
                    nfn = (lambda r, n_: row_noise(args.device, K, n_, args.seed, 1, s, K, r)) if msk else None  # noqa: E731
                    maps, grid, n, n_tot = mode_b(fn, sar, opt, H, W, s, args.batch, args.device, nfn, need,
                                                  not args.no_skip)
                    art, tag = None, f"B_w{s}{ktag}"
                    for v in (med or {}):
                        save(f"{tag}_{v}_median", med[v].cpu().numpy().astype(np.float32))
                    for v in maps:
                        save(f"{tag}_{v}", maps[v].half().cpu().numpy()); save(f"{tag}_{v}_grid", grid[v])
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                dt = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 1e9 if args.device.startswith("cuda") else None
                r = dict(tag=tag, mode=mode, stride=s, K=K if ex.method in MASKED else None, seconds=round(dt, 2),
                         n_inputs=n, n_inputs_without_skip=n_tot, peak_gpu_gb=peak, variants=list(maps),
                         map_stats={v: dict(mean=float(m[torch.from_numpy(valid).to(m.device)].mean()),
                                            std=float(m[torch.from_numpy(valid).to(m.device)].std()))
                                    for v, m in maps.items()},
                         artifact=art, bias_correction=bc)
                rep["runs"].append(r)
                print(f"{tag:14s} {dt:8.1f} s  inputs {n:8d} / {n_tot:8d}  peak {peak if peak is None else round(peak, 2)} GB"
                      + ("" if art is None else "  edge-centre " + ", ".join(
                          f"{v} {a['edge_minus_centre']:+.4f}" for v, a in art.items())), flush=True)
    with open(os.path.join(od, "run.json"), "x") as f:
        json.dump(rep, f, indent=1)
    print("done ->", od)


if __name__ == "__main__":
    main()
