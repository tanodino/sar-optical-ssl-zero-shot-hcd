# Zero-shot SAR-optical change detection with cross-modal self-supervised representations

Code of the paper (citation below). Three cross-modal self-supervised frameworks (DeCUR, CROMA, CroCoMAE) are pre-trained
under the same protocol on a subset of SSL4EO-S12 v1.1, then applied **zero-shot** to detect burnt areas between a pre-event
Sentinel-2 image and a post-event Sentinel-1 image (change vector analysis of their features on sliding windows), and compared
with four dedicated heterogeneous change detection methods trained on each scene (CAA, SCCN, cGAN, CDRL).

Related releases: the **EMS Wildfires 2025 benchmark** (https://doi.org/10.5281/zenodo.24166214) and the **pre-trained weights** (https://huggingface.co/tanodino/sar-optical-ssl-zero-shot-cd).

## Layout

```
sslcd/                       training engine (data, augmentations, optimiser / schedule, checkpoints, collapse check)
  methods/                   DeCUR, CROMA, CroCoMAE (one module each: model + losses)
pretrain.py                  pre-training entry point
# pre-training corpus (SSL4EO-S12 v1.1)
exclude_leakage.py           selection of the corpus subset (spatial exclusion around the test areas)
download_ssl4eo_subset.py    download of the selected shards (Hugging Face)
verify_ssl4eo_subset.py      checks of the downloaded subset (pairs, dates)
preprocess_corpus.py         64 x 64 tiles, clipping / standardisation, norm_stats.json
leakage_overlap_check.py     helper used by the selection
# benchmark (rebuild from scratch; normally just download the released benchmark)
select_ems_wildfire.py       selection of the EMS AOIs (EMS API + Google Earth Engine availability)
download_ems_wildfire.py     download of Sentinel-2 / Sentinel-1 from Google Earth Engine
preprocess_ems_wildfire.py   reference / masks / normalised arrays per AOI
prepare_scenes.py            released benchmark (GeoTIFFs) -> per-scene arrays read by the pipeline
# zero-shot change detection and evaluation
extract_features.py          change maps of a pre-trained framework on a scene (sliding windows; bias correction optional)
model_loading.py             strict loading of the pre-trained weights (used by extract_features.py)
cva.py                       metrics (exact AUROC / AUPRC from histograms), thresholds, helpers
evaluate_benchmark.py        AUROC / AUPRC (pooled and per-scene), ranks, Friedman / Nemenyi tests, for any set of methods
ew_resample.py               helper used by cva.py / extract_features.py
# dedicated methods (third-party code, not included)
caa_run.py                   CAA wrapper
hcd_baselines_run.py         SCCN and cGAN wrapper
cdrl_run.py                  CDRL wrapper
tests/                       unit tests (python -m unittest discover -s tests)
envs/                        pinned requirements of the three environments
```

## Environments

- `envs/requirements.txt`: main environment (Python 3.10, PyTorch 2.3.1): everything except the items below.
- `envs/requirements_tf.txt`: TensorFlow 2.14 environment for CAA, SCCN and cGAN.
- `envs/requirements_gee.txt`: Google Earth Engine environment, only to rebuild the benchmark from scratch.

## Usage

**1. Benchmark.** Download the released benchmark and convert it to the pipeline's format:
```
python prepare_scenes.py --benchmark <path>/ems_wildfire_2025_v1 --out_dir data/scenes
```

**2. Pre-trained weights.** Download the released weights (one folder per framework), or pre-train them (step 5).

**3. Change maps** of one framework on one scene (Mode B = sliding windows of 64 x 64, stride 8; K = 10 random splits for the
masked frameworks CROMA and CroCoMAE):
```
python extract_features.py --run <weights>/croma --scene data/scenes/EMSR832_AOI01 --out_dir features/croma \
    --modes B --b_strides 8 --ctx_ks 10 --seed 0
```
The aligned vectors used in the paper are `proj_common` (DeCUR), `head` (CROMA) and `avg_shared` (CroCoMAE); the encoder-only
vectors are `enc_cls` (DeCUR) and `enc_avg` (CROMA).

**4. Evaluation** of any set of continuous change maps (one `.npy` per scene and method), identical for all methods:
```
python evaluate_benchmark.py --scenes_root data/scenes --out_dir results/benchmark \
    --map "CROMA=features/croma/{scene}/B_w8_k10_head.npy" --map "CDRL=cdrl/{scene}/diff_raw.npy" ...
```
It reports AUROC and AUPRC pooled over all valid pixels and averaged over the scenes, average ranks, the Friedman test and
the Nemenyi critical difference. Run on the maps of the paper, it reproduces Table III. The dedicated methods are evaluated
on their continuous maps: `diff_crf.npy` for CAA, SCCN and cGAN, `diff_raw.npy` for CDRL.

**5. Pre-training** (one GPU; 148,000 of a 150,000-step schedule were used in the paper, batch 512, see the paper):
```
python download_ssl4eo_subset.py ...   # see the script header
python preprocess_corpus.py ...
python pretrain.py --method croma --sar <tiles>/SAR_data_normalized.npy --opt <tiles>/MS_data_normalized.npy \
    --norm_stats <tiles>/norm_stats.json --out_dir pretrain/croma --steps 150000 --seed 0
```

**6. Dedicated methods.** Clone the third-party code at the commits used in the paper (not redistributed here):
```
git clone https://github.com/llu025/Heterogeneous_CD third_party/Heterogeneous_CD && git -C third_party/Heterogeneous_CD checkout ef0784f
git clone https://github.com/OMEGA-RS/CDRL third_party/CDRL && git -C third_party/CDRL checkout b1c755b
```
The same commits are permanently archived by Software Heritage (https://archive.softwareheritage.org):
`swh:1:rev:ef0784fba5ee26653e02f8b8f92683518c628b60` (Heterogeneous_CD; CAA, SCCN, cGAN) and
`swh:1:rev:b1c755bddeda6585e200de06db468522bf6ce131` (CDRL). This code is not redistributed here.

Then run `caa_run.py`, `hcd_baselines_run.py --method sccn|cgan` (TensorFlow environment) or `cdrl_run.py` on each scene (see the
script headers). The wrappers keep the settings of these implementations; their adaptations are documented in the headers.

## Licence

Code: **GNU Affero General Public License v3.0** (`LICENSE`).
Two short functions follow the algorithms of MAE (He et al., CVPR 2022): `sslcd/vit.py::random_masking` and
`sslcd/methods/croma.py::get_2d_sincos_pos_embed` (independent implementations). The methods follow their papers and official
code: DeCUR (Wang et al., ECCV 2024), CROMA (Fuller et al., NeurIPS 2023), CroCoMAE (Recchia et al., arXiv:2611.14124).

## How to cite

TODO: paper reference.
