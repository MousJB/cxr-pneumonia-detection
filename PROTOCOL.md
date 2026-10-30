# Pre-registered protocol — lung-opacity detection (RSNA Pneumonia Detection Challenge)

[日本語](PROTOCOL.ja.md)

Written on 2026-10-30 and 2026-10-31 and committed **before any final training**. Every decision below
(models, recipe, budget, metrics, selection rule, test criteria) is fixed by the commit
that adds this file; its hash is written in every training log. The machine-readable
part for MedVision QA is [`config.toml`](config.toml). A change after this commit is only
allowed as a dated amendment (§13). [`PROTOCOL.ja.md`](PROTOCOL.ja.md) has the same content
in Japanese; in case of discrepancy, this English version prevails.

## 0. State of the project when this protocol was written

| Done | Not done |
|---|---|
| Step 1: download, volume checks, SHA-256 manifest of the 29,688 files | Any final training |
| Step 2 (commit `abfc675`): 512 px PNG, patient-level split, duplicate audit, competition metric (9 tests) | Any detection metric of any model, on validation or test |
| Computational benchmarks (§5): short runs on **training** images, weights discarded | Any use of the test split other than counting images |
| Smoke tests of `scripts/train.py` (20 steps per epoch, 40 validation images) to check the code, including resuming after an interruption; outputs not used | |

The benchmark results were known when this protocol was written. They are used only for
the budget and for the choice of configurations, never as evidence of detection quality.
The first benchmark ran in float16 mixed precision, which turned out to produce non-finite
losses on this GPU from the first step; its numbers are kept in §5.2 but are not used.
Every number used for a decision comes from float32 measurements (§5.3–§5.5).

## 1. Objective

Compare detection accuracy and computational cost of Faster R-CNN configurations for
lung-opacity localisation on chest radiographs, on a GTX 1650 (4 GB), and deliver one
selected detector whose test performance is measured once, under criteria fixed now.

## 2. Research questions and hypotheses

Two kinds of answers are kept apart: **observations** from the computational benchmark
(already measured, §5) and **pre-registered hypotheses** about detection quality (not
yet measured; tested by the final training, §6–§9).

| # | Question | Answered by | Status |
|---|---|---|---|
| Q1 | Does reducing the input from 512 to 384 px significantly reduce training cost? | Benchmark (cost) + runs A and B (quality) | Cost: observed. Quality: H1, H2 |
| Q2 | Does MobileNetV3 offer a meaningful efficiency advantage over ResNet-50? | Benchmark only | Cost observed; quality **not tested** (§4.3) |
| Q3 | Does FPN v2 provide a measurable advantage over v1? | Benchmark only | Cost observed; quality **not tested** (§4.3) |
| Q4 | Does batch size 8 improve throughput on a GTX 1650? | Benchmark | Observed: no (§5.5) |
| Q5 | What is the best accuracy / speed / memory trade-off? | Runs A and B + benchmark | Only between A and B (§8) |

**Observations (§5, float32, not hypotheses):** at 384 px the training throughput is
1.38 times that at 512 px (6.57 ± 0.04 vs 4.75 ± 0.19 images/s, 3 repetitions each) and
the peak allocated memory 27 % lower (1.19 vs 1.63 GB). In single float32 runs,
MobileNetV3-Large FPN trains 4.7 times faster (22.4 images/s, 0.75 GB), FPN v2 is 1.7
times slower (2.75 images/s, 3.54 GB) and batch 8 is slower than batch 4 (3.64 images/s,
3.43 GB) (§5.5). Float16 mixed precision is not usable on this GPU (non-finite loss from
the first step, §5.3).

**Pre-registered hypotheses (quality, not yet measured):**

- **H1 (primary).** Model B (384 px) is **non-inferior** to model A (512 px) on the
  competition metric, margin 0.02 (absolute): lower bound of the 95 % CI of
  Δ = B − A above −0.02 (§9.1).
- **H2 (secondary).** B is non-inferior to A at image level: AUROC (margin 0.02),
  sensitivity and specificity (margin 0.05). MedVision QA criteria C1–C3 (§9.2).
- **H3 (exploratory, no decision attached).** The recall of small opacities drops more
  than the overall recall at 384 px.

The margins are a priori judgements, not derived from any data: 0.02 of the competition
metric is the largest loss we accept in exchange for a model about 1.4 times cheaper;
0.05 is the largest loss of sensitivity or specificity we accept. No clinical
requirement exists for this research model, so no absolute clinical threshold is set.

## 3. Data

### 3.1 Source and rules

RSNA Pneumonia Detection Challenge (Kaggle, stage 2 training set): 26,684 frontal chest
radiographs from NIH ChestX-ray14, 6,012 with at least one "lung opacity" box (9,555
boxes). Labels come from radiologists and are known to be noisy (readers disagree).
Competition rules: non-commercial use, no redistribution of the images. This repository
publishes code, metadata, hashes and predictions only. Research project, **not a
medical device**, no clinical validation.

### 3.2 Preparation (Step 2, commit `abfc675`, not changed by this protocol)

1. Every DICOM is checked to be 1024 × 1024, `MONOCHROME2`, **8-bit** pixels
   (`scripts/prepare_rsna.py` stops otherwise); no intensity rescaling is needed.
2. Reduced to 512 × 512 with `PIL.Image.resize(..., Image.Resampling.BOX)`: for this exact
   factor of 2, each output pixel is the **average of a 2 × 2 block** (area averaging,
   computed by PIL in fixed point: within 1 grey level of the exact mean). It is not a
   nearest-neighbour resize. Saved as 8-bit greyscale PNG.
3. Boxes multiplied by 0.5. Pixel `(r, c)` of the PNG covers input pixels
   `[2c, 2c + 2) × [2r, 2r + 2)`, so the box scaling is exact in the pixel-edge convention.
4. Split by **real patient**: the RSNA `patientId` identifies an image; the patient comes
   from the public RSNA-to-NIH mapping, checked image by image with the DICOM
   `SOPInstanceUID`. Stratified on "the patient has at least one positive image",
   70 / 10 / 20 %, seed 20261030, no patient in two sets.
   `metadata/split.csv`, SHA-256 `9ec285f5399c198c1d0cd23f1e5b3a9e9b99cee3ab18bccd473115fd4cb7040c`.

| Set | Images | With opacity | Boxes | Patients |
|---|---:|---:|---:|---:|
| train | 18,575 | 4,154 | 6,581 | 8,017 |
| val | 2,760 | 651 | 1,027 | 1,145 |
| test | 5,349 | 1,207 | 1,947 | 2,290 |

5. Duplicate audit (`metadata/data_audit.json`): 1 exact pixel duplicate pair (same
   patient, both in train). Near duplicates: 64-bit pHash of a 32 × 32 LANCZOS thumbnail,
   Hamming distance ≤ 4: 1,967 pairs, 65 of the same patient, 891 across splits. The 4
   closest cross-split pairs were checked by eye: different patients each time. At
   32 px, pHash does not discriminate chest radiographs well; the other 887 pairs are
   not checked, so some leakage cannot be excluded (§14).

### 3.3 Inputs and augmentation (`cxr/data.py`)

| Stage | Operation |
|---|---|
| Loading | 8-bit PNG → float in [0, 1], repeated on 3 channels |
| Training only | Horizontal flip, p = 0.5 |
| Training only | Affine, every image: rotation uniform in ±15°, **isotropic** scale uniform in 1 ± 0.05, translation uniform in ±25.6 px (5 % of 512) per axis, about the image centre; bilinear, black fill |
| Validation, test | None |
| Inside the model | ImageNet mean/std normalisation; resize to `min_size = max_size` (512: identity, checked; 384: bilinear 512 → 384, boxes predicted back in 512 px coordinates); batch padded to a multiple of 32 |

Boxes follow the image exactly (flip) or through the same affine map (envelope of the
4 transformed corners, clipped to the image). A box whose clipped width or height is
below 1 px is removed; the others keep their order and the labels are rebuilt from the
remaining boxes. A positive image can only lose all its boxes if an opacity is pushed
out of the frame, which the ranges above make very rare. Vertical flips are not used.
These rules are tested on synthetic images in `tests/test_data.py` (the previous,
uncommitted version transformed the boxes with the inverse map and swapped axes; it was
found and fixed before any training).

### 3.4 Training sampler

Each epoch: all 4,154 positive training images + 4,154 negatives = **8,308 images,
2,077 optimiser steps** at batch size 4. The negatives (14,421) are drawn without
replacement from a seeded permutation; when it is exhausted, a new permutation starts.
Validation and test always use **all** their images, at their natural prevalence.

## 4. Models

### 4.1 Common to all configurations

torchvision 0.29.1 `fasterrcnn_*` builders with COCO weights (`COCO_V1`), box predictor
replaced by a 2-class one (background, opacity). torchvision default
`trainable_backbone_layers = 3`: the stem (`conv1`) and the first stage (`layer1`) of
ResNet-50 are **frozen**, the rest is trained. `box_score_thresh = 0.01`,
`box_detections_per_img = 20`, `box_nms_thresh = 0.5` (default).

### 4.2 Configurations that receive final training

| Run | Architecture | Input | Batch | Change vs A |
|---|---|---|---|---|
| **A** (`frcnn-r50-fpn-512`) | `fasterrcnn_resnet50_fpn` (v1) | 512 px | 4 | reference |
| **B** (`frcnn-r50-fpn-384`) | `fasterrcnn_resnet50_fpn` (v1) | 384 px | 4 | input resolution only |

ResNet-50 FPN v1: frozen batch normalisation in the backbone (`FrozenBatchNorm2d`), FPN
without normalisation, RPN head with 1 convolution, box head of 2 fully connected
layers (`TwoMLPHead`); 41.30 M parameters with the 2-class head.

### 4.3 Configurations excluded from final training, and why

- **`fasterrcnn_resnet50_fpn_v2`.** Not a single change from v1. In torchvision 0.29.1 it
  uses trainable `BatchNorm2d` in the backbone and the FPN, a 2-convolution RPN head and
  a box head of 4 convolutions + batch norm + 1 fully connected layer (43.26 M
  parameters), and its COCO weights come from a different, longer "enhanced" training
  recipe (COCO box mAP 46.7 vs 37.0 for v1, according to the torchvision weight
  metadata). A difference between v1 and v2 could not be attributed to one cause, its
  trainable batch norm would run on batches of 4 images, and in float32 it is the
  slowest configuration (2.75 vs 4.75 images/s, about 8.4 h of training alone for 10
  epochs) with a peak memory of 3.54 GB, close to the 4 GB of the card (§5.5).
  **Q3 is answered on cost only.**
- **`fasterrcnn_mobilenet_v3_large_fpn`.** By far the cheapest (4.7 times the training
  throughput of R50 FPN v1 in float32, 0.75 GB), but the user chose two
  complementary runs over more runs. It also changes more than the backbone (torchvision
  builder defaults: other anchors, `rpn_score_thresh = 0.05`). **Q2 is answered on cost
  only;** whether it is accurate enough is left open.
- **Batch size 8.** In float32, slower than batch 4 (3.64 vs 4.75 images/s, single run)
  for 2.1 times the peak memory (3.43 vs 1.63 GB), and it would require a change of
  learning rate. Not used.

## 5. Computational benchmarks (done; not final training)

All measurements: GTX 1650 4 GB (TU117, 50 W power limit), torch 2.14.1+cu126, Windows
10, training images only, weights discarded, no detection metric computed. The raw
outputs are versioned in [`results/benchmarks/`](results/benchmarks/):
`benchmark.csv` (§5.2), `precision_benchmark.jsonl` (§5.3, one line per run, with the
loss curve), `profile_*.txt` and `precision_profile.jsonl` (profiled runs),
`precision_nan_check.jsonl` (§5.2), `benchmark_optim.csv` (§5.4, §5.5; it also holds
float16 rows of the optimisation tests, invalid for the same reason as §5.2).

### 5.1 First benchmark (`scripts/benchmark.py`, float16 mixed precision)

200 training images (100 with opacity, 100 without, seed 0, no augmentation), 4 data
workers, SGD (lr 0.005, momentum 0.9, weight decay 1e-4), float16 autocast + GradScaler.
One warm-up training pass (not timed), one timed training pass (50 optimiser steps at
batch 4, 25 at batch 8), one timed inference pass over the same 200 images. Throughput
includes data loading. Peak memory = `torch.cuda.max_memory_allocated`. One measurement
per configuration. The loss values were not recorded.

### 5.2 Results of the first benchmark — **invalid as training costs**

| Architecture | Input | Batch | Peak GB | Train img/s | Inference img/s |
|---|---:|---:|---:|---:|---:|
| R50 FPN v1 | 512 | 4 | 1.64 | 1.91 | 4.83 |
| R50 FPN v2 | 512 | 4 | 1.34 | 1.61 | 4.40 |
| R50 FPN v1 | 384 | 4 | 1.11 | 3.20 | 8.00 |
| MobileNetV3-L FPN | 512 | 4 | 0.70 | 10.11 | 21.61 |
| R50 FPN v1 | 512 | 8 | 2.85 | 1.95 | 4.85 |

These numbers were the basis of an earlier draft of this protocol (384 px "1.68 times
faster", a 21.8 h budget). The controlled study of §5.3 showed that float16 autocast
gives a **non-finite loss from the first step** on this GPU: GradScaler skips every
optimiser step, the weights never change and the model predicts no box. The table
therefore times a computation that does not train. It is kept for transparency and is
not used for any decision. Short checks (5 + 20 steps, `precision_nan_check.jsonl`)
give the same non-finite loss in float16 for R50 FPN v2 at 512 px and R50 FPN v1 at
384 px. MobileNetV3-Large FPN trains with finite losses in float16 (one step skipped
while the loss scale calibrates, parameters updated, boxes predicted) but is still
2.4 times slower than in float32 (9.5 vs 22.4 images/s). The failure is therefore tied
to the ResNet-50 models on this card, and float16 is slower for every model tried.

### 5.3 Float32 vs float16 autocast (`scripts/precision_benchmark.py`)

**Method.** R50 FPN v1, batch 4, one process per repetition, **3 repetitions** per
setting, settings interleaved. Identical in every run: initial weights
(`torch.manual_seed(20261030)` before building the model), image list and order (920
training images, half with opacity, seeded), augmentation of §3.3 (seeded), SGD as above,
4 data workers. Per run: 30 warm-up steps (not timed), then **200 timed steps**
(800 images) with `torch.cuda.synchronize()` before and after, then one timed inference
pass over 400 of these images in a new data loader (its timing includes the start of 4
data workers, so it is a conservative inference speed). Recorded per step: total loss
and its 4 components, gradient norm (after `GradScaler.unscale_` with AMP), time waiting
for data; per run: steps skipped by GradScaler, final loss scale, norm of the parameter
change of the box classifier and of one `layer4` convolution, finiteness of all
parameters, number of predicted boxes, `max_memory_allocated` and
`max_memory_reserved`, and `nvidia-smi` sampled every 500 ms (utilisation, SM clock,
temperature, power). Separate profiled runs (`torch.profiler`, CUDA kernels) are not
used for timing.

**Results at 512 px, cuDNN autotuning off** (mean ± SD of 3 runs; individual values in
brackets):

| Precision | Train img/s | Inference img/s | Peak allocated / reserved GB | Loss, last 50 steps | Skipped steps | Boxes predicted (400 images) |
|---|---:|---:|---:|---:|---:|---:|
| float32 | **4.75 ± 0.19** [4.79, 4.54, 4.92] | 3.95 ± 0.22 [3.95, 4.18, 3.73] | 1.63 / 2.07 | 0.181 ± 0.001 | 0 / 230 | 7,682–7,989 |
| float16 autocast | 1.91 ± 0.02 [1.89, 1.91, 1.92] | 3.14 ± 0.26 [2.85, 3.22, 3.36] | 1.65 / 1.79 | **NaN** | **230 / 230** | **0** |

With cuDNN autotuning on, the same picture: float32 5.02 ± 0.18 images/s, float16
2.13 ± 0.04 images/s with NaN loss and 230 / 230 skipped steps.

**Correctness.** float32: every loss and gradient finite in all 3 runs, parameters
updated (norm of the change over the timed steps 0.10–0.11 for the box classifier,
0.15–0.16 for the `layer4` convolution), loss falling from 0.270 ± 0.002 (first 10
steps) to 0.181 ± 0.001 (last 50), consistent across runs (median gradient norm
1.32 ± 0.01). float16 autocast: the loss and its four components are **NaN from the
first step** in all runs; GradScaler skips every step and halves the scale each time
down to 0; the parameters stay finite but never change (update norm 0.0); the model
outputs no box. The NaN appears in the forward pass, before any gradient, so it is not
a gradient overflow that a loss scale could absorb.

**Speed and memory.** float16 autocast is also 2.5 times slower (1.91 vs 4.75 images/s)
for the same allocated memory (1.65 vs 1.63 GB), at 97 % GPU utilisation. The CUDA
profile shows why: with autocast, convolutions run float16 implicit-GEMM kernels
(`sm75_xmma_*_f16f16_*`) that take 11.7 s of convolution backward over 10 profiled
steps, against 2.3 s for the float32 kernels over the same 10 steps.

**Inference timing caveat.** The inference passes above include the start of the data
workers in a new loader. With warm workers, `scripts/benchmark.py` measured 8.0–9.1
images/s in float32 at 512 px (4 single runs, 2 of them with cuDNN autotuning) and
9.9–10.7 at 384 px (3 single runs, 2 with autotuning).
The validation of §6 starts its workers at every epoch, so §11 uses the conservative
numbers of the table.

**Why this depends on the hardware.** The GTX 1650 (TU117) is a Turing GPU **without
tensor cores**; the float16 speed-ups usually reported for mixed precision come from
tensor cores (Volta, Turing RTX, Ampere and later). On this card the float16 kernels
chosen by cuDNN are slower, and the forward pass of this model produces non-finite
values in float16. Problems with float16 on GTX 16xx cards are widely reported by users
of other software; the exact cause here (a kernel, or an overflow in a specific layer)
was not investigated further, because float32 is both correct and faster. On a GPU with
tensor cores the conclusion would likely differ and must be measured again.

**Decision: float32**, no autocast, no GradScaler, for training and inference.

### 5.4 Other optimisations (float32, 512 px unless stated)

| Option | Measured | Decision |
|---|---|---|
| `torch.backends.cudnn.benchmark = True` | 512 px: 5.02 ± 0.18 vs 4.75 ± 0.19 images/s (+6 %, 3 runs each); 384 px: 6.68 ± 0.01 vs 6.57 ± 0.04 (+2 %); peak allocated memory **3.28 vs 1.63 GB** at 512 px (3.08 vs 1.19 GB at 384 px) | **Off.** A gain of about one standard deviation does not justify doubling the peak memory on a 4 GB card shared with the display, nor a less reproducible choice of algorithms |
| `channels_last` memory format | measured only under float16 (§5.2 conditions): 1.91 vs 1.78 images/s, one run each, within run-to-run spread | **Off** (no measured gain; not repeated in float32) |
| Trainable backbone stages (default 3; 0 and 5 tried) | measured only under float16, therefore invalid | **Default kept** (torchvision recipe): freezing more would change the model, not only its cost |
| Resolution 384 px | 6.57 ± 0.04 vs 4.75 ± 0.19 images/s, inference 5.37 ± 0.13 vs 3.95 ± 0.22 images/s, peak allocated 1.19 vs 1.63 GB | Trained as run B (§4.2): the question is its quality |
| Data loading | time waiting for data over 200 steps: 1.1 ± 0.8 s at 512 px, 0.5 ± 0.1 s at 384 px (under 1 % of the step time); GPU utilisation 90–91 % | Not a bottleneck; 4 workers kept |
| Image conversions | PNG decoded once per sample, uint8 → float32 on the CPU workers; resize to 384 px inside the model on the GPU | No change needed (data loading is not the bottleneck) |

### 5.5 Excluded configurations, re-measured in float32

`scripts/benchmark.py --fp32`, method of §5.1 except the precision (warm workers for the
inference pass, so its inference column is not comparable with §5.3), **single runs**.
The R50 FPN v1 row is the mean of §5.3.

| Architecture | Input | Batch | Peak allocated GB | Train img/s | Inference img/s | Train min / balanced epoch |
|---|---:|---:|---:|---:|---:|---:|
| R50 FPN v1 (§5.3, 3 runs) | 512 | 4 | 1.63 | 4.75 | — | 29.2 |
| R50 FPN v2 | 512 | 4 | 3.54 | 2.75 | 0.59 | 50.3 |
| MobileNetV3-L FPN | 512 | 4 | 0.75 | 22.40 | 38.63 | 6.2 |
| R50 FPN v1 | 512 | 8 | 3.43 | 3.64 | 6.94 | 38.0 |

The inference speed of FPN v2 (0.59 images/s) is far below its training speed. The run
was close to the memory limit of the card (3.9 of 4 GB used, display included), which
probably explains it; this was not investigated, because it does not change any
decision. These rows answer Q2–Q4 on cost only (§4.3).

## 6. Final training (Stage 4)

| Item | Value (identical for A and B) |
|---|---|
| Initialisation | COCO weights (§4.1), new 2-class predictor |
| Optimiser | SGD, momentum 0.9, weight decay 1e-4 |
| Learning rate | 0.005 (the torchvision reference 0.02 for 16 images, scaled linearly to 4); linear warm-up from 0.001 × lr over the first 500 steps |
| LR schedule | `ReduceLROnPlateau(mode="max", factor=0.1, patience=1)` on the monitored metric |
| Batch size | 4 |
| Precision | **float32** (no autocast, no GradScaler; float16 gives non-finite losses on this GPU, §5.3); `cudnn.benchmark = False` (§5.4) |
| Gradient clipping | none (as in the torchvision reference recipe and the benchmark) |
| Epoch | balanced, 8,308 images, 2,077 steps (§3.4), augmentation of §3.3 |
| Maximum budget | **10 epochs** |
| Validation | after every epoch, all 2,760 validation images, no augmentation |
| Monitored metric | competition metric on validation at the best threshold τ of the grid (§7.2) |
| Early stopping | stop after **3** consecutive epochs without a strict increase of the monitored metric |
| Kept checkpoint | epoch with the highest monitored metric (earliest on a tie), with its τ |
| Seed | 20261030 for Python, NumPy, PyTorch, sampler and augmentation (rng per image = (seed, epoch, index)) |
| Order | B first (cheaper: a failure of the new training script costs less), then A |

With these patiences the learning rate is divided by 10 after 2 epochs without
improvement, and training stops after 3. If the best epoch is the 10th (still
improving), the run is reported as **not converged within the budget**; the budget is
not extended. A run whose loss becomes non-finite is stopped at that step and reported;
a restart requires an amendment (§13).

**Interruptions.** After every epoch the script saves the full state (`last.pt`:
weights, optimiser, LR scheduler, early-stopping state, epoch, PyTorch RNG states). If
the session stops (crash, power), the run is resumed from the last completed epoch with
`--resume`; the interrupted epoch is restarted from its beginning with the same data
order and augmentation. A resume is logged and reported; it is not an amendment. The
machine is kept awake during training (`SetThreadExecutionState`).

The epoch number reaches the data workers through the sampler index (with
`persistent_workers=True`, workers keep their own copy of the dataset, so an attribute
set in the main process would not reach them and every epoch would repeat the same
augmentations); tested in `tests/test_training.py`.

## 7. Evaluation

### 7.1 Competition metric (primary)

`cxr/metric.py`, reproducing the organisers' description (Shih et al., *Radiology: AI*
2019, arXiv:1811.08939); 9 tests. For each image and each IoU threshold t in
{0.40, 0.45, …, 0.75}, predictions are taken in decreasing confidence and each is matched
to the still-unmatched ground-truth box of highest IoU if that IoU is **> t**; image
score at t = TP / (TP + FP + FN); image score = mean over the 8 thresholds; dataset
score = mean over the images that have at least one ground-truth or predicted box (an
image with neither is excluded; predictions on an image without opacity score 0). Not
the COCO mAP. The metric has no threshold of its own: only boxes with score ≥ τ are
passed to it.

### 7.2 Threshold τ (validation only)

For each epoch and model, τ maximises the validation competition metric over the grid
{0.01, 0.05, 0.10, 0.15, …, 0.95} (20 values; on a tie, the highest τ). The τ of the kept
checkpoint is frozen and used unchanged on the test split. It is also the image-level
operating point: an image is called positive if it has at least one box with score ≥ τ.

### 7.3 Secondary metrics (validation, then test)

- Image level, at τ: sensitivity (= recall), specificity, precision (PPV), AUROC of the
  maximum box score (0 if no box).
- Box level, at τ, a prediction matching a ground-truth box at IoU > 0.4: recall,
  precision, and mean IoU of the matched pairs (localisation quality).
- **Small opacities:** box recall restricted to ground-truth boxes of area
  < 11,503 px² (512 px coordinates), the smallest third of the **training** boxes.
- Confidence intervals: 95 %, percentile bootstrap over **patients** (2,000 resamples,
  seed 20261030), paired when A and B are compared.

### 7.4 Image-level files for MedVision QA

MedVision QA v0.1 is a classification tool (no box metric, no decision threshold), so
each image becomes one row: `score_opacity = f(s)` where `s` is the maximum box score
(0 if none) and `f` is piecewise linear, strictly increasing, with
`f(τ) = 0.5`: `f(s) = 0.5·s/τ` below τ, `0.5 + 0.5·(s − τ)/(1 − τ)` above;
`score_no_opacity = 1 − f(s)`; `predicted_label = opacity` if `s ≥ τ`. AUROC is unchanged
by `f`, and the argmax rule of MedVision QA gives exactly the decision at τ (tie at 0.5
resolved to `opacity`, which the format allows). `ground_truth.csv` has the NIH patient
as `patient_id`, so MedVision QA resamples by patient. Tested in `tests/test_config.py`.

### 7.5 Computational measurements of the final runs

Logged per epoch and per run: training wall time, validation wall time, images
processed, optimiser steps, training and inference images/s, peak memory
(`max_memory_allocated` and `max_memory_reserved`), checkpoint saving time, total GPU
time. Inference speed is measured on the validation pass (batch 4, float32, including
the start of the data workers).

## 8. Model selection (validation only, end of Stage 4)

The final model is chosen on the **validation** split, before the test split is read.
Model B (384 px) is selected if **all three** conditions hold; otherwise model A is
selected:

1. competition metric: lower bound of the 95 % CI of Δ = B − A > −0.02;
2. image-level sensitivity at each model's τ: lower bound of the 95 % CI of Δ > −0.05;
3. small-opacity box recall: B − A ≥ −0.05 (point estimate).

Rationale: A is the reference and is kept unless B is shown not to be worse; when B is
not worse, its measured cost (1.38 times the training throughput, 1.36 times the
inference throughput, 27 % less peak memory in the benchmark of §5.3; the costs of the
final runs are reported too) decides. An uncertain comparison (interval
across the margin) keeps A. The other metrics of §7.3 and §7.5 are reported for both and
do not change the choice. Validation results also chose the checkpoint and τ, so they
are optimistic for both models; only the test split gives an unbiased estimate.

## 9. Single test evaluation (Stage 5)

Once the selection, both checkpoints, both τ and the evaluation code are frozen and
committed, the test split is evaluated **once**, for A and B in the same run. Nothing is
tuned afterwards, and the test results **cannot change the selection** of §8; a
disagreement between validation and test is reported as such.

### 9.1 Primary: competition metric, B vs A

Δ = B − A on the test split, paired patient bootstrap of §7.3. Same reading rule as
MedVision QA: lower bound > −0.02 → non-inferiority **satisfied**; upper bound < −0.02 →
**not satisfied**; otherwise **inconclusive**.

### 9.2 MedVision QA (`config.toml`)

A = `frcnn-r50-fpn-512`, B = `frcnn-r50-fpn-384`, image level, 95 % CI, 2,000 patient
bootstrap resamples, seed 20261030:

| Id | Criterion | Margin |
|---|---|---|
| C1 | AUROC of B not inferior to A | 0.02 |
| C2 | Sensitivity of B (at its τ) not inferior to A (at its τ) | 0.05 |
| C3 | Specificity of B (at its τ) not inferior to A (at its τ) | 0.05 |

The Version Verification Record is published whatever its decision (SATISFIED,
NOT_SATISFIED or INCONCLUSIVE).

### 9.3 Reported for the selected model

Competition metric, every metric of §7.3 with its 95 % CI, and the costs of §7.5.

## 10. Error analysis (test split, after §9, descriptive only)

Metrics of §7.3 for the selected model (and A vs B) by view (PA / AP), sex, age group
(≤ 20, 21–40, 41–60, 61–80, > 80), opacity size (terciles of the training boxes), number
of boxes per image, and, for false positives, the RSNA detailed class of negative images
("No Lung Opacity / Not Normal" vs "Normal"). Aggregate tables and figures only, no
redistribution of images. Nothing in this analysis changes the protocol or the
selection.

## 11. GPU budget

Computed from the float32 means of §5.3 (batch 4, cuDNN autotuning off), not from the
invalid float16 table: per epoch = balanced training (8,308 images at the training
speed) + full validation (2,760 images at the conservative inference speed, which
includes the start of the data workers). Checkpoint saving (about 0.17 GB for the
weights and 0.33 GB for the full state, per epoch) and the τ grid on the CPU add an estimated 1–2 min per epoch, not
measured yet.

| Run | Train img/s | Train min/epoch | Validation min | Min/epoch | 10 epochs (ceiling) | Earliest stop (4 epochs) |
|---|---:|---:|---:|---:|---:|---:|
| B, 384 px | 6.57 | 21.1 | 8.6 | 29.6 | 4.9 h | 2.0 h |
| A, 512 px | 4.75 | 29.2 | 11.6 | 40.8 | 6.8 h | 2.7 h |
| **Total** | | | | | **11.7 h** (about 12–13 h with checkpoints) | 4.7 h |

For comparison, the invalid float16 numbers had given 21.8 h. Sustained throughput over
several hours may be lower than over 200 steps (the GPU reaches 89 °C; its SM clock was
1,470–1,550 MHz in the runs of §5.3), so the real costs are logged and reported (§7.5);
a slower run is not shortened and a faster one is not lengthened.

Answers to the budget questions: (1) final training for A and B only; (2) the others are
excluded for the reasons of §4.3; (3) both resolutions are trained because the choice
between them must rest on detection metrics, not on speed; (4) at most 10 epochs each,
early stopping patience 3; (5) at most about 12–13 h of GPU time; (6) run A gives the
reference quality, run B tells whether training and inference about 1.4 times faster
cost detection quality (H1–H3).

## 12. Reproducibility

- Pinned environment: `requirements.txt` (torch 2.14.1+cu126, torchvision 0.29.1+cu126,
  MedVision QA at commit `97e83b3`), Python 3.12, Windows 10.
- Fixed and versioned: split (`metadata/split.csv`, SHA-256 above), raw-data manifest
  (`metadata/raw_manifest.csv`), seed 20261030, this protocol, `config.toml`.
- Each training log records: git commit, SHA-256 of this file and of `config.toml`,
  command line, library versions, GPU, seed, per-epoch metrics and costs, chosen τ,
  SHA-256 of the kept weights.
- GPU training is not bit-for-bit deterministic (cuDNN algorithm choice, atomic
  additions in RoIAlign backward); seeds fix the data order and augmentation, not
  every floating-point result. Each configuration is trained **once**, so the
  seed-to-seed variance is not measured (§14).

## 13. Deviations

Any change after this commit (bug fix, crash, budget) is written as a dated amendment at
the end of this file **before** the affected run, with its reason, and listed in the
results. Results are never removed because they are unfavourable.

## 14. Limitations declared before training

1. One run per configuration: seed variance unknown; small differences between A and B
   may be noise even when the bootstrap interval (which covers patient sampling only)
   is narrow.
2. 887 cross-split near-duplicate pairs are unchecked; leakage would inflate both
   validation and test scores.
3. Noisy labels (reader disagreement), single annotation per image.
4. τ and the checkpoint are chosen on the same validation split used for selection.
5. Accuracy of MobileNetV3 and FPN v2 is not measured; Q2 and Q3 are answered on cost
   only.
6. Throughputs come from short measurements (3 repetitions of 200 steps for the
   float32 / float16 comparison and for 384 vs 512 px, single runs for the other
   options); sustained throughput over hours may differ.
7. Float16 mixed precision is unusable on this GPU (§5.3). The finding is specific to
   this hardware (no tensor cores) and is not a statement about mixed precision in
   general.
8. No external chest X-ray pretraining; COCO initialisation only.
9. 384 px is produced by a bilinear resize inside the model (no antialiasing), from
   the 512 px PNG.
10. Not a medical device; no clinical validation; no claim of clinical performance.
