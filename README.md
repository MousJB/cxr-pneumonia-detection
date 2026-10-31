# cxr-pneumonia-detection

[日本語](README.ja.md)

Lung-opacity **detection** on real chest X-rays (DICOM) from the
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge),
with two model versions compared under a protocol fixed **before** training
([`PROTOCOL.md`](PROTOCOL.md)) and verified with [MedVision QA](https://github.com/MousJB/medvision-qa).

> **Result (test split, evaluated once):** the 384 px model (B) is **non-inferior** to the
> 512 px model (A) on the competition metric (0.219 vs 0.205, Δ = +0.014, 95 % CI
> [+0.004, +0.024], margin 0.02) and on the three MedVision QA criteria, for 1.3 times
> fewer GPU hours. One training run per model on a GTX 1650. Research project, **not a
> medical device**, no clinical validation.

## Status

| Stage | State |
|---|---|
| 1. Setup, download, data manifest | done |
| 2. Preparation (DICOM → PNG, patient-level split, duplicate audit, competition metric) | done (commit `abfc675`) |
| 3. Computational benchmark and pre-registered protocol ([`PROTOCOL.md`](PROTOCOL.md), [`config.toml`](config.toml)) | done (commit `d4aa648`) |
| 4. Final training of A (512 px) and B (384 px), selection on validation | done (commit `afe4b6d`): B selected |
| 5. Single evaluation on the test split, MedVision QA, error analysis | done |
| 6. Results, limitations, publication | results below; MedVision QA demo integration not done |

## Results

All numbers come from [`results/`](results/) and [`reports/`](reports/), produced by
`scripts/train.py` and `scripts/evaluate.py`. 95 % confidence intervals: percentile
bootstrap over patients (2,000 paired resamples, seed 20261030). They cover patient
sampling only, not seed variance (one run per model).

### Training and selection (validation split, 2,760 images, 1,145 patients)

| Run | Epochs | Best epoch | τ | Validation competition metric | GPU h | Train img/s | Val img/s | Peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A, 512 px | 10 | 9 | 0.80 | 0.2272 [0.1976, 0.2561] | 5.76 | 4.82 | 8.01 | 1.63 |
| B, 384 px | 10 | 8 | 0.80 | 0.2322 [0.2029, 0.2597] | 4.39 | 6.38 | 10.19 | 1.19 |

Both runs reached their best epoch before the 10th (converged within the budget, as
defined in `PROTOCOL.md` §6). No non-finite loss, no skipped step. B satisfies the three
selection conditions of §8 (competition metric Δ lower bound −0.009 > −0.02; sensitivity
Δ lower bound 0.000 > −0.05; small-opacity recall Δ −0.016 ≥ −0.05), so **B was selected
before the test split was read**: [`results/validation_selection.md`](results/validation_selection.md).
Training logs: `results/training_log_*.jsonl`.

![Validation competition metric per epoch, A and B](results/figures/validation_curves.svg)

Run A was interrupted once, during epoch 7: Windows ran out of commit memory (error
1455) in a data-loading worker. It was resumed from the state saved after epoch 6
(`--resume`, logged as a `resume` event) and epoch 7 was restarted from its beginning,
as `PROTOCOL.md` §6 provides. The GPU hours above count completed epochs only.

### Test split (5,349 images, 2,290 patients, evaluated once)

Full tables: [`results/test/test_results.md`](results/test/test_results.md).

![Test differences B minus A with 95 % confidence intervals and pre-registered margins](results/figures/test_differences.svg)

| Metric | A, 512 px | B, 384 px | B − A |
|---|---|---|---|
| Competition metric (primary) | 0.2053 [0.1785, 0.2340] | 0.2190 [0.1883, 0.2521] | +0.0137 [+0.0040, +0.0238] |
| Image AUROC | 0.8883 [0.8749, 0.9028] | 0.8889 [0.8753, 0.9031] | +0.0006 [−0.0028, +0.0038] |
| Sensitivity (image, at τ) | 0.6835 [0.6376, 0.7290] | 0.7117 [0.6654, 0.7533] | +0.0282 [+0.0083, +0.0495] |
| Specificity (image, at τ) | 0.8725 [0.8553, 0.8880] | 0.8590 [0.8407, 0.8758] | −0.0135 [−0.0202, −0.0069] |
| Box recall (IoU > 0.4) | 0.4910 [0.4424, 0.5419] | 0.5126 [0.4619, 0.5626] | +0.0216 [+0.0059, +0.0377] |
| Box precision | 0.4890 [0.4430, 0.5352] | 0.4773 [0.4311, 0.5259] | −0.0117 [−0.0248, +0.0015] |
| Small-opacity box recall | 0.1814 [0.1476, 0.2149] | 0.1756 [0.1472, 0.2055] | −0.0058 [−0.0299, +0.0191] |

- **Primary (§9.1):** non-inferiority of B, margin 0.02: **satisfied** (lower bound
  +0.004 > −0.02). The validation and test conclusions agree.
- **MedVision QA (§9.2):** C1 AUROC, C2 sensitivity, C3 specificity: **SATISFIED**, all
  three. Version Verification Records: [`reports/vvr_en.html`](reports/vvr_en.html),
  [`reports/vvr_ja.html`](reports/vvr_ja.html); input files in [`medvision/`](medvision/).
  MedVision QA runs its own patient bootstrap: same point estimates, interval bounds within
  0.004 of the ones above (0.002 for the differences B − A).
- At the same τ = 0.80, B flags more images than A: higher sensitivity, lower
  specificity (−1.4 points, interval excluding zero). This is a different operating
  point, not a better detector: the image AUROC is the same.
- Costs of B relative to A, measured on the final runs: 1.32 times the training
  throughput, 1.27 times the validation throughput, 27 % less peak GPU memory, 1.31 times
  fewer GPU hours.

### Error analysis of B (test, descriptive, `PROTOCOL.md` §10)

- **Opacity size:** box recall 0.83 for large opacities, 0.55 medium, **0.18 small**
  (terciles of the training boxes). Small opacities are the main weakness of both models
  (A: 0.18).
- **View:** AP images 0.246 (sensitivity 0.79, specificity 0.71); PA images 0.132
  (sensitivity 0.44, specificity 0.95). Opacities are four times more frequent on AP
  test images (38 %) than on PA (9 %), and the model flags AP images far more often.
- **False positives:** 23.8 % of the negative images labelled "No Lung Opacity / Not
  Normal" receive a box, against 0.6 % of the "Normal" images: 574 of the 584
  false-positive images show another abnormality.
- **Age:** 0.279 (21–40), 0.206 (41–60), 0.171 (61–80); the > 80 group has only 40
  images. **Sex:** F 0.226, M 0.213.
- Per-image competition metric of an image without opacity is 0 if any box is predicted
  and the image is left out otherwise, hence the 0.000 of the "0 boxes" row in the full
  table; specificity is the relevant measure for that group.

![Test box recall by opacity size, A and B](results/figures/test_recall_by_size.svg)

Figures: `scripts/make_figures.py`, from the committed result files.

### Limitations of these results

One run per model (seed variance unknown); τ and checkpoint chosen on the validation
split; 887 unchecked cross-split near-duplicate pairs; noisy single-reader labels; the
validation and test splits are our own patient-level partition of the Kaggle training
set, so the scores are **not comparable** to the Kaggle leaderboard, which used other
images. Complete list: `PROTOCOL.md` §14. Not a medical device; no claim of
clinical performance.

## Pre-registered plan (summary of `PROTOCOL.md`)

- **Models:** torchvision Faster R-CNN ResNet-50 FPN (v1), COCO weights, 2-class head.
  **A** at 512 px (reference), **B** at 384 px (only change). FPN v2, MobileNetV3 and
  batch size 8 were measured in the benchmark but are not trained (reasons in §4.3).
- **Training:** balanced epochs (all positives + as many negatives, 8,308 images),
  SGD lr 0.005, batch 4, float32, at most 10 epochs, early stopping patience 3 on the
  validation competition metric, seed 20261030.
- **Selection:** on validation only; B is chosen if it is non-inferior to A (competition
  metric margin 0.02, sensitivity margin 0.05, small-opacity recall).
- **Test:** evaluated once, after every decision is frozen; competition metric
  (non-inferiority of B, margin 0.02) and MedVision QA criteria C1–C3 (image-level AUROC,
  sensitivity, specificity).
- **GPU budget:** at most about 12–13 h on a GTX 1650 (4 GB), from measured float32
  throughputs.

### Computational benchmarks (training images, not a quality measurement)

Float16 mixed precision does **not** work on this GPU: with the ResNet-50 models the loss
is NaN from the first step (every optimiser step skipped, no box predicted), and it is
2.4–2.5 times slower than float32 for every model tried. The GTX 1650 has no tensor
cores. Training therefore runs in float32. An earlier float16 table, which timed a
computation that did not train, is kept in `PROTOCOL.md` §5.2 and marked invalid.

| Architecture | Input | Batch | Peak GB | Train img/s (float32) | Runs |
|---|---:|---:|---:|---:|---:|
| R50 FPN v1 (run A) | 512 | 4 | 1.63 | 4.75 ± 0.19 | 3 |
| R50 FPN v1 (run B) | 384 | 4 | 1.19 | 6.57 ± 0.04 | 3 |
| R50 FPN v2 | 512 | 4 | 3.54 | 2.75 | 1 |
| MobileNetV3-L FPN | 512 | 4 | 0.75 | 22.40 | 1 |
| R50 FPN v1 | 512 | 8 | 3.43 | 3.64 | 1 |

Method, correctness checks (finite losses, parameter updates), cuDNN autotuning (+6 %
for twice the memory: off), channels_last, data loading, and limits: `PROTOCOL.md` §5.

## Data

The images are **not** in this repository: the competition rules forbid redistribution.
Only code, metadata, file hashes and predictions are published. To reproduce, accept the
competition rules on Kaggle, create an API token (`~/.kaggle/kaggle.json`), then:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/download_rsna.py
.venv\Scripts\python scripts/prepare_rsna.py
```

`scripts/download_rsna.py` checks the volumes and writes `metadata/raw_manifest.csv`
(path, size and SHA-256 of every downloaded file) and `metadata/raw_summary.json`.
`scripts/prepare_rsna.py` writes the 512 px PNG and the box and image tables to `data/`
(ignored by git) and `metadata/split.csv` and `metadata/data_audit.json` (versioned).

### Step 2 summary (commit `abfc675`)

26,684 DICOM images (1024 px, 8-bit) reduced to 512 px PNG by 2 × 2 area averaging.
Patient-level split via the RSNA-to-NIH mapping, verified image by image with the DICOM
`SOPInstanceUID`. No patient is in more than one set.

| Set | Images | Positive | Patients |
|---|---:|---:|---:|
| train | 18,575 | 4,154 | 8,017 |
| val | 2,760 | 651 | 1,145 |
| test | 5,349 | 1,207 | 2,290 |

**Duplicate audit:** exact (SHA-256 of the pixels) and near-duplicate (64-bit pHash of a
32 px thumbnail, Hamming distance ≤ 4) pairs are listed in `metadata/data_audit.json`.
891 near-duplicate pairs cross split boundaries; the 4 closest were checked by eye — all
are different patients. At 32 px, pHash produces false near-duplicate pairs for chest
X-rays; the other 887 pairs are not checked.

## Tests

```
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest
```

`tests/test_metric.py`: the competition metric. `tests/test_data.py`: boxes follow the
image under every augmentation (synthetic rectangles). `tests/test_config.py`:
`config.toml` passes the MedVision QA validator and agrees with the protocol.
`tests/test_training.py`: balanced sampler, epoch reaching persistent data workers,
warm-up, early stopping, τ selection. `tests/test_evaluation.py`: box matching, image
and box metrics, AUROC, paired patient bootstrap, selection rule.

## Repository layout

| Path | Content |
|---|---|
| `cxr/data.py` | data loading, augmentation, torchvision dataset |
| `cxr/models.py` | Faster R-CNN builders (COCO weights, 2-class head) |
| `cxr/metric.py` | competition metric |
| `cxr/training.py`, `cxr/evaluation.py` | sampler, early stopping, τ selection; evaluation and bootstrap |
| `scripts/` | download, preparation, GPU benchmarks (`benchmark.py`, `precision_benchmark.py`), training (`train.py`), evaluation (`evaluate.py`), figures (`make_figures.py`) |
| `metadata/` | manifests, split, duplicate audit (byte-exact, hashed) |
| `PROTOCOL.md`, `config.toml` | pre-registered protocol and MedVision QA configuration |
| `results/` | benchmarks, training logs, validation selection, test results and box predictions |
| `medvision/`, `reports/` | MedVision QA input files (test split, image level) and Version Verification Records |

## Licence

Code: see `LICENSE`. Data: RSNA Pneumonia Detection Challenge, competition rules
(non-commercial use, no redistribution of the images).
