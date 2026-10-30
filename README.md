# cxr-pneumonia-detection

[日本語](README.ja.md)

Lung-opacity **detection** on real chest X-rays (DICOM) from the
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge),
with two model versions compared under a protocol fixed **before** training
([`PROTOCOL.md`](PROTOCOL.md)) and verified with [MedVision QA](https://github.com/MousJB/medvision-qa).

> **No model has been trained yet; this repository reports no detection result.** A result
> will only appear once it has been produced by the code in this repository. Research
> project, **not a medical device**, no clinical validation.

## Status

| Stage | State |
|---|---|
| 1. Setup, download, data manifest | done |
| 2. Preparation (DICOM → PNG, patient-level split, duplicate audit, competition metric) | done (commit `abfc675`) |
| 3. Computational benchmark and pre-registered protocol ([`PROTOCOL.md`](PROTOCOL.md), [`config.toml`](config.toml)) | done |
| 4. Final training of A (512 px) and B (384 px), selection on validation | not started |
| 5. Single evaluation on the test split, MedVision QA, error analysis | not started |
| 6. Results, limitations, publication | not started |

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
| `scripts/` | download, preparation, GPU benchmarks (`benchmark.py`, `precision_benchmark.py`), training (`train.py`), evaluation (`evaluate.py`) |
| `metadata/` | manifests, split, duplicate audit (byte-exact, hashed) |
| `PROTOCOL.md`, `config.toml` | pre-registered protocol and MedVision QA configuration |

## Licence

Code: see `LICENSE`. Data: RSNA Pneumonia Detection Challenge, competition rules
(non-commercial use, no redistribution of the images).
