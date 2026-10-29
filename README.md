# cxr-pneumonia-detection

[日本語](README.ja.md)

Lung-opacity **detection** on real chest X-rays (DICOM) from the
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge),
with two model versions (v1, v2) compared under a protocol fixed **before** training and
evaluated with [MedVision QA](https://github.com/MousJB/medvision-qa).

> Work in progress. No result is reported until it has been produced by the code in this
> repository. Research project, **not a medical device**, no clinical validation.

## Status

| Step | State |
|---|---|
| 1. Setup, download, data manifest | in progress |
| 2. Preparation (DICOM → PNG, patient-level split, duplicate audit) | — |
| 3. Pre-registered protocol (`PROTOCOL.md`) | — |
| 4. v1 baseline | — |
| 5. v2 | — |
| 6. Single evaluation on the test split | — |
| 7. Results and limitations | — |

## Data

The images are **not** in this repository: the competition rules forbid redistribution.
Only code, metadata, file hashes and predictions are published. To reproduce, accept the
competition rules on Kaggle, create an API token (`~/.kaggle/kaggle.json`), then:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/download_rsna.py
```

`scripts/download_rsna.py` checks the volumes and writes `metadata/raw_manifest.csv`
(path, size and SHA-256 of every downloaded file) and `metadata/raw_summary.json`.

## Licence

Code: see `LICENSE`. Data: RSNA Pneumonia Detection Challenge, competition rules
(non-commercial use, no redistribution of the images).
