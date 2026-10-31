# Test results — single evaluation (PROTOCOL.md §9, §10)

Evaluated once at commit `afe4b6d1e44ee5d9e571c6aaebcbfdd6b3835411`. Test split: 5349 images, 2290 patients. 95 % CI: percentile bootstrap over patients, 2000 paired resamples, seed 20261030. τ from validation: A = 0.8, B = 0.8. Selected model (validation, before the test): **B (frcnn-r50-fpn-384)**.

## Primary: competition metric, non-inferiority of B (384 px) vs A (512 px), margin 0.02

Δ = 0.0137, 95 % CI [0.0040, 0.0238] → **SATISFIED**

## All metrics

| Metric | A | B | B - A |
|---|---|---|---|
| rsna_metric | 0.2053 [0.1785, 0.2340] | 0.2190 [0.1883, 0.2521] | 0.0137 [0.0040, 0.0238] |
| sensitivity | 0.6835 [0.6376, 0.7290] | 0.7117 [0.6654, 0.7533] | 0.0282 [0.0083, 0.0495] |
| specificity | 0.8725 [0.8553, 0.8880] | 0.8590 [0.8407, 0.8758] | -0.0135 [-0.0202, -0.0069] |
| precision_image | 0.6098 [0.5699, 0.6477] | 0.5953 [0.5563, 0.6359] | -0.0145 [-0.0276, -0.0003] |
| box_recall | 0.4910 [0.4424, 0.5419] | 0.5126 [0.4619, 0.5626] | 0.0216 [0.0059, 0.0377] |
| box_precision | 0.4890 [0.4430, 0.5352] | 0.4773 [0.4311, 0.5259] | -0.0117 [-0.0248, 0.0015] |
| mean_iou_matched | 0.6610 [0.6501, 0.6708] | 0.6742 [0.6614, 0.6867] | 0.0132 [0.0054, 0.0200] |
| small_box_recall | 0.1814 [0.1476, 0.2149] | 0.1756 [0.1472, 0.2055] | -0.0058 [-0.0299, 0.0191] |
| auroc_image | 0.8883 [0.8749, 0.9028] | 0.8889 [0.8753, 0.9031] | 0.0006 [-0.0028, 0.0038] |

## MedVision QA (C1–C3)

```
SATISFIED — all 3 criteria are satisfied
Report: C:\Users\jabie\Downloads\cxr-pneumonia-detection\reports\vvr_en.html
```

## Error analysis of the selected model (B), descriptive

### By view

| Group | Images | Competition metric | Sensitivity | Specificity |
|---|---|---|---|---|
| AP | 2433 | 0.2456 | 0.7912 | 0.7071 |
| PA | 2916 | 0.1316 | 0.4396 | 0.9451 |

### By sex

| Group | Images | Competition metric | Sensitivity | Specificity |
|---|---|---|---|---|
| F | 2396 | 0.2263 | 0.7283 | 0.8526 |
| M | 2953 | 0.2130 | 0.6987 | 0.8642 |

### By age

| Group | Images | Competition metric | Sensitivity | Specificity |
|---|---|---|---|---|
| 0-20 | 419 | 0.2206 | 0.6929 | 0.8938 |
| 21-40 | 1426 | 0.2793 | 0.7528 | 0.8602 |
| 41-60 | 2218 | 0.2056 | 0.7016 | 0.8592 |
| 61-80 | 1246 | 0.1710 | 0.6943 | 0.8440 |
| >80 | 40 | 0.0260 | 0.1667 | 0.9412 |

### By n_boxes

| Group | Images | Competition metric | Sensitivity | Specificity |
|---|---|---|---|---|
| 0 | 4142 | 0.0000 | nan | 0.8590 |
| 1 | 504 | 0.2466 | 0.5536 | nan |
| 2 | 669 | 0.3943 | 0.8206 | nan |
| 3 | 31 | 0.1293 | 0.9032 | nan |
| 4 | 3 | 0.0625 | 1.0000 | nan |

### Box recall by opacity size (training terciles)

| Size | Boxes | Recall |
|---|---|---|
| large | 668 | 0.8293 |
| medium | 590 | 0.5475 |
| small | 689 | 0.1756 |

### False-positive rate among images without opacity, by RSNA class

| Class | Images | FP rate |
|---|---|---|
| No Lung Opacity / Not Normal | 2414 | 0.2378 |
| Normal | 1728 | 0.0058 |
