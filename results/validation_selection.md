# Validation results and model selection (PROTOCOL.md §8)

Validation split: 2760 images, 1145 patients. 95 % CI: percentile bootstrap over patients, 2000 paired resamples, seed 20261030. Each model at its own τ.

| Run | Best epoch / run | τ | Stopped early | Best within budget | GPU h | Train img/s | Peak GB |
|---|---|---|---|---|---|---|---|
| A (frcnn-r50-fpn-512) | 9 / 10 | 0.8 | False | True | 5.76 | 4.82 | 1.63 |
| B (frcnn-r50-fpn-384) | 8 / 10 | 0.8 | False | True | 4.39 | 6.38 | 1.19 |

| Metric | A | B | B - A |
|---|---|---|---|
| rsna_metric | 0.2272 [0.1976, 0.2561] | 0.2322 [0.2029, 0.2597] | 0.0050 [-0.0094, 0.0174] |
| sensitivity | 0.7281 [0.6696, 0.7778] | 0.7542 [0.6945, 0.8028] | 0.0261 [0.0000, 0.0508] |
| specificity | 0.8696 [0.8389, 0.8963] | 0.8596 [0.8294, 0.8864] | -0.0100 [-0.0191, -0.0009] |
| precision_image | 0.6328 [0.5891, 0.6726] | 0.6239 [0.5801, 0.6632] | -0.0090 [-0.0268, 0.0075] |
| box_recall | 0.5511 [0.5000, 0.5976] | 0.5696 [0.5162, 0.6158] | 0.0185 [-0.0022, 0.0370] |
| box_precision | 0.4948 [0.4412, 0.5444] | 0.4920 [0.4419, 0.5377] | -0.0027 [-0.0255, 0.0162] |
| mean_iou_matched | 0.6602 [0.6485, 0.6729] | 0.6677 [0.6563, 0.6803] | 0.0075 [-0.0018, 0.0160] |
| small_box_recall | 0.2533 [0.1917, 0.3173] | 0.2368 [0.1840, 0.2925] | -0.0164 [-0.0598, 0.0287] |
| auroc_image | 0.8927 [0.8759, 0.9074] | 0.8941 [0.8775, 0.9088] | 0.0014 [-0.0035, 0.0062] |

Selection rule:

- condition_1_rsna_lower_bound_gt_-0.02: True
- condition_2_sensitivity_lower_bound_gt_-0.05: True
- condition_3_small_box_recall_diff_ge_-0.05: True
- selected: B

**Selected model: B (frcnn-r50-fpn-384)**
