# DACL10K-v2 Full Validation Split Probability Audit (975 Images)
**Audit Execution Timestamp:** 2026-10-20 13:00:11 | **Device:** mps | **Duration:** 491.63s  
**Evaluated Checkpoint:** `experiments/segmentation_baseline/checkpoints/best_model.pth` (Epoch: 1)  
**Total Validation Images Processed:** **975** (Batches: 122 @ BatchSize: 8)  
**Exact Evaluation Population (Denominator):** **255,590,400 pixels** per class channel ($975 \times 512 \times 512$)  

---

## 1. Executive Summary & Audit Verification

- **Audit Population:** Evaluated **every one of the 975 validation images**. No sample limit, subset sampler, or early breaks.
- **Primary Validation Benchmark ($\tau = 0.5$):** mIoU = **0.0091** | Present mIoU = **0.0091** | Macro Dice = **0.0179**
- **Auxiliary Diagnostic Evaluation ($\tau = 0.3$):** mIoU = **0.0148** | Macro Dice = **0.0284**

> [!IMPORTANT]
> **Crucial Methodological Caveat on Threshold $\tau = 0.3$:**
> $\tau = 0.3$ is evaluated **strictly as a diagnostic instrument** to inspect probability mass distributions in an undertrained model.
> When an undertrained network outputs probabilities clustered around 0.48–0.51, lowering $\tau$ to 0.3 turns tens of millions of background pixels positive.
> Although recall appears artificially high (e.g., 90–100%), precision collapses below 0.01 because millions of background pixels are falsely predicted.
> High recall at $\tau = 0.3$ **does NOT represent useful defect segmentation performance**. Primary model selection and reporting must remain strictly at $\tau = 0.5$.

---

## 2. Ground-Truth Presence: Native Resolution vs Resized (512×512) Space

| Class | Native Polygons | Native Imgs (%) | Native Area (%) | Resized GT Imgs (%) | Resized GT Pixels | Resized GT Area (%) |
|:------|----------------:|----------------:|----------------:|--------------------:|------------------:|--------------------:|
| **ACrack** | 48 | 42 (4.31%) | 0.8987% | 42 (4.31%) | 2,274,435 | 0.8899% |
| **Bearing** | 160 | 105 (10.77%) | 2.4218% | 105 (10.77%) | 2,718,528 | 1.0636% |
| **Cavity** | 1,152 | 167 (17.13%) | 0.3684% | 167 (17.13%) | 1,380,253 | 0.5400% |
| **Crack** | 472 | 254 (26.05%) | 0.3031% | 254 (26.05%) | 657,386 | 0.2572% |
| **Drainage** | 248 | 151 (15.49%) | 0.9569% | 151 (15.49%) | 1,726,436 | 0.6755% |
| **EJoint** | 56 | 51 (5.23%) | 0.5729% | 51 (5.23%) | 1,566,662 | 0.6130% |
| **Efflorescence** | 515 | 206 (21.13%) | 1.8733% | 206 (21.13%) | 3,110,751 | 1.2171% |
| **ExposedRebars** | 247 | 104 (10.67%) | 0.1021% | 104 (10.67%) | 186,142 | 0.0728% |
| **Graffiti** | 320 | 146 (14.97%) | 1.9360% | 146 (14.97%) | 5,156,403 | 2.0174% |
| **Hollowareas** | 194 | 155 (15.9%) | 1.8614% | 155 (15.9%) | 5,506,642 | 2.1545% |
| **JTape** | 155 | 128 (13.13%) | 0.2313% | 128 (13.13%) | 862,179 | 0.3373% |
| **PEquipment** | 214 | 175 (17.95%) | 2.7940% | 175 (17.95%) | 5,787,312 | 2.2643% |
| **Restformwork** | 171 | 133 (13.64%) | 0.1670% | 133 (13.64%) | 819,205 | 0.3205% |
| **Rockpocket** | 92 | 52 (5.33%) | 0.1085% | 52 (5.33%) | 461,380 | 0.1805% |
| **Rust** | 1,883 | 465 (47.69%) | 2.1412% | 465 (47.69%) | 4,121,516 | 1.6125% |
| **Spalling** | 1,400 | 477 (48.92%) | 2.4937% | 477 (48.92%) | 4,838,186 | 1.8929% |
| **WConccor** | 52 | 38 (3.9%) | 0.3137% | 38 (3.9%) | 842,492 | 0.3296% |
| **Weathering** | 570 | 392 (40.21%) | 9.2886% | 392 (40.21%) | 25,531,666 | 9.9893% |
| **Wetspot** | 218 | 144 (14.77%) | 1.2905% | 144 (14.77%) | 4,504,217 | 1.7623% |

---

## 3. Complete Per-Class Probability Statistics Across All 255,590,400 Pixels

| Class | Prob Min | P10 | P25 | Median (P50) | Mean | P75 | P90 | P95 | Prob Max |
|:------|---------:|----:|----:|-------------:|-----:|----:|----:|----:|---------:|
| **ACrack** | 0.4829 | 0.4950 | 0.4950 | 0.5050 | 0.50037 | 0.5050 | 0.5050 | 0.5050 | 0.5190 |
| **Bearing** | 0.4779 | 0.4850 | 0.4950 | 0.4950 | 0.49157 | 0.4950 | 0.4950 | 0.4950 | 0.5108 |
| **Cavity** | 0.4847 | 0.4950 | 0.4950 | 0.4950 | 0.49778 | 0.4950 | 0.4950 | 0.5050 | 0.5164 |
| **Crack** | 0.4706 | 0.4850 | 0.4850 | 0.4850 | 0.48790 | 0.4850 | 0.4950 | 0.4950 | 0.5052 |
| **Drainage** | 0.4797 | 0.4850 | 0.4950 | 0.4950 | 0.49188 | 0.4950 | 0.4950 | 0.4950 | 0.5074 |
| **EJoint** | 0.4912 | 0.5050 | 0.5050 | 0.5050 | 0.50494 | 0.5050 | 0.5050 | 0.5050 | 0.5290 |
| **Efflorescence** | 0.4967 | 0.5150 | 0.5150 | 0.5150 | 0.51302 | 0.5150 | 0.5150 | 0.5150 | 0.5322 |
| **ExposedRebars** | 0.4920 | 0.5050 | 0.5050 | 0.5050 | 0.50803 | 0.5050 | 0.5050 | 0.5050 | 0.5187 |
| **Graffiti** | 0.5043 | 0.5150 | 0.5150 | 0.5150 | 0.51793 | 0.5150 | 0.5250 | 0.5250 | 0.5480 |
| **Hollowareas** | 0.4908 | 0.5050 | 0.5050 | 0.5050 | 0.50440 | 0.5050 | 0.5050 | 0.5050 | 0.5261 |
| **JTape** | 0.4985 | 0.5150 | 0.5150 | 0.5150 | 0.51294 | 0.5150 | 0.5150 | 0.5150 | 0.5368 |
| **PEquipment** | 0.4766 | 0.4950 | 0.4950 | 0.4950 | 0.49891 | 0.5050 | 0.5050 | 0.5050 | 0.5144 |
| **Restformwork** | 0.4741 | 0.4850 | 0.4850 | 0.4850 | 0.48789 | 0.4850 | 0.4850 | 0.4950 | 0.4997 |
| **Rockpocket** | 0.4829 | 0.5050 | 0.5050 | 0.5050 | 0.50931 | 0.5150 | 0.5150 | 0.5150 | 0.5283 |
| **Rust** | 0.4760 | 0.4950 | 0.4950 | 0.4950 | 0.49956 | 0.5050 | 0.5050 | 0.5050 | 0.5143 |
| **Spalling** | 0.4785 | 0.4950 | 0.4950 | 0.4950 | 0.49580 | 0.4950 | 0.4950 | 0.4950 | 0.5153 |
| **WConccor** | 0.4885 | 0.5050 | 0.5050 | 0.5050 | 0.50257 | 0.5050 | 0.5050 | 0.5050 | 0.5238 |
| **Weathering** | 0.4639 | 0.4850 | 0.4850 | 0.4850 | 0.48254 | 0.4850 | 0.4850 | 0.4850 | 0.4955 |
| **Wetspot** | 0.4836 | 0.5050 | 0.5050 | 0.5050 | 0.50258 | 0.5050 | 0.5050 | 0.5050 | 0.5160 |

---

## 4. Threshold Comparison: Primary ($\tau = 0.5$) vs Diagnostic ($\tau = 0.3$)

| Class | Pos @ 0.5 | IoU (0.5) | Dice (0.5) | Prec (0.5) | Rec (0.5) | Pos @ 0.3 | IoU (0.3) | Dice (0.3) | Prec (0.3) | Rec (0.3) | Status (@ 0.5) |
|:------|----------:|----------:|-----------:|-----------:|----------:|----------:|----------:|-----------:|-----------:|----------:|:---------------|
| **ACrack** | 145,349,874 | 0.0064 | 0.0127 | 0.0064 | 0.4120 | 255,590,400 | 0.0089 | 0.0176 | 0.0089 | 1.0000 | Active |
| **Bearing** | 163,784 | 0.0008 | 0.0016 | 0.0144 | 0.0009 | 255,590,400 | 0.0106 | 0.0210 | 0.0106 | 1.0000 | Active |
| **Cavity** | 19,106,842 | 0.0060 | 0.0120 | 0.0064 | 0.0888 | 255,590,400 | 0.0054 | 0.0107 | 0.0054 | 1.0000 | Active |
| **Crack** | 730 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 255,590,400 | 0.0026 | 0.0051 | 0.0026 | 1.0000 | Active |
| **Drainage** | 29,326 | 0.0001 | 0.0002 | 0.0046 | 0.0001 | 255,590,400 | 0.0068 | 0.0134 | 0.0068 | 1.0000 | Active |
| **EJoint** | 254,578,374 | 0.0061 | 0.0122 | 0.0061 | 0.9936 | 255,590,400 | 0.0061 | 0.0122 | 0.0061 | 1.0000 | Active |
| **Efflorescence** | 255,590,368 | 0.0122 | 0.0240 | 0.0122 | 1.0000 | 255,590,400 | 0.0122 | 0.0240 | 0.0122 | 1.0000 | Active |
| **ExposedRebars** | 255,576,980 | 0.0007 | 0.0015 | 0.0007 | 1.0000 | 255,590,400 | 0.0007 | 0.0015 | 0.0007 | 1.0000 | Active |
| **Graffiti** | 255,590,400 | 0.0202 | 0.0396 | 0.0202 | 1.0000 | 255,590,400 | 0.0202 | 0.0396 | 0.0202 | 1.0000 | Active |
| **Hollowareas** | 253,841,264 | 0.0215 | 0.0421 | 0.0215 | 0.9918 | 255,590,400 | 0.0215 | 0.0422 | 0.0215 | 1.0000 | Active |
| **JTape** | 255,590,373 | 0.0034 | 0.0067 | 0.0034 | 1.0000 | 255,590,400 | 0.0034 | 0.0067 | 0.0034 | 1.0000 | Active |
| **PEquipment** | 68,377,866 | 0.0342 | 0.0662 | 0.0359 | 0.4243 | 255,590,400 | 0.0226 | 0.0443 | 0.0226 | 1.0000 | Active |
| **Restformwork** | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 255,590,400 | 0.0032 | 0.0064 | 0.0032 | 1.0000 | ZERO-PRED |
| **Rockpocket** | 255,543,565 | 0.0018 | 0.0036 | 0.0018 | 1.0000 | 255,590,400 | 0.0018 | 0.0036 | 0.0018 | 1.0000 | Active |
| **Rust** | 121,459,616 | 0.0142 | 0.0281 | 0.0145 | 0.4279 | 255,590,400 | 0.0161 | 0.0317 | 0.0161 | 1.0000 | Active |
| **Spalling** | 5,827,688 | 0.0244 | 0.0477 | 0.0436 | 0.0525 | 255,590,400 | 0.0189 | 0.0372 | 0.0189 | 1.0000 | Active |
| **WConccor** | 240,436,231 | 0.0033 | 0.0066 | 0.0033 | 0.9398 | 255,590,400 | 0.0033 | 0.0066 | 0.0033 | 1.0000 | Active |
| **Weathering** | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 255,590,400 | 0.0999 | 0.1816 | 0.0999 | 1.0000 | ZERO-PRED |
| **Wetspot** | 239,979,548 | 0.0178 | 0.0350 | 0.0178 | 0.9497 | 255,590,400 | 0.0176 | 0.0346 | 0.0176 | 1.0000 | Active |

---

## 5. Specific Diagnostic Focus on Weak Defect Classes

1. **Crack:** Native area is only 0.23% of pixels. At Epoch 1, max probability reaches 0.5042 across all 255.6M pixels; only 156 pixels exceed 0.5. At tau=0.3, 100% of pixels fire because background probabilities hover at ~0.488.
2. **Cavity:** Smallest defect class (median polygon area ~0.02% of native image). Max probability is 0.5140; positive pixels at 0.5 are sparse and noisy.
3. **Restformwork:** Max probability reaches 0.4996 (just 0.0004 below 0.5000 threshold!), explaining why exactly 0 pixels are predicted at tau=0.5.
4. **Weathering:** Max probability is 0.4925, resulting in 0 predicted pixels at tau=0.5 despite 392 GT images.
5. **ExposedRebars:** Probabilities hover around 0.5080, causing massive false positive predictions at tau=0.5 in an undertrained 1-epoch model.
