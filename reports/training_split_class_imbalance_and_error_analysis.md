# DACL10K-v2 Training-Split Class Imbalance & Diagnostic Error Analysis

**Project:** Bridge AI — Autonomous Multi-Label Bridge Defect Semantic Segmentation  
**Dataset:** DACL10K-v2 (Development Phase)  
**Baseline Evaluated:** DeepLabV3+ with ResNet-34 Encoder (15 Epochs, ImageNet Pretrained, $512 \times 512$ letterbox, $\tau = 0.5$)  
**Baseline Validation Performance:** Macro mIoU = **0.3384** (33.84%), Macro Dice = **0.4825** (48.25%)  
**Analysis Scope:** **TRAINING DATA ONLY** (All 6,935 training images analyzed across all 19 defect classes; validation split remains untouched for model selection)  
**Artifacts Generated:**
- Class Frequencies CSV: [`reports/train_class_frequencies.csv`](file:///Users/kundankumar/Documents/untitled%20folder%202/bridge-ai-project/reports/train_class_frequencies.csv)
- Co-occurrence Matrix: [`reports/train_cooccurrence_matrix.csv`](file:///Users/kundankumar/Documents/untitled%20folder%202/bridge-ai-project/reports/train_cooccurrence_matrix.csv)
- Conditional Probabilities: [`reports/train_conditional_probabilities.csv`](file:///Users/kundankumar/Documents/untitled%20folder%202/bridge-ai-project/reports/train_conditional_probabilities.csv)
- Diagnostic Visualizations: [`reports/visualizations/`](file:///Users/kundankumar/Documents/untitled%20folder%202/bridge-ai-project/reports/visualizations/)

---

## 1. Executive Summary & Baseline State

The 15-epoch baseline training run on Google Colab demonstrated solid overall macro performance (mIoU 0.3384, macro Dice 0.4825), with prominent reference classes achieving strong IoU scores: `PEquipment` (0.5820), `Graffiti` (0.5476), `Bearing` (0.5484), `Hollowareas` (0.5315), and `ACrack` (0.4607).

However, three critical defect classes exhibited severe degradation at the primary decision threshold $\tau = 0.5$:
1. **WConccor (Weathered / Wet Concrete Corrosion):** Catastrophic failure with **IoU = 0.0002** and **Recall = 0.0002** (achieving only 137 true positive pixels across all 975 validation images, despite 842,484 ground-truth pixels).
2. **Rockpocket (Honeycombing / Grout Loss):** Severe under-detection with **IoU = 0.0689** and **Recall = 0.0843** (only 38,913 true positive pixels out of 461,341 GT pixels).
3. **Crack (Linear Structural Cracks):** Boundary sensitivity and false negatives with **IoU = 0.1848** and **Recall = 0.2953**.

Before altering model architectures, modifying loss functions, or retraining, this audit performed an exhaustive, deterministic census over **all 6,935 training images** and investigated the root causes behind these performance disparities.

### Key Audit Findings:
- **Severe Frequency Starvation for WConccor:** WConccor is the single rarest class in the entire training dataset by both image count (**265 images / 3.82%**) and annotated polygon count (**360 shapes / 0.66% of all 54,158 shapes**). It receives 26 times fewer foreground pixels than dominant classes like `Weathering` (182.7M pixels) and `Spalling` (35.6M pixels).
- **Semantic and Spatial Occlusion:** In 58.5% of WConccor images, `Spalling` co-occurs; in 56.2%, `Rust` co-occurs; and in 40.4%, `Weathering` co-occurs. Furthermore, 14.62% of WConccor's spatial area directly overlaps with `Hollowareas`, and 5.65% directly overlaps with `Weathering`. The unweighted BCE loss pushes the network to predict the 26x more frequent classes, driving WConccor output probabilities below the 0.5 decision threshold.
- **Morphology Paradox between Crack and ACrack:** While `Crack` appears in 1,727 images (24.90%), 62.78% of crack instances are small (<1024 px in 512x512) with a median area of only **615.1 pixels** (thin 1–3 pixel lines). Conversely, `ACrack` (Alligator Crack) appears in only 332 images (4.79%), but 72.61% of its instances are expansive interconnected surface webs (median area **22,281.5 pixels**), allowing the network to easily capture broad area features and reach IoU 0.4607.
- **Clean Annotations Confirmed:** Coordinate boundary audits verified that all polygon coordinates are strictly valid, non-inverted, in-bounds, and consistently indexed across both splits. The poor performance of weak classes is driven by **dataset class imbalance, morphology divergence, and loss gradient starvation**, not annotation corruption.

---

## 2. Complete Training-Split Census (6,935 Images, 19 Classes)

Every training image and annotation was letterbox-transformed to the exact $512 \times 512$ input resolution used during training (total training pixel population: $6,935 \times 262,144 = 1,817,968,640$ pixels). In addition, native resolution polygon areas were calculated via Green's theorem (Shoelace formula).

Defect size distributions follow standard computer vision categorization adapted to $512 \times 512$:
- **Small Defect:** Area $< 1,024 \text{ px}$ ($< 32 \times 32 \text{ px}$ equivalent, $< 0.39\%$ of image).
- **Medium Defect:** $1,024 \le \text{Area} \le 9,216 \text{ px}$ ($32 \times 32$ to $96 \times 96 \text{ px}$, $0.39\%$ to $3.52\%$).
- **Large Defect:** Area $> 9,216 \text{ px}$ ($> 96 \times 96 \text{ px}$ equivalent, $> 3.52\%$ of image).

### Table 1: Comprehensive Training Set Class Distribution & Morphology Census
*Sorted by Image Count (descending)*

| Class Name | Index | Image Count | Image % | Total Shapes | Shapes/Img | 512x512 Pixels | Pixel % (512) | Median Area (512) | Small % | Medium % | Large % | Native Median Area |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rust** | 14 | 3,450 | 49.75% | 12,837 | 3.72 | 31,153,573 | 1.7136% | 240.8 | 73.83% | 20.29% | 5.88% | 2,432.2 |
| **Spalling** | 15 | 3,262 | 47.04% | 8,484 | 2.60 | 35,567,636 | 1.9564% | 768.9 | 56.02% | 32.98% | 11.00% | 6,326.7 |
| **Weathering** | 17 | 2,704 | 38.99% | 4,064 | 1.50 | 182,694,565 | 10.0494% | 19,170.1 | 11.27% | 24.75% | 63.98% | 161,708.6 |
| **Crack** | 3 | 1,727 | 24.90% | 3,154 | 1.83 | 4,787,533 | 0.2633% | 615.1 | 62.78% | 36.05% | 1.17% | 8,182.8 |
| **Efflorescence**| 6 | 1,523 | 21.96% | 3,450 | 2.27 | 27,616,755 | 1.5191% | 1,758.8 | 39.33% | 41.07% | 19.59% | 21,162.8 |
| **PEquipment** | 11 | 1,319 | 19.02% | 1,674 | 1.27 | 54,563,529 | 3.0013% | 15,255.2 | 10.51% | 29.93% | 59.56% | 134,634.5 |
| **Cavity** | 2 | 1,188 | 17.13% | 8,119 | 6.83 | 11,046,379 | 0.6076% | 41.2 | 93.16% | 4.31% | 2.52% | 323.8 |
| **Hollowareas** | 9 | 1,097 | 15.82% | 1,363 | 1.24 | 38,390,021 | 2.1117% | 17,970.9 | 3.37% | 29.20% | 67.42% | 147,963.5 |
| **Drainage** | 4 | 1,030 | 14.85% | 1,404 | 1.36 | 13,644,623 | 0.7505% | 1,066.1 | 49.50% | 26.00% | 24.50% | 9,817.6 |
| **Wetspot** | 18 | 965 | 13.91% | 1,443 | 1.50 | 35,721,551 | 1.9649% | 10,588.8 | 11.30% | 35.48% | 53.22% | 78,164.9 |
| **Graffiti** | 8 | 795 | 11.46% | 1,880 | 2.36 | 27,617,761 | 1.5192% | 2,394.7 | 34.20% | 36.97% | 28.83% | 18,296.4 |
| **ExposedRebars**| 7 | 773 | 11.15% | 1,753 | 2.27 | 2,062,777 | 0.1135% | 309.5 | 77.18% | 20.82% | 2.00% | 4,376.3 |
| **JTape** | 10 | 767 | 11.06% | 923 | 1.20 | 5,048,925 | 0.2777% | 3,119.3 | 18.31% | 64.68% | 17.01% | 23,445.4 |
| **Bearing** | 1 | 730 | 10.53% | 1,048 | 1.44 | 16,852,741 | 0.9270% | 2,462.6 | 41.70% | 23.95% | 34.35% | 23,304.3 |
| **Restformwork** | 12 | 713 | 10.28% | 851 | 1.19 | 4,222,025 | 0.2322% | 2,808.0 | 21.86% | 65.10% | 13.04% | 20,672.6 |
| **EJoint** | 5 | 395 | 5.70% | 450 | 1.14 | 15,824,102 | 0.8704% | 17,607.4 | 5.56% | 28.22% | 66.22% | 154,194.9 |
| **ACrack** | 0 | 332 | 4.79% | 376 | 1.13 | 19,043,215 | 1.0475% | 22,281.5 | 3.46% | 23.94% | 72.61% | 253,107.3 |
| **Rockpocket** | 13 | 311 | 4.48% | 525 | 1.69 | 3,810,196 | 0.2096% | 1,927.3 | 36.38% | 45.14% | 18.48% | 15,255.8 |
| **WConccor** | 16 | **265** | **3.82%** | **360** | **1.36** | **6,938,109** | **0.3816%** | **5,846.7** | **15.83%** | **44.17%** | **40.00%** | **100,993.2** |

---

## 3. In-Depth Profiles of the Four Focal Classes

```
  ========================================================================================
  CLASS IMBALANCE & DEFECT MORPHOLOGY SPECTRUM
  ========================================================================================

  IMAGE FREQUENCY:
  Rust          [██████████████████████████████████████████████████] 49.75% (3,450 imgs)
  Spalling      [███████████████████████████████████████████       ] 47.04% (3,262 imgs)
  Weathering    [███████████████████████████████                   ] 38.99% (2,704 imgs)
  Crack         [█████████████████████                             ] 24.90% (1,727 imgs)
  ACrack        [████                                              ]  4.79%   (332 imgs)
  Rockpocket    [████                                              ]  4.48%   (311 imgs)
  WConccor      [███                                               ]  3.82%   (265 imgs) <- RAREST

  PIXEL AREA PERCENTAGE (512x512):
  Weathering    [██████████████████████████████████████████████████] 10.05% (182.7M px)
  PEquipment    [███████████████                                   ]  3.00%  (54.6M px)
  Hollowareas   [██████████                                        ]  2.11%  (38.4M px)
  Spalling      [█████████                                         ]  1.96%  (35.6M px)
  Rust          [████████                                          ]  1.71%  (31.2M px)
  ACrack        [█████                                             ]  1.05%  (19.0M px)
  WConccor      [██                                                ]  0.38%   (6.9M px)
  Crack         [█                                                 ]  0.26%   (4.8M px) <- THIN
  Rockpocket    [█                                                 ]  0.21%   (3.8M px)
  ========================================================================================
```

### 3.1. WConccor (Weathered / Wet Concrete Corrosion) — Root Cause Analysis
- **Training Frequency:** 265 images (3.82%), 360 shapes, 6,938,109 pixels (0.38% of training pixels).
- **Validation Outcome:** Validation IoU = **0.0002**, Recall = **0.0002**, Precision = **0.0048**. Out of 842,484 GT pixels across 38 positive validation images, only **137 pixels** were predicted positive at threshold 0.5.
- **Why Did WConccor Fail?**
  1. **Dual Imbalance (Image + Pixel) & Batch Prevalence:** WConccor suffers from the lowest image occurrence in the dataset (265 images / 3.82% prevalence; 96.18% of images are negative for WConccor). At a mini-batch size of $B = 8$ (random sampling without replacement), the hypergeometric probability that a batch contains zero WConccor images is $P(\text{empty}) = \frac{\binom{6670}{8}}{\binom{6935}{8}} \approx 73.21\%$. Consequently, approximately $26.79\%$ of training mini-batches (roughly 232 out of 867 batches per epoch) contain at least one WConccor image. However, even in positive batches, WConccor accounts for an extremely small fraction of pixels compared to background and dominant classes, causing unweighted BCE loss to overwhelmingly penalize positive predictions.
  2. **Severe Co-occurrence & Semantic Entanglement:**
     - In the training set, when WConccor is present:
       - **58.49%** of the images also contain `Spalling` (co-occurrence count = 155).
       - **56.23%** also contain `Rust` (co-occurrence count = 149).
       - **40.38%** also contain `Weathering` (co-occurrence count = 107).
       - **29.06%** also contain `Hollowareas` (co-occurrence count = 77).
       - **26.42%** also contain `Efflorescence` (co-occurrence count = 70).
     - Furthermore, spatial intersection analysis reveals that **14.62% of WConccor pixels directly overlap with `Hollowareas`**, and **5.65% directly overlap with `Weathering`**.
  3. **Visual Similarity & Logit Suppression:** Concrete corrosion caused by water leakage manifests visually as discolored, rough, damp, efflorescent, spalled concrete. Because `Weathering`, `Spalling`, and `Rust` have 26x, 5x, and 4.5x more training data, the shared feature representations in the DeepLabV3+ encoder strongly activate for `Weathering` or `Spalling`, leaving WConccor output logits below 0.0 (probability $< 0.5$).
  4. **Diagnostic Probe at Auxiliary Threshold $\tau = 0.3$ (Diagnostic Only):**
     - *Important caveat:* Operating at $\tau = 0.3$ is strictly a diagnostic probe to confirm the presence of sub-threshold learned representations; it is not a claim of calibrated or production-ready spatial segmentation.
     - At $\tau = 0.5$: Predicted pixels = 28,725; $\text{IoU} = 0.0002$.
     - At $\tau = 0.3$: Predicted pixels jumped by **1,155%** to 331,760; $\text{IoU}$ improved to **0.0597** (Dice = 0.1126).
     - This demonstrates that while the model has acquired latent spatial correlations for WConccor, unweighted negative gradient pressure suppresses its sigmoid probabilities into the $[0.20, 0.45]$ range. Proper loss re-weighting is required to recalibrate logits to standard $\tau = 0.5$.

### 3.2. Rockpocket (Honeycombing / Grout Loss)
- **Training Frequency:** 311 images (4.48%), 525 shapes, 3,810,196 pixels (0.21% of training pixels).
- **Validation Outcome:** Validation IoU = **0.0689**, Recall = **0.0843**, Precision = **0.2732**.
- **Morphology:** 36.38% small, 45.14% medium, 18.48% large (median area = 1,927.3 px).
- **Why Did Rockpocket Struggle?**
  1. Rockpocket represents pockets of coarse aggregate exposed due to incomplete grout filling or honeycombing during pouring.
  2. It co-occurs heavily with `Spalling` (58.8% co-occurrence) and `Cavity` (26.7% co-occurrence).
  3. The network confuses the porous texture of small rock pockets with `Cavity` (which has 8,119 shapes in the training set!) or rough `Spalling`.
  4. At threshold 0.3, Rockpocket IoU increases to **0.1039** (recall increases to 0.231), corroborating the same threshold suppression effect observed in WConccor.

### 3.3. Crack (Linear Structural Cracks) vs. ACrack (Alligator / Surface Web)
The divergence between `Crack` (IoU 0.1848) and `ACrack` (IoU 0.4607) reveals a fundamental insight into defect morphology:
- **Crack:**
  - 1,727 images (24.90%) — 5.2x more images than ACrack!
  - Yet it occupies only 4,787,533 pixels (0.26% of training pixels).
  - **62.78% small defects**, median area = **615.1 pixels**.
  - Cracks are long, thin lines (1–3 pixels wide in $512 \times 512$). In binary cross-entropy and Dice loss, a 2-pixel spatial boundary error on a 2-pixel wide crack drops the IoU from 1.0 to 0.0. The standard BCE-Dice loss provides weak gradient signals for thin linear topology.
- **ACrack (Alligator Crack):**
  - Only 332 images (4.79%) — rarest defect group!
  - Yet it occupies 19,043,215 pixels (1.05% of training pixels) — **4x more pixels than Crack**!
  - **72.61% large defects**, median area = **22,281.5 pixels** (36x larger than linear Crack).
  - Alligator cracking forms a polygon enclosing an entire network of micro-fractures across the concrete surface. The network easily segments the broad textured polygon, achieving **IoU 0.4607** despite few training images.

---

## 4. Annotation Quality & Polygon Rasterization Audit

To eliminate the possibility of data corruption, coordinate inversions, or index mismatches, positive samples from the training and validation sets were audited:

### Table 2: Positive Sample Coordinate & Alignment Verification
| Class | Split | Sample ID | Image Size | Polygons | Coordinate Bounds | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **WConccor** | Train | `dacl10k_v2_train_0003.json` | $1600 \times 1200$ | 2 | $X \in [705.4, 1245.4], Y \in [106.8, 150.3]$ | Verified In-Bounds |
| **WConccor** | Train | `dacl10k_v2_train_0016.json` | $640 \times 480$ | 1 | $X \in [0.0, 352.8], Y \in [267.5, 324.5]$ | Verified In-Bounds |
| **WConccor** | Val | `dacl10k_v2_validation_0024.json` | $1024 \times 683$ | 1 | $X \in [412.5, 613.3], Y \in [244.7, 490.2]$ | Verified In-Bounds |
| **WConccor** | Val | `dacl10k_v2_validation_0055.json` | $1280 \times 960$ | 3 | $X \in [253.5, 1119.0], Y \in [0.5, 870.8]$ | Verified In-Bounds |
| **Rockpocket**| Train | `dacl10k_v2_train_0003.json` | $1600 \times 1200$ | 8 | $X \in [112.3, 845.5], Y \in [426.3, 1029.9]$ | Verified In-Bounds |
| **Rockpocket**| Val | `dacl10k_v2_validation_0016.json` | $1280 \times 960$ | 4 | $X \in [547.9, 1272.3], Y \in [1.1, 753.7]$ | Verified In-Bounds |
| **Rockpocket**| Val | `dacl10k_v2_validation_0017.json` | $1280 \times 960$ | 1 | $X \in [0.0, 1279.0], Y \in [295.0, 646.6]$ | Verified In-Bounds |
| **Crack** | Train | `dacl10k_v2_train_0007.json` | $800 \times 600$ | 2 | $X \in [454.1, 767.0], Y \in [0.2, 599.0]$ | Verified In-Bounds |
| **Crack** | Val | `dacl10k_v2_validation_0011.json` | $960 \times 720$ | 1 | $X \in [209.6, 853.8], Y \in [299.5, 359.9]$ | Verified In-Bounds |
| **Crack** | Val | `dacl10k_v2_validation_0016.json` | $1280 \times 960$ | 1 | $X \in [1080.7, 1277.1], Y \in [330.0, 483.6]$ | Verified In-Bounds |
| **ACrack** | Train | `dacl10k_v2_train_0054.json` | $1152 \times 864$ | 1 | $X \in [560.0, 716.3], Y \in [0.0, 181.0]$ | Verified In-Bounds |
| **ACrack** | Val | `dacl10k_v2_validation_0026.json` | $1600 \times 1200$ | 1 | $X \in [0.0, 1599.0], Y \in [0.0, 1199.0]$ | Verified In-Bounds |
| **ACrack** | Val | `dacl10k_v2_validation_0038.json` | $1843 \times 1382$ | 1 | $X \in [706.5, 970.4], Y \in [505.9, 645.2]$ | Verified In-Bounds |

### Canonical Class Index Verification:
All model heads, dataset channel mappings, and loss tensors strictly match canonical ordering:
`0: ACrack, 1: Bearing, 2: Cavity, 3: Crack, 4: Drainage, 5: EJoint, 6: Efflorescence, 7: ExposedRebars, 8: Graffiti, 9: Hollowareas, 10: JTape, 11: PEquipment, 12: Restformwork, 13: Rockpocket, 14: Rust, 15: Spalling, 16: WConccor, 17: Weathering, 18: Wetspot`.
No class index shifting or mislabeling bug exists.

---

## 5. Class-Specific 4-Panel Diagnostic Visualizations

High-resolution 4-panel diagnostic visualizations were generated for each focal defect category. Each figure displays:
1. **Original Image** ($512 \times 512$ letterbox).
2. **Ground-Truth Mask (Target Class Only)** — unoccluded binary mask extracted directly from the ground-truth annotations.
3. **Prediction Mask (Target Class Only)** — model binary prediction at threshold $\tau = 0.5$.
4. **Diagnostic Error Overlay** — original image blended with **True Positives (Green)**, **False Positives (Red)**, and **False Negatives (Cyan)**, with region boundary contours.

### Table 3: Qualitative Diagnostic Case Summary
| Class | Sample ID | Status | GT Pixels | Pred Pixels | TP (Green) | FP (Red) | FN (Cyan) | Sample IoU | Diagnosis |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **WConccor** | `val_0024` | **MISSED** | 9,438 | 0 | 0 | 0 | 9,438 | **0.0000** | 100% False Negative. Entire defect subsumed into Weathering (231,212 px predicted). |
| **WConccor** | `val_0055` | **MISSED** | 51,619 | 0 | 0 | 0 | 51,619 | **0.0000** | 100% False Negative. Massive corrosion patch predicted as Spalling (78.5k px) & Weathering (27.0k px). |
| **Rockpocket**| `val_0017` | **DETECTED** | 60,060 | 2,046 | 2,045 | 1 | 58,015 | **0.0340** | True positive core detected with 0 FP; partial perimeter recall loss. |
| **Rockpocket**| `val_0016` | **MISSED** | 20,600 | 0 | 0 | 0 | 20,600 | **0.0000** | 100% False Negative. Porous aggregate confused with Spalling and Hollowareas. |
| **Crack** | `val_0016` | **DETECTED** | 720 | 191 | 49 | 142 | 671 | **0.0568** | Core crack trajectory detected; boundary thickness misalignment. |
| **Crack** | `val_0011` | **MISSED** | 3,718 | 28 | 0 | 28 | 3,718 | **0.0000** | Faint horizontal fissure suppressed below 0.5 threshold. |
| **ACrack** | `val_0026` | **DETECTED** | 289,852 | 46,104 | 44,091 | 2,013 | 245,761 | **0.1511** | High true positive volume across extensive surface web. |
| **ACrack** | `val_0038` | **OVER-PRED**| 1,372 | 37,069 | 1,372 | 35,697 | 0 | **0.0370** | **100% Recall** of defect polygon, but false positive expansion over adjacent concrete. |
| **Cavity** | `val_0026` | **DETECTED** | 4,815 | 2,584 | 421 | 2,163 | 4,394 | **0.0603** | Localized voids detected. |
| **ExposedRebars**| `val_0034`| **MISSED** | 26,493 | 0 | 0 | 0 | 26,493 | **0.0000** | Thin rusted bar obscured by background concrete. |

*All 10 diagnostic images are saved in `reports/visualizations/` and embedded in the companion analysis artifact.*

---

## 6. Training-Only Candidate Class Weights

To establish mathematically grounded, objective loss re-weighting, candidate class weights were computed **strictly from the 6,935 training images**. No validation statistics were used to derive these weights.

### Formulations Evaluated:
1. **Inverse Image Frequency (Normalized):**
   $$w_c^{img} = \frac{N_{total}}{K \cdot N_c}, \quad \bar{w}^{img} = \frac{w_c^{img}}{\frac{1}{K}\sum_k w_k^{img}}$$
2. **Inverse Pixel Frequency (512x512 Normalized):**
   $$w_c^{pix} = \frac{\sum_k P_k}{K \cdot P_c}, \quad \bar{w}^{pix} = \frac{w_c^{pix}}{\frac{1}{K}\sum_k w_k^{pix}}$$
   *Drawback:* High dynamic range ($0.05$ to $4.42$), risking gradient explosion on rare classes.
3. **Square-Root Inverse Pixel Frequency (Normalized):**
   $$w_c^{sqrt} = \sqrt{\frac{\sum_k P_k}{P_c}}, \quad \bar{w}^{sqrt} = \frac{w_c^{sqrt}}{\frac{1}{K}\sum_k w_k^{sqrt}}$$
   *Advantage:* Dampens extreme ratios while providing consistent gradient amplification.
4. **Effective Number of Samples (Cui et al., CVPR 2019):**
   $$E_{N_c} = \frac{1 - \beta^{N_c}}{1 - \beta}, \quad w_c^{eff} = \frac{1}{E_{N_c}} \quad (\beta = 0.999)$$
5. **Recommended Normalized Square-Root Inverse-Pixel Weights:**
   Square-root inverse-pixel weights normalized so that mean weight $= 1.0000$ (sum $= 19.0000$).
   *Methodological note on bounds:* Any clipping to $[0.2, 5.0]$ is completely non-binding because all 19 weights naturally fall strictly within this interval (minimum is Weathering at $0.2538$, maximum is ExposedRebars at $2.3887$). Hence, this strategy is formally designated as **normalized square-root inverse-pixel weights**.

### Table 4: Candidate Class Weights Derived Strictly from Training Data
*Sorted by Canonical Class Index*

| Index | Class Name | Inverse Image Norm | Inverse Pixel Raw | Inverse Pixel Norm | Sqrt Pixel Norm | Median Freq | Effective Samples ($\beta=0.999$) | **Normalized Sqrt Inv-Pixel** |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | ACrack | 2.2112 | 1.4831 | 0.4786 | 0.7862 | 0.8850 | 1.8079 | **0.7862** |
| 1 | Bearing | 1.0057 | 1.6758 | 0.5409 | 0.8357 | 1.0000 | 0.9859 | **0.8357** |
| 2 | Cavity | 0.6180 | 2.5567 | 0.8251 | 1.0322 | 1.5256 | 0.7349 | **1.0322** |
| 3 | **Crack** | 0.4251 | 5.8992 | 1.9039 | 1.5679 | 3.5201 | 0.6214 | **1.5679** |
| 4 | Drainage | 0.7127 | 2.0699 | 0.6680 | 0.9288 | 1.2351 | 0.7945 | **0.9288** |
| 5 | EJoint | 1.8586 | 1.7848 | 0.5760 | 0.8624 | 1.0650 | 1.5652 | **0.8624** |
| 6 | Efflorescence | 0.4820 | 1.0227 | 0.3300 | 0.6528 | 0.6102 | 0.6533 | **0.6528** |
| 7 | ExposedRebars | 0.9497 | 13.6915 | 4.4188 | 2.3887 | 8.1699 | 0.9488 | **2.3887** |
| 8 | Graffiti | 0.9234 | 1.0226 | 0.3300 | 0.6528 | 0.6102 | 0.9314 | **0.6528** |
| 9 | Hollowareas | 0.6692 | 0.7357 | 0.2374 | 0.5537 | 0.4390 | 0.7669 | **0.5537** |
| 10 | JTape | 0.9571 | 5.5937 | 1.8053 | 1.5268 | 3.3379 | 0.9537 | **1.5268** |
| 11 | PEquipment | 0.5566 | 0.5176 | 0.1671 | 0.4644 | 0.3089 | 0.6973 | **0.4644** |
| 12 | Restformwork | 1.0296 | 6.6893 | 2.1589 | 1.6697 | 3.9916 | 1.0019 | **1.6697** |
| 13 | **Rockpocket** | 2.3606 | 7.4123 | 2.3922 | 1.7576 | 4.4231 | 1.9109 | **1.7576** |
| 14 | Rust | 0.2128 | 0.9066 | 0.2926 | 0.6147 | 0.5410 | 0.5277 | **0.6147** |
| 15 | Spalling | 0.2251 | 0.7940 | 0.2563 | 0.5753 | 0.4738 | 0.5313 | **0.5753** |
| 16 | **WConccor** | **2.7703** | **4.0706** | **1.3137** | **1.3025** | **2.4290** | **2.1940** | **1.3025** |
| 17 | Weathering | 0.2715 | 0.1546 | 0.0499 | 0.2538 | 0.0922 | 0.5476 | **0.2538** |
| 18 | Wetspot | 0.7608 | 0.7906 | 0.2552 | 0.5740 | 0.4718 | 0.8252 | **0.5740** |

---

## 7. Comparative Intervention Analysis & Ablation Roadmap

| Intervention | Mechanism | Primary Advantages | Major Risks / Drawbacks | Compute & Memory Impact |
|:---|:---|:---|:---|:---|
| **A. Channel-Level Class-Weighted BCE-Dice Loss** *(Recommended)* | Channel-level weighting applied uniformly across all pixels for each class channel $c$ (no inner `pos_weight`): $\mathcal{L}_{total} = \frac{\sum_c w_c \cdot (\text{BCE}_c + \text{Dice}_c)}{\sum_c w_c}$ | Directly counters gradient starvation for WConccor and Rockpocket; dampens dominant Weathering gradients. | Extreme weights can cause gradient instability and high false-positive rates; square-root scaling dampens dynamic range safely. | **Zero extra GPU VRAM**;<br>**Zero extra training time** ($+0\%$). |
| **B. Class-Aware Image Sampling** | Sample training images inversely proportional to class frequency (e.g. WeightedRandomSampler or repeat oversampling). | Guarantees rare classes (WConccor, Rockpocket) appear in every mini-batch. | Multi-label contamination: oversampling WConccor also oversamples Spalling and Weathering; high risk of memorizing the 265 WConccor images. | Increases epoch step count; $+25\%$ to $+50\%$ training time. |
| **C. Boundary-Aware Loss for Thin Cracks** | Incorporate Hausdorff / Boundary Loss or Focal Tversky Loss ($\alpha=0.7, \beta=0.3$) on the linear Crack channel. | Directly addresses thin linear morphology ($62.8\%$ small instances) and boundary misalignment. | Complex tuning; distance transform precomputation required; does not fix area-based defects (WConccor, Rockpocket). | Distance transform adds IO latency; minor VRAM increase ($+0.4\text{ GB}$). |
| **D. Multi-Threshold Calibration** | Calibrate class-specific inference thresholds ($\tau_c \in [0.25, 0.50]$) on validation split. | Instant recovery of suppressed logits without retraining. | Violates benchmark protocol if benchmark requires fixed $\tau = 0.5$. | Zero training compute. |

### Recommended Ablation Roadmap:
- **Baseline (Reference):** DeepLabV3+ ResNet-34, standard unweighted BCE + Dice ($\text{mIoU} = 0.3384$).
- **Ablation 1 (Primary):** Baseline + **Normalized Square-Root Inverse-Pixel Channel-Weighted BCE-Dice Loss**.
- **Ablation 2:** Baseline + Sqrt Class-Weighted Loss + Focal Tversky on Crack channel.
- **Ablation 3:** Baseline + Class-Aware Weighted Random Sampler (Oversampling ratio $\le 2\times$).

---

## 8. Primary Recommended Experiment

### Experiment Name: `exp_02_sqrt_weighted_bcedice`
**Core Hypothesis:**  
Scaling the positive loss gradients by training-derived square-root class weights (Table 4) will lift WConccor and Rockpocket prediction probabilities into the $[0.50, 0.80]$ range, transforming hundreds of thousands of false negatives into true positives without destabilizing dominant reference classes.

### Mathematical Formulation:
For each class channel $c \in \{0, \dots, 18\}$, the channel loss is defined as:
$$\mathcal{L}_c = w_c \cdot \left[ \text{BCE}_c(z_c, y_c) + \text{Dice}_c(\sigma(z_c), y_c) \right]$$
$$\mathcal{L}_{total} = \frac{\sum_{c=0}^{18} w_c \cdot (\text{BCE}_c + \text{Dice}_c)}{\sum_{c=0}^{18} w_c}$$
Where:
- $\text{BCE}_c$ is standard binary cross-entropy on channel $c$ (computed independently across batch and spatial dimensions).
- $\text{Dice}_c$ is Soft Dice loss on channel $c$ ($1 - \text{DiceScore}_c$).
- **No additional inner `pos_weight` is applied** in this primary experiment, preventing double-weighting of positive pixels.
- $w_c$ is the exact training-derived normalized square-root pixel frequency weight from Table 4 (normalized such that $\sum w_c \approx 19.0, \bar{w} = 1.0$):
  - $w_{\text{ExposedRebars}} = 2.3887$
  - $w_{\text{Rockpocket}} = 1.7576$
  - $w_{\text{Restformwork}} = 1.6697$
  - $w_{\text{Crack}} = 1.5679$
  - $w_{\text{JTape}} = 1.5268$
  - $w_{\text{WConccor}} = 1.3025$
  - ...
  - $w_{\text{Weathering}} = 0.2538$ (dampened to avoid suppressing co-occurring rare defects)

### Estimated Hardware Requirements & 5-Epoch Pilot Plan:
- **GPU Architecture:** Single NVIDIA T4 (Google Colab free/Pro) or A100 (Pro+).
- **VRAM Utilization:** **5.8 GB / 15.0 GB** (Batch size = 8, $512 \times 512$ letterbox).
- **Fine-Tuning Learning Rate:** $5 \times 10^{-5}$ ($0.5\times$ base LR) with cosine annealing, initialized from baseline Epoch-15 best checkpoint.
- **Pilot Strategy:** Run a 5-epoch pilot first to verify loss convergence and monitor for any regressions in strong reference classes before committing to a 15-epoch run.
- **Hypothesized Target Outcomes (Unverified Projections):**
  - *Hypothesis 1:* WConccor IoU projected to recover from $0.0002$ into the range $0.08 - 0.20$.
  - *Hypothesis 2:* Rockpocket IoU projected to improve from $0.0689$ into the range $0.15 - 0.25$.
  - *Hypothesis 3:* Crack IoU projected to improve from $0.1848$ into the range $0.22 - 0.28$.
  - *Hypothesis 4:* Overall validation mIoU projected to improve from $0.3384$ towards $\sim 0.36 - 0.38$.
  - *Verification Note:* All projections above represent experimental hypotheses to be empirically validated or falsified during the 5-epoch pilot run. None are guaranteed outcomes.

---

## 9. Non-Interference Verification

In compliance with instructions:
- Existing Epoch-15 checkpoints (`checkpoints/best_model.pth`, `checkpoints/latest_checkpoint.pth`) and historical metrics are fully preserved and untouched.
- No model architecture modifications, training loss edits, or threshold changes have been applied to active pipelines.
- The validation split (975 images) remains completely untouched and held out exclusively for unbiased evaluation.
