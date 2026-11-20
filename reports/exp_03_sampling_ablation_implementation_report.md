# Experiment 03: Rare-Class-Aware Sampling Ablation Implementation Report

**Project:** Bridge AI — Autonomous Multi-Label Bridge Defect Semantic Segmentation  
**Dataset:** DACL10K-v2 (Development Phase — 6,935 Train, 975 Validation)  
**Experiment Name:** `exp_03_rare_class_sampling`  
**Model Architecture:** DeepLabV3+ with ResNet-34 Encoder (ImageNet Pretrained)  
**Output Configuration:** 19 Independent Multi-Label Sigmoid Channels ($512 \times 512$ letterbox)  
**Core Objective:** Isolate image-level rare-class oversampling using the **original unweighted CombinedBCEDiceLoss**, establishing an ablation comparison against Experiment 02 (loss re-weighting) and the Stage 1 baseline.

---

## 1. Mathematical Sampling Formulation & Distribution

### 1.1. Image Weighting Formula
For each training image $i \in \{1, \dots, N\}$ (where $N = 6,935$):
$$w_i = \min\left(4.0, \; 1.0 + 2.0 \cdot \mathbf{1}_{\text{WConccor}}(i) + 1.5 \cdot \mathbf{1}_{\text{Rockpocket}}(i) + 0.5 \cdot \mathbf{1}_{\text{Crack}}(i)\right)$$

Where:
- $\mathbf{1}_c(i) = 1$ if class label $c$ appears in the verified LabelMe JSON annotation shapes for image $i$, else $0$.
- These are **preliminary image-level sampling probabilities**, not loss weights. No validation split annotations were examined.
- Maximum weight is strictly capped at $4.0$ to prevent runaway repetition and feature memorization.

### 1.2. Weight Distribution on Full 6,935 Training Split

| Combination | Target Classes Present | Weight ($w_i$) | Image Count | Image Fraction |
|:---|:---|:---:|:---:|:---:|
| **None** | Background / Other 16 classes | **1.0** | 4,810 | 69.36% |
| **Crack Only** | Crack | **1.5** | 1,573 | 22.68% |
| **Rockpocket Only** | Rockpocket | **2.5** | 201 | 2.90% |
| **WConccor Only** | WConccor | **3.0** | 201 | 2.90% |
| **Rockpocket + Crack** | Rockpocket, Crack | **3.0** | 68 | 0.98% |
| **WConccor + Crack** | WConccor, Crack | **3.5** | 58 | 0.84% |
| **WConccor + Rockpocket**| WConccor, Rockpocket (clamped from 4.5) | **4.0** | 2 | 0.03% |
| **All Three Target Classes**| WConccor, Rockpocket, Crack (clamped from 5.0) | **4.0** | 22 | 0.32% |
| **Total** | — | — | **6,935** | **100.00%** |

- **Sum of Weights ($W_{\text{total}}$):** $8,778.00$
- **Mean Sampling Weight ($\bar{w}$):** $1.2658$
- **Weight Range:** $[1.00, 4.00]$

### 1.3. Expected Class Exposure vs. Baseline Prevalence
Drawing $N = 6,935$ samples per epoch with replacement via `WeightedRandomSampler`:

$$P(c) = \frac{\sum_{i \in \mathcal{S}_c} w_i}{W_{\text{total}}}, \quad \mathbb{E}[\text{samples with } c] = P(c) \times N, \quad \text{Boost}_c = \frac{P(c)}{N_c / N}$$

| Class Name | Index | Base Images | Base Prevalence | Expected Sampling Prob | Expected Drawn Imgs/Epoch | Expected Boost Factor |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **WConccor** | 16 | **265** | **3.82%** | **9.66%** | **670.0** | **2.53x Boost** |
| **Rockpocket** | 13 | **311** | **4.48%** | **9.76%** | **676.7** | **2.18x Boost** |
| **Crack** | 3 | **1,727** | **24.90%** | **32.59%** | **2,259.9** | **1.31x Boost** |
| Non-target (avg) | — | 4,810 | 69.36% | 54.80% | 3,800.4 | 0.79x (attenuation) |

---

## 2. Experimental Rigor & Isolation Guarantees

1. **Loss Function Isolation:**
   - Experiment 03 strictly isolates sampling by using the **original unweighted CombinedBCEDiceLoss**:
     $$\mathcal{L} = \text{BCEWithLogitsLoss}(z, y) + \text{MultiLabelSoftDiceLoss}(\sigma(z), y)$$
   - No channel weights ($w_c$) and no positive-pixel weights (`use_pos_weight: false`).
   - This directly benchmarks whether rare-class oversampling alone resolves gradient starvation without changing the optimization loss surface.

2. **Validation Split Protection:**
   - Validation loader remains completely unweighted, evaluated deterministically on all 975 validation images with `shuffle=False`, `replacement=False`, and `augment=False`.
   - Primary metric threshold remains $\tau = 0.5$.

3. **Deterministic Sampling & Resumption Contract:**
   - To guarantee reproducible epoch sampling and consistency across interruptions/resumptions:
     $$\text{Sampler Seed}(\text{epoch}) = \text{seed} + \text{epoch} = 42 + \text{epoch}$$
   - Resuming from an epoch checkpoint (e.g. at Epoch 3) seeds the `torch.Generator` identically to a continuous run through Epochs 1 to 3.

4. **Exposure & Repetition Tracking:**
   - Using `RecordingWeightedRandomSampler`, exact sample indices drawn per epoch are captured and exported to `exp_dir / f"sampling_epoch_{epoch:03d}.json"` and `exp_dir / "sampling_exposure_history.json"`.
   - Tracks unique image coverage, maximum and mean repetition counts, and actual exposure for all 19 classes separately from validation metrics.

---

## 3. Configuration & Artifact Isolation

- **Configuration File:** [`configs/exp_03_rare_class_sampling.yaml`](file:///Users/kundankumar/Documents/untitled%20folder%202/bridge-ai-project/configs/exp_03_rare_class_sampling.yaml)
- **Local Smoke Test Configuration:** [`configs/exp_03_smoke_test.yaml`](file:///Users/kundankumar/Documents/untitled%20folder%202/bridge-ai-project/configs/exp_03_smoke_test.yaml)
- **Pretrained Weights Source:** `/content/drive/MyDrive/bridge_ai_experiments/segmentation_baseline/checkpoints/best_model.pth` (Baseline 15-epoch best model)
- **Dedicated Output Directory:** `/content/drive/MyDrive/bridge_ai_experiments/exp_03_rare_class_sampling/`
- **Baseline & Exp 02 Invariance:** Neither the baseline checkpoints nor the Exp 02 checkpoints/logs are altered.

---

## 4. Automated Verification Results

- **Targeted Exp 03 Unit Tests ([`tests/test_exp_03_sampling.py`](file:///Users/kundankumar/Documents/untitled%20folder%202/bridge-ai-project/tests/test_exp_03_sampling.py)):** 9/9 tests passed.
  - Formula validation (all 8 binary combinations).
  - Label extraction and canonical class indexing.
  - Subset alignment on filtered datasets.
  - Deterministic sampling and resume consistency.
  - Disabled-sampler backward compatibility.
  - Validation isolation.
  - Unweighted loss verification.
  - Path and checkpoint isolation.
  - Recording sampler index recording and exposure tracking.
- **Full Project Regression Test Suite:** **38/38 tests passed in 6.91s**.
- **1-Batch CPU Smoke Test:** Successfully executed forward pass, unweighted loss calculation, backward step, optimizer update, checkpointing, and sampling exposure JSON export.
