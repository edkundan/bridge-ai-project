# Stage 2: Defect Measurement and Severity Assessment Foundation

**Project:** Explainable AI-Based Bridge Defect Severity Assessment and Risk-Based Maintenance Prioritization  
**Repository:** `bridge-ai-project`  
**Dataset:** DACL10K-v2 (19 semantic classes, multi-label segmentation)  
**Upstream Models:** DeepLabV3+ with ResNet-34 (Baseline mIoU: 0.3384 | Exp 02 Weighted Loss: 0.3542 | Exp 03 Rare-Class Sampling: 0.3515)  
**Date:** November 2026  

---

## 1. Executive Summary & Objective

Stage 1 established the semantic segmentation foundation for detecting 19 bridge defects and components from visual inspection imagery. However, raw segmentation masks (binary or probability arrays) are insufficient for civil infrastructure asset management. Real-world bridge management systems (such as AASHTO BrM or German SIB-Bauwerke) require **objective defect quantification** (area, perimeter, length, width, location) and **defensible severity categorization** to prioritize maintenance interventions.

Stage 2 establishes the quantitative bridge defect measurement foundation and an extensible, literature-grounded severity assessment framework. Crucially, this implementation enforces an architectural boundary:
1. **Measured Visual Evidence (Directly Observable):** Pixel counts, image/ROI-relative areas, bounding boxes, spatial centroids, confidence distributions, and 2D morphological shape descriptors.
2. **Structural Severity Assessment (Engineering Decisions):** Decoupled through an extensible abstract interface. It categorizes visual evidence according to bridge inspection guidelines while explicitly disclaiming uncalibrated metric or structural capacity claims until engineering ground truth is available.

---

## 2. Module Architecture

The Stage 2 codebase resides in `src/measurement/` and integrates with existing segmentation predictors (`src/segmentation/predict.py`) as read-only inputs without altering model weights or prior experiments:

```
src/measurement/
├── __init__.py          # Public API export
├── schema.py            # Dataclasses, validation contracts & JSON serialization
├── geometry.py          # Morphological skeletonization, connected components & shape analysis
├── pipeline.py          # DefectMeasurementPipeline (multi-channel input -> structured measurements)
├── visualization.py     # Publication-quality overlay rendering & annotation banner
└── severity.py          # Abstract BaseSeverityEvaluator & literature-grounded baseline
```

### Component Roles & Data Flow

```
                      19-Channel Probabilities (C, H, W)
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │     DefectMeasurementPipeline     │
                   │  - Configurable tau (default 0.5) │
                   │  - Overlapping channels preserved │
                   │  - Optional structural ROI mask   │
                   └─────────────────┬─────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │   geometry.py             │   │   schema.py               │
       │ - cv2.connectedComponents │   │ - ImageMeasurementResult  │
       │ - Morphological Skeleton  │   │ - ClassMeasurementSummary │
       │ - Bounding Boxes & Extent │   │ - ShapeFeatures           │
       │ - Area % (Image & ROI)    │   │ - ConfidenceStats         │
       └─────────────┬─────────────┘   └─────────────┬─────────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                      ImageMeasurementResult (JSON)
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │   visualization.py        │   │   severity.py             │
       │ - Semi-transparent masks  │   │ - BaseSeverityEvaluator   │
       │ - Bounding box callouts   │   │ - AASHTO/FHWA Tiers (1-4) │
       │ - Structural ROI contour  │   │ - Missing parameter flags │
       │ - Metadata summary banner │   │ - Explicit disclaimers    │
       └───────────────────────────┘   └───────────────────────────┘
```

---

## 3. Mathematical & Geometric Formulation

### 3.1 Pixel Area and Relative Area Percentages
For an image with resolution $H \times W$ ($A_{\text{img}} = H \cdot W$), let $M_c \in \{0, 1\}^{H \times W}$ denote the binary mask for class $c$ obtained at threshold $\tau_c$:

$$M_c(y, x) = \mathbf{1}\left[P_c(y, x) \ge \tau_c\right]$$

The image-relative area percentage for class $c$ is:

$$\alpha_{c, \text{img}} = \frac{\sum_{y, x} M_c(y, x)}{H \cdot W} \times 100\%$$

When an optional structural Region-of-Interest (ROI) mask $R \in \{0, 1\}^{H \times W}$ is provided (representing the concrete beam, pier column, or deck slab with $A_{\text{roi}} = \sum R(y, x) > 0$), the ROI-relative defect area is:

$$\alpha_{c, \text{roi}} = \frac{\sum_{y, x} \left(M_c(y, x) \land R(y, x)\right)}{A_{\text{roi}}} \times 100\%$$

### 3.2 Connected Component Analysis & Spatial Isolation
For each class channel $M_c$, 8-connectivity connected component labeling partitions positive pixels into $K_c$ disjoint subsets $\{S_{c, k}\}_{k=1}^{K_c}$:

$$\bigcup_{k=1}^{K_c} S_{c, k} = \{(y, x) \mid M_c(y, x) = 1\}, \quad S_{c, i} \cap S_{c, j} = \emptyset \; (i \ne j)$$

Components with pixel count $|S_{c, k}| < A_{\text{min}}$ (default $A_{\text{min}} = 10$ pixels) are filtered out as isolated optical noise.

For each retained component $k$, the axis-aligned bounding box $\mathcal{B} = [x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ and spatial centroid $(\bar{x}, \bar{y})$ are computed:

$$\bar{x} = \frac{1}{|S_{c, k}|} \sum_{(y, x) \in S_{c, k}} x, \quad \bar{y} = \frac{1}{|S_{c, k}|} \sum_{(y, x) \in S_{c, k}} y$$

### 3.3 Elongated / Linear Defect Morphology (Cracks)
For linear defects (`Crack`, `ACrack`, `Efflorescence`) or components with aspect ratio $\ge 3.0$, topological centerline thinning is computed via iterative morphological erosion and opening:

$$S_{i+1} = S_i \setminus \left( (S_i \ominus B) \circ B \right)$$

where $B$ is a $3 \times 3$ cross structuring element. The skeleton length $L_{\text{skel}}$ and estimated average pixel width $\bar{W}_{\text{px}}$ are:

$$L_{\text{skel}} = \sum_{(y, x)} \text{Skeleton}(y, x), \quad \bar{W}_{\text{px}} = \frac{|S_{c, k}|}{\max(1, L_{\text{skel}})}$$

Additional shape descriptors include:
- **Circularity (Isoperimetric Quotient):** $\mathcal{C} = \frac{4 \pi \cdot |S|}{P^2} \in [0, 1]$ (smooth circles $\approx 1.0$; thin cracks $\ll 0.1$).
- **Aspect Ratio:** $\mathcal{AR} = \frac{\text{major\_axis}}{\text{minor\_axis}} \ge 1.0$ (from minimum-area oriented bounding rectangle $\text{minAreaRect}$).
- **Solidity:** $\mathcal{S} = \frac{|S|}{\text{Area}(\text{ConvexHull}(S))} \in [0, 1]$.
- **Extent:** $\mathcal{E} = \frac{|S|}{(x_{\text{max}} - x_{\text{min}}) \cdot (y_{\text{max}} - y_{\text{min}})}$.

---

## 4. Directly Observable Evidence vs. Unobservable Engineering Reality

A core tenet of this research is scientific honesty. Optical 2D computer vision models operate under fundamental physical limits. We delineate what is directly measurable versus what requires physical sensing or certified engineering validation:

| Dimension / Feature | Directly Measurable from 2D RGB (Stage 2) | Requires In-Situ Calibration or Engineering Ground Truth |
| :--- | :--- | :--- |
| **Defect Footprint** | Pixel area ($A_{\text{px}}$), bounding box coordinates, image-relative coverage ($\%$). | Physical area ($\text{m}^2, \text{cm}^2$) requires metric calibration target, sensor focal length + rangefinder distance. |
| **Crack Length & Width** | Centerline pixel length ($L_{\text{skel}}$), apparent pixel width ($\bar{W}_{\text{px}}$), orientation angle. | Physical crack width in millimeters ($w \ge 0.3\,\text{mm}$ critical threshold) requires known pixel pitch ($\mu\text{m/pixel}$) and surface-normal camera angle. |
| **Crack Depth / Void Depth** | **NOT MEASURABLE** (2D RGB sensors cannot observe depth beneath the concrete surface plane). | Ultrasonic pulse velocity (UPV), impact-echo, or ground-penetrating radar (GPR). |
| **Component Boundaries** | Defect pixels inside vs outside user-supplied structural ROI mask. | Automated semantic element segmentation (girders, piers, bearings) or bridge BIM models. |
| **Rebar Section Loss** | Visual exposure flag (`ExposedRebars`), bounding box, surface area. | Residual steel bar diameter ($\Phi$), cross-sectional mass loss ($\%$), corrosion pitting depth (requires caliper / electrochemical testing). |
| **Concrete Delamination** | Visual spalling (`Spalling`), hollow areas (`Hollowareas`). | True acoustic sounding (chain dragging, rotary percussion) to detect subsurface unseparated voids. |
| **Structural Capacity Impact** | Visual presence of high-criticality defects. | Structural load rating, finite element analysis (FEA), dead/live load ratio, fracture-critical element status. |

---

## 5. Literature-Grounded Severity Framework

Prior to integrating automated severity scoring, we propose a transparent, four-tier hierarchy grounded in established bridge inspection standards:

### 5.1 Standards Grounding
1. **AASHTO Manual for Bridge Element Inspection (MBEI, 2019):**
   - **Condition State 1 (CS1 - Good):** Little to no deterioration.
   - **Condition State 2 (CS2 - Fair):** Minor cracking/corrosion within allowable limits.
   - **Condition State 3 (CS3 - Poor):** Spalling, corrosion section loss, active structural cracking.
   - **Condition State 4 (CS4 - Severe):** Advanced section loss affecting element structural integrity.
2. **FHWA Bridge Inspection Reference Manual (BIRM, 2012):** Defect rating guide for NBIS Items 58 (Deck), 59 (Superstructure), and 60 (Substructure).
3. **DIN 1076 / RI-EBW-PRÜF (Germany):** Condition grades 1.0 (very good) to 4.0 (inadequate). Evaluation of damage extent (*Schadensausmaß*) and damage intensity (*Schadensintensität*).
4. **FIB Model Code 2010 & Eurocode 2 (EN 1992-1-1):** Crack width design limit $w_{\text{max}} = 0.2\text{--}0.3\,\text{mm}$ for reinforced concrete under environmental exposure classes XD/XS.

### 5.2 Literature-Grounded Criticality Taxonomy (DACL10K Classes)

| Criticality Tier | Definition & Structural Mechanism | DACL10K Semantic Classes | AASHTO MBEI Analog |
| :--- | :--- | :--- | :--- |
| **Tier 1: Critical Structural** | Direct threat to load path, reinforcement continuity, or articulation. Rebar corrosion, concrete crushing, bearing seizure. | `ExposedRebars`, `Spalling`, `Rockpocket`, `Bearing`, `EJoint` | CS3 / CS4 |
| **Tier 2: Progressive Cracking** | Active pathways for water/chloride ingress, freeze-thaw degradation, potential shear/flexural fatigue cracks. | `Crack`, `ACrack`, `WConccor` (Weak Concrete) | CS2 / CS3 |
| **Tier 3: Superficial / Environmental** | Surface weathering, chemical leaching, cosmetic or early-stage indicators without immediate structural loss. | `Efflorescence`, `Rust`, `Weathering`, `Wetspot`, `Cavity`, `Hollowareas` | CS1 / CS2 |
| **Tier 4: Non-Structural / Auxiliary** | Non-structural appurtenances, formwork remnants, surface markings. | `Graffiti`, `Drainage`, `JTape`, `PEquipment`, `Restformwork` | Non-structural |

### 5.3 Required Data for Validation
To validate this framework in future work, the following ground-truth datasets are required:
1. **Certified Bridge Inspector Evaluations:** Paired National Bridge Inventory (NBI) condition ratings (0–9) or AASHTO element-level CS1–CS4 distributions.
2. **Physical Metric Ground Truth:** Calibrated laser rangefinder distance or fiducial scale markers for true metric conversion.
3. **NDT Depth Logs:** Core samples, GPR scans, or impact-echo sounding logs for subsurface validation.

---

## 6. JSON Output Schema

The pipeline produces consistent, machine-readable JSON outputs with complete spatial, geometric, and statistical metadata:

```json
{
  "metadata": {
    "image_id": "dacl10k_v2_validation_0001",
    "image_dimensions": {
      "height": 512,
      "width": 512,
      "total_pixels": 262144
    },
    "timestamp_utc": "2026-11-23T06:25:32.880171+00:00",
    "threshold": 0.5,
    "min_component_area_pixels": 10,
    "roi": {
      "present": true,
      "roi_pixel_area": 102400,
      "roi_relative_area_pct": 39.0625
    },
    "disclaimer": "OPTICAL MEASUREMENT NOTICE: All lengths, widths, perimeters, and areas are reported strictly in pixel-domain units..."
  },
  "summary": {
    "total_detected_classes": 2,
    "detected_class_names": ["Crack", "Spalling"],
    "total_components_count": 3,
    "total_defect_pixel_area": 2450,
    "total_defect_image_area_pct": 0.9346,
    "total_defect_roi_area_pct": 2.3925
  },
  "class_measurements": {
    "Crack": {
      "class_id": 3,
      "class_name": "Crack",
      "detected": true,
      "component_count": 1,
      "total_pixel_area": 450,
      "image_relative_area_pct": 0.17166,
      "roi_relative_area_pct": 0.43945,
      "confidence": {
        "mean": 0.8842,
        "min": 0.5120,
        "max": 0.9754,
        "median": 0.9123,
        "std": 0.0812
      },
      "components": [
        {
          "component_id": 1,
          "class_id": 3,
          "class_name": "Crack",
          "pixel_area": 450,
          "image_relative_area_pct": 0.17166,
          "roi_relative_area_pct": 0.43945,
          "pixels_inside_roi": 450,
          "centroid_xy": [152.4, 210.1],
          "bbox": {
            "xmin": 120,
            "ymin": 100,
            "xmax": 185,
            "ymax": 320,
            "width": 65,
            "height": 220,
            "xyxy": [120, 100, 185, 320],
            "xywh": [120, 100, 65, 220],
            "bbox_area_pixels": 14300
          },
          "confidence": { "mean": 0.8842, "min": 0.5120, "max": 0.9754, "median": 0.9123, "std": 0.0812 },
          "shape_features": {
            "perimeter_pixels": 462.1,
            "circularity": 0.0265,
            "aspect_ratio": 4.1250,
            "solidity": 0.0821,
            "extent": 0.0315,
            "orientation_deg": 88.5,
            "is_linear_defect": true,
            "skeleton_length_pixels": 218,
            "estimated_avg_width_pixels": 2.06
          }
        }
      ]
    }
  }
}
```

---

## 7. Verification & Test Results

The new Stage 2 module was thoroughly verified with 11 targeted unit tests in `tests/test_defect_measurement.py`. All 11 tests passed in 0.14s:

1. `test_canonical_classes_order_preserved`: Verified 19 classes match DACL10K canonical ordering.
2. `test_empty_mask_handling`: Verified zero detections do not crash and report 0% area.
3. `test_single_component_measurement_correctness`: Verified exact area, bounding box coordinates, and confidence statistics on a known $20 \times 30$ rectangle.
4. `test_multiple_connected_components_same_class`: Verified disjoint regions split into independent components.
5. `test_overlapping_classes_independence`: Verified simultaneous spatial overlap (e.g. `ExposedRebars` inside `Spalling`) measured independently across channels.
6. `test_noise_filtering_min_component_area`: Verified components $< A_{\text{min}}$ are cleanly filtered out.
7. `test_roi_mask_calculations`: Verified image-relative vs ROI-relative calculations.
8. `test_crack_skeleton_and_shape_features`: Verified morphological thinning, skeleton length, and average width estimation.
9. `test_json_schema_serialization_and_validity`: Verified strict JSON schema compliance.
10. `test_literature_grounded_severity_evaluator`: Verified tier mapping, highest criticality assignment, and validation parameter tracking.
11. `test_visualizer_overlay_generation`: Verified multi-class rendering, bounding box labels, banner stacking, and PNG export.

**Full Project Regression Suite:**  
All **49 tests** in the repository passed cleanly in 9.62s (`Ran 49 tests in 9.621s, OK`).

---

## 8. Instructions for Running Stage 2

### 8.1 Local Execution (Mac / Linux)

```bash
# Activate your environment
cd "/Users/kundankumar/Documents/untitled folder 2/bridge-ai-project"

# 1. Run unit tests
python3.11 -m unittest tests/test_defect_measurement.py -v

# 2. Run CLI measurement on synthetic test demonstration
python3.11 scripts/run_defect_measurement.py \
    --output-dir outputs/stage2_demo \
    --threshold 0.5 \
    --min-area 10

# 3. Run on a specific validation image using baseline checkpoint
python3.11 scripts/run_defect_measurement.py \
    --image data/raw/dacl10k_v2_devphase/images/validation/dacl10k_v2_validation_0001.jpg \
    --checkpoint experiments/segmentation_baseline/checkpoints/best_model.pth \
    --output-dir outputs/stage2_measurements \
    --threshold 0.5 \
    --min-area 10 \
    --device auto
```

### 8.2 Kaggle / Google Colab Execution

In Kaggle notebooks or Colab:

```python
# Cell 1: Pull latest repository changes
!git pull origin main

# Cell 2: Run verification tests
!python -m unittest tests/test_defect_measurement.py -v

# Cell 3: Batch measurement on a validation image directory
!python scripts/run_defect_measurement.py \
    --image-dir /kaggle/input/dacl10k-v2/dacl10k_v2_devphase/images/validation \
    --checkpoint /kaggle/input/bridge-ai-checkpoints/best_model.pth \
    --output-dir /kaggle/working/stage2_measurements \
    --threshold 0.5 \
    --min-area 10 \
    --device cuda
```
