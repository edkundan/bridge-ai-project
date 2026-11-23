# Stage 3: Explainable Visual Severity Assessment Foundation

**Project:** Explainable AI-Based Bridge Defect Severity Assessment and Risk-Based Maintenance Prioritization  
**Repository:** `bridge-ai-project`  
**Dataset:** DACL10K-v2 (19 semantic classes, multi-label segmentation)  
**Upstream Models:** DeepLabV3+ ResNet-34 (Baseline mIoU: 0.3384 | Exp 02 Weighted Loss: 0.3542 | Exp 03 Rare-Class Sampling: 0.3515)  
**Date:** November 2026  

---

## 1. Executive Summary & Problem Formulation

Stage 2 established the geometric measurement pipeline (extracting pixel footprints, bounding boxes, aspect ratios, and crack centerlines). However, practical deployment revealed four fundamental methodological flaws in naive severity assignment:

1. **Misleading Severity Tiering (`TIER_1_CRITICAL` on Superficial Defects):**  
   Classifying any detected `Spalling` as `TIER_1_CRITICAL` falsely asserts that the bridge or structural element is in a critical, unsafe condition. In civil engineering standards (AASHTO MBEI, FHWA BIRM, DIN 1076), a shallow concrete spall without deep section loss or fractured rebar is typically Condition State 2 (Fair) or 3 (Poor), never an immediate emergency.
2. **Non-Structural Distortion (e.g. 11.92% Graffiti vs 0.32% Spalling):**  
   Non-structural aesthetic markings (`Graffiti`, `Drainage`, `JTape`, `PEquipment`, `Restformwork`) cover large contiguous surface areas. Naive area summation causes cosmetic paint to drown out localized, high-consequence structural defects.
3. **Double Counting in Overlapping Multi-Label Predictions:**  
   When `ExposedRebars` sits inside a `Spalling` cavity, summing per-class pixel areas counts the overlapping footprint twice, misrepresenting physical damage extent.
4. **Fabrication of Uncalibrated Risk Scores:**  
   Without structural context (element load path, depth, metric scale, design load rating), predicting a numerical structural risk score or certified condition state is scientifically ungrounded and poses severe liability risks in civil infrastructure management.

Stage 3 resolves these challenges by introducing an **Explainable Visual Severity Assessment Foundation** that strictly decouples **observable visual evidence** from **official structural condition states**, enforces **explicit abstention (`NOT_ASSESSED`)**, and provides **human-in-the-loop review flags** with literature-grounded citations.

---

## 2. Mathematical & Methodological Formulation

### 2.1 Segregation of Structural vs. Non-Structural Footprints
Let $M_c \in \{0, 1\}^{H \times W}$ denote the binary segmentation mask for class $c \in \{0, \dots, 18\}$. We partition the 19 DACL10K classes into two disjoint sets:
- **Structural Classes ($\mathcal{C}_{\text{struct}}$):** `ExposedRebars`, `Spalling`, `Rockpocket`, `Bearing`, `EJoint`, `Crack`, `ACrack`, `WConccor`, `Efflorescence`, `Rust`, `Weathering`, `Wetspot`, `Cavity`, `Hollowareas` (14 classes).
- **Non-Structural Classes ($\mathcal{C}_{\text{non\_struct}}$):** `Graffiti`, `Drainage`, `JTape`, `PEquipment`, `Restformwork` (5 classes).

We compute the spatial union masks:

$$M_{\text{struct}}(y, x) = \bigvee_{c \in \mathcal{C}_{\text{struct}}} M_c(y, x), \quad M_{\text{non\_struct}}(y, x) = \bigvee_{c \in \mathcal{C}_{\text{non\_struct}}} M_c(y, x)$$

The respective image-relative area percentages are:

$$\alpha_{\text{struct}} = \frac{\sum_{y, x} M_{\text{struct}}(y, x)}{H \cdot W} \times 100\%, \quad \alpha_{\text{non\_struct}} = \frac{\sum_{y, x} M_{\text{non\_struct}}(y, x)}{H \cdot W} \times 100\%$$

In the real-image evaluation case, this separation guarantees:
$$\alpha_{\text{struct}} = 0.304\% \quad (\text{Spalling}), \quad \alpha_{\text{non\_struct}} = 12.064\% \quad (\text{Graffiti})$$
Non-structural graffiti coverage is isolated and cannot distort structural damage metrics.

### 2.2 Summed vs. Union Pixel Coverage
Let $A_c = \sum_{y, x} M_c(y, x)$ be the pixel area of class $c$. The overall union mask is:

$$M_{\text{union}}(y, x) = \bigvee_{c=0}^{18} M_c(y, x)$$

We explicitly distinguish:
1. **Summed Class Area:** $A_{\text{sum}} = \sum_{c=0}^{18} A_c$
2. **Union Footprint Area:** $A_{\text{union}} = \sum_{y, x} M_{\text{union}}(y, x)$
3. **Overlapping Defect Area:** $A_{\text{overlap}} = \sum_{y, x} \mathbf{1}\left[ \sum_{c=0}^{18} M_c(y, x) \ge 2 \right]$

When $A_{\text{overlap}} > 0$ (e.g. `ExposedRebars` inside `Spalling`), $A_{\text{sum}} > A_{\text{union}}$. Both values are reported to ensure mathematical consistency.

---

## 3. Visual Concern Levels vs. Official Structural Condition

Stage 3 replaces legacy "criticality tiers" with an observation-level classification:

```
                            Visual Inspection Imagery
                                       │
                                       ▼
       ┌───────────────────────────────────────────────────────────────┐
       │                Visual Concern Categorization                  │
       │                    (Observation-Level)                        │
       ├───────────────────────────────────────────────────────────────┤
       │ HIGH_VISUAL_CONCERN      : Cover breach, rebars, articulation │
       │ MODERATE_VISUAL_CONCERN  : Linear cracking, matrix pathways   │
       │ LOW_VISUAL_CONCERN       : Moisture, leaching, superficial    │
       │ NEGLIGIBLE_NON_STRUCTURAL: Graffiti, formwork, hardware       │
       │ NO_DEFECTS_DETECTED      : Clean inspection scan              │
       └───────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
       ┌───────────────────────────────────────────────────────────────┐
       │             Official Structural Condition State               │
       │                                                               │
       │                    STATE: NOT_ASSESSED                        │
       │                                                               │
       │  Explicit Abstention: Missing element load path, internal     │
       │  depth, metric scale, and physical rebar cross-section loss   │
       │  prevents legal certification of bridge structural safety.    │
       └───────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
       ┌───────────────────────────────────────────────────────────────┐
       │                 Structured Human Review Flags                 │
       │  - POSSIBLE_SPALLING_REVIEW       - REBAR_EXPOSURE_REVIEW     │
       │  - CRACK_DELAMINATION_REVIEW      - BEARING_REVIEW            │
       │  - HONEYCOMBING_REVIEW            - NON_STRUCTURAL_COVERAGE   │
       └───────────────────────────────────────────────────────────────┘
```

### Observation-Level Concern vs. Certified Condition States

| Framework Level | Terminology | Meaning | Authority |
| :--- | :--- | :--- | :--- |
| **Observation-Level (Stage 3 AI)** | `HIGH_VISUAL_CONCERN`<br>`MODERATE_VISUAL_CONCERN`<br>`LOW_VISUAL_CONCERN`<br>`NEGLIGIBLE_NON_STRUCTURAL` | Visual classification of optical defect morphology on surface concrete. | Proposed XAI Research Logic |
| **Official Structural Rating (Abstention)** | **`NOT_ASSESSED`** | System explicitly abstains from assigning AASHTO CS1–CS4 or NBIS 0–9 condition ratings. | Authoritative Bridge Engineering Standard Requirement |

---

## 4. Observable Evidence vs. Unobservable Parameters

| Dimension | Directly Observable from 2D RGB (Stage 3) | Unobservable / Missing Engineering Data (Checklist) |
| :--- | :--- | :--- |
| **Defect Geometry** | 2D pixel area, bounding box, skeleton length, aspect ratio, circularity, solidity. | **Metric dimensions (mm, cm, m):** Requires optical rangefinder distance or metric camera calibration. |
| **Defect Depth** | **NOT OBSERVABLE** (surface pixels only). | **Internal depth & delamination:** Requires ultrasonic pulse velocity (UPV), impact echo, or GPR. |
| **Reinforcement State** | Visual rebar exposure flag (`ExposedRebars`), surface bounding box. | **Steel section loss:** Requires mechanical caliper measurement of residual bar diameter or electrochemical half-cell potential. |
| **Element Structural Role** | Visual presence of components inside user-supplied ROI mask. | **Component hierarchy:** Primary fracture-critical girder vs. secondary barrier/parapet (requires Bridge BIM or NBI inventory). |
| **Structural Impact** | Morphological visual concern level. | **Live-load capacity:** Requires finite element structural analysis (FEA) and design load rating. |
| **Model Confidence** | Mean pixel classification probability ($P \in [0, 1]$). | **NOT EQUAL TO SEVERITY CONFIDENCE:** High optical confidence ($P=0.95$) simply means clear image pixels, not high probability of bridge failure. |

---

## 5. Structured Review Flags & Literature Citations

When visual concern triggers review, the system generates structured review flags with precise authoritative citations:

1. **`POSSIBLE_SPALLING_REVIEW`**  
   - *Physical Mechanism:* Concrete cover delamination and spalling driven by rebar corrosion expansion or freeze-thaw cycles.  
   - *Literature Citation:* AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1080); FHWA BIRM (2012) Section 7.2.2; DIN 1076 Section 5.3.  
   - *Inspector Action Items:* Sound concrete boundaries with acoustic percussion hammer; check cavity base for corroded steel; measure cavity depth with depth gauge.
2. **`REBAR_EXPOSURE_REVIEW`**  
   - *Physical Mechanism:* Loss of cover concrete exposing tensile/shear steel to atmospheric moisture and deicing salts.  
   - *Literature Citation:* AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1090); FHWA BIRM (2012) Section 7.2.3; Eurocode 2 (EN 1992-1-1 Section 4).  
   - *Inspector Action Items:* Measure residual bar diameter with vernier caliper to calculate percentage cross-sectional loss; check for debonding length and pitting corrosion.
3. **`CRACK_DELAMINATION_REVIEW`**  
   - *Physical Mechanism:* Tensile stress exceeding concrete tensile capacity, forming pathways for moisture ingress.  
   - *Literature Citation:* AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1130); FIB Model Code 2010 Section 7.7.3 ($w_{\lim} = 0.2\text{--}0.3\,\text{mm}$); Eurocode 2 (EN 1992-1-1 Section 7.3).  
   - *Inspector Action Items:* Measure physical crack width using optical comparator card; determine whether pattern is flexural, shear, or map cracking (ASR); assess active vs. dormant status.
4. **`NON_STRUCTURAL_SURFACE_COVERAGE`**  
   - *Physical Mechanism:* Aesthetic paint markings or temporary hardware on concrete face.  
   - *Literature Citation:* FHWA BIRM (2012) Section 2.1 (Aesthetic / Appurtenance condition).  
   - *Inspector Action Items:* Confirm surface paint does not conceal underlying structural cracking; route to municipal aesthetic cleanup team (non-structural priority).

---

## 6. Machine-Readable JSON Schema (Stage 3 & Stage 4 Contract)

```json
{
  "image_id": "graffiti_spalling_case",
  "visual_severity_assessment": {
    "visual_concern_level": "HIGH_VISUAL_CONCERN",
    "official_structural_condition_state": "NOT_ASSESSED",
    "abstention_reason": "OFFICIAL CONDITION STATE ABSTENTION: An official bridge condition state cannot be established from 2D optical imagery alone...",
    "review_flags": [
      "NON_STRUCTURAL_SURFACE_COVERAGE",
      "POSSIBLE_SPALLING_REVIEW"
    ]
  },
  "coverage_breakdown": {
    "summed_class_pixel_area": 32170,
    "summed_class_image_area_pct": 12.2718,
    "union_pixel_area": 32170,
    "union_defect_image_area_pct": 12.2718,
    "structural_union_pixel_area": 841,
    "structural_union_image_area_pct": 0.3208,
    "non_structural_union_pixel_area": 31329,
    "non_structural_union_image_area_pct": 11.951,
    "overlapping_defect_pixel_area": 0,
    "overlapping_defect_image_area_pct": 0.0,
    "roi_coverage": {
      "roi_available": false,
      "summed_class_roi_pixels": null,
      "union_defect_roi_area_pct": null,
      "structural_union_roi_area_pct": null,
      "non_structural_union_roi_area_pct": null
    }
  },
  "confidence_contract": {
    "segmentation_model_confidence_mean": 0.945,
    "calibrated_structural_severity_confidence": null,
    "confidence_distinction_notice": "CONFIDENCE CONTRACT: The reported confidence score is the mean pixel-level classification probability... NOT calibrated engineering confidence in structural severity."
  },
  "missing_data_checklist": {
    "structural_component_identity": { "status": "MISSING", "required_source": "Bridge BIM, element segmentation..." },
    "element_importance": { "status": "MISSING", "required_source": "National Bridge Inventory (NBI)..." },
    "physical_defect_dimensions": { "status": "MISSING", "required_source": "Optical rangefinder, calibrated stereo rig..." },
    "internal_defect_depth": { "status": "MISSING", "required_source": "Non-destructive testing (GPR, UPV)..." },
    "reinforcement_section_loss": { "status": "MISSING", "required_source": "Physical caliper measurement..." },
    "inspection_history": { "status": "MISSING", "required_source": "AASHTO BrM database..." },
    "engineering_assessment": { "status": "MISSING", "required_source": "Licensed PE field evaluation..." }
  },
  "observations": [
    {
      "finding_id": "FINDING-SPALL-001",
      "class_name": "Spalling",
      "class_id": 15,
      "component_id": 1,
      "visual_concern_level": "HIGH_VISUAL_CONCERN",
      "is_structural": true,
      "pixel_area": 841,
      "image_relative_extent_pct": 0.3208,
      "roi_relative_extent_pct": null,
      "bbox": { "xmin": 104, "ymin": 144, "xmax": 136, "ymax": 176, "width": 32, "height": 32 },
      "confidence_stats": { "mean": 0.93, "min": 0.93, "max": 0.93, "median": 0.93, "std": 0.0 },
      "shape_features": { "perimeter_pixels": 108.0, "circularity": 0.908, "aspect_ratio": 1.0, "is_linear_defect": false },
      "visual_evidence": "Detected Spalling component of 841 px (0.3208% of image). Spatial centroid at (120.0, 160.0)...",
      "explanation": "Spalling represents the physical loss of surface concrete cover...",
      "literature_reference": "AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1080); FHWA BIRM (2012) Section 7.2.2.",
      "inspector_action_items": [
        "Perform acoustic hammer sounding around defect boundaries...",
        "Inspect the base of the cavity for exposed rebar...",
        "Measure physical cavity depth with depth gauge."
      ],
      "triggered_review_flags": ["POSSIBLE_SPALLING_REVIEW"]
    }
  ],
  "stage_4_prioritization_contract": {
    "ready_for_automated_prioritization": false,
    "abstention_reason": "Automated maintenance priority and repair scheduling cannot be assigned from visual features alone...",
    "candidate_review_flags": ["NON_STRUCTURAL_SURFACE_COVERAGE", "POSSIBLE_SPALLING_REVIEW"],
    "primary_visual_concerns": ["Spalling"],
    "structural_union_coverage_pct": 0.3208,
    "non_structural_union_coverage_pct": 11.951,
    "actionable_observations_count": 2
  }
}
```

---

## 7. Verification & Regression Outcomes

The Stage 3 implementation was verified through targeted unit tests and full project regression testing:
1. **Stage 3 Dedicated Tests (`tests/test_explainable_severity.py`):**
   - 10 comprehensive tests covering Graffiti/Spalling segregation, summed vs. union pixel areas, explicit abstention, missing data checklist completeness, review flag triggers, XAI explanation generation, Stage 4 contract, and ROI unavailability.
   - **Result:** `10/10 PASSED` in 0.764s.
2. **Stage 2 Regression Tests (`tests/test_defect_measurement.py`):**
   - 11 unit tests covering geometry, skeletonization, connected components, and legacy tier compatibility.
   - **Result:** `11/11 PASSED` in 0.127s.
3. **Full Project Regression Test Suite:**
   - Ran `python3.11 -m unittest discover -s tests -v` covering checkpoint selection, resume logic, weighted losses, rare-class sampling, pipeline conversions, runtime recovery, and Stage 2/3 measurement.
   - **Result:** `59/59 PASSED` in 11.106s (zero regressions across all 59 project tests).

---

## 8. Limitations & Future Expert-Validation Requirements

Before Stage 4 (Risk-Based Maintenance Prioritization) can transition from review flagging to automated decision support, the following data must be collected and integrated:
1. **Component-Level Hierarchy Labels:** Multi-task segmentation of structural bridge elements (e.g., pier column, prestressed girder, deck slab, abutment wall) to distinguish primary fracture-critical components from secondary appurtenances.
2. **In-Situ Metric Scale Calibration:** Paired stereo camera telemetry or laser rangefinder distance logs to convert pixel lengths into physical crack widths ($w$ in mm) and spall areas ($A$ in $\text{m}^2$).
3. **Non-Destructive Testing (NDT) Depth Logs:** Ultrasonic pulse velocity (UPV), impact echo, or ground-penetrating radar (GPR) sounding scans to detect subsurface delamination depth.
4. **Historical Maintenance Records:** Longitudinal inspection logs from DOT databases (AASHTO BrM) tracking defect progression rates over 5–20 year intervals.
5. **Certified Bridge Inspector Validation:** Formal blind evaluation by licensed Professional Engineers (PE / SE) comparing AI-generated review flags against physical on-site National Bridge Inspection Standards (NBIS) reports.
