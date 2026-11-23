# Stage 4: Explainable, Evidence-Aware Maintenance Prioritization Foundation

**Project:** Explainable AI-Based Bridge Defect Severity Assessment and Risk-Based Maintenance Prioritization  
**Repository:** `bridge-ai-project`  
**Dataset:** DACL10K-v2 (19 semantic classes, multi-label segmentation)  
**Upstream Stages:**
- Stage 1: Segmentation Models (Baseline mIoU: 0.3384 | Exp 02 Weighted Loss: 0.3542 | Exp 03 Rare-Class Sampling: 0.3515)
- Stage 2: Geometric Defect Measurement Pipeline (Footprints, bounding boxes, aspect ratios, crack centerlines)
- Stage 3: Explainable Visual Severity Assessment Foundation (Structural vs. non-structural partitioning, union coverage, human review flags, `NOT_ASSESSED` condition state)
**Date:** November 2026  

---

## 1. Executive Summary & Problem Formulation

In civil infrastructure management, automated computer vision systems often conflate **visual defect detection** with **statutory bridge maintenance prioritization**. Computer vision models operating on monocular 2D images cannot observe:
1. Internal structural load paths, boundary conditions, or redundancy (e.g., tension flange vs. compression zone).
2. Defect penetration depth, volumetric loss, or delamination planes beneath concrete cover.
3. Metric physical dimensions without calibrated laser LiDAR, stereoscopic targets, or photogrammetric ground truth.
4. Material degradation kinematics, active crack propagation rates, or environmental chemical exposure (e.g., deicing salts, carbonation).
5. Formal engineering accountability under statutory jurisdiction (Professional Engineer PE/SE sign-off, DOT standard operating procedures).

Consequently, directly issuing statutory work orders, automated repair deadlines, or AASHTO condition ratings from raw pixels is technically invalid, methodologically irresponsible, and legally impermissible under AASHTO and FHWA inspection guidelines.

**Stage 4 resolves this gap by establishing an Explainable, Evidence-Aware Maintenance Prioritization Foundation** that:
1. Enforces a **Strict Dual-Output Contract**:
   - **Output A: Inspection Review Queue:** A provisional, observation-level ranking designed to triage field inspectors to regions of interest (incorporating visual concern, physical extent, optical uncertainty, and review flags).
   - **Output B: Engineering Maintenance Priority:** Strictly **`NOT_ASSESSED`** under production policy, enforcing explicit abstention until certified physical engineering data and DOT-adopted decision policies are provided.
2. Quarantines non-structural aesthetic defects (e.g., `Graffiti`, `Drainage`, `PEquipment`) so they can never inflate structural urgency or outrank structural deterioration.
3. Detects optical uncertainties and physical contradictions between visual evidence and engineering metadata.
4. Provides an extensible policy interface (`BasePrioritizationPolicy`) separating certified production standards from isolated academic what-if simulations (`ResearchDemonstrationPolicy`).
5. Supports traceable human inspector override audit trails.

---

## 2. Requirement 9 Resolution: Set-Theoretic Coverage Verification

During review of synthetic test cases in Stage 3, a discrepancy appeared in documentation text between two statements:
- Stated snippet: $797 + 31,625 = 32,379$
- Underlying exact sum: $797 + 31,626 = 32,423$

### 2.1 Audit of Underlying JSON Artifacts
An audit of the persisted machine-generated JSON artifact `outputs/stage3_demo/graffiti_spalling_case_measurement.json` revealed:
```json
{
  "structural_union_pixel_area": 797,
  "non_structural_union_pixel_area": 31626,
  "overlapping_defect_pixel_area": 0,
  "union_defect_pixel_area": 32423
}
```
The exact mathematical identity holds:
$$797 + 31,626 - 0 = 32,423 \quad (\text{Exact Identity Verified})$$

### 2.2 Root-Cause Identification
The reported number $32,379$ was **purely a typographical error in the markdown prose summary** of the conversational turn, while the underlying Python execution code in `src/measurement/geometry.py` and `src/measurement/severity.py` was mathematically exact.

### 2.3 Set-Theoretic Mathematical Proof & Regression Invariant
For any two defect masks $A, B \subset \Omega$ where $\Omega = \{1, \dots, H\} \times \{1, \dots, W\}$:
$$|A \cup B| = |A| + |B| - |A \cap B|$$
In general, for $K$ classes with binary masks $M_1, \dots, M_K$:
$$|\bigcup_{k=1}^K M_k| \le \sum_{k=1}^K |M_k|$$
with equality if and only if all class masks are pairwise disjoint ($\forall i \neq j, M_i \cap M_j = \emptyset$).

To prevent future regression, a permanent unit test `test_set_theoretic_coverage_consistency_regression` was added to `tests/test_maintenance_prioritization.py`. It tests:
1. **Disjoint Case:** $|A| = 797, |B| = 31,626, |A \cap B| = 0 \implies |A \cup B| = 32,423$.
2. **Overlapping Case:** $A$ and $B$ sharing an overlapping subregion of $500$ pixels:
   $$|A| = 1,000, |B| = 2,000, |A \cap B| = 500 \implies |A \cup B| = 2,500 = 1,000 + 2,000 - 500$$
All assertions hold down to the single pixel.

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Dual-Output Contract Architecture
Stage 4 ingests the Stage 3 Explainable Severity JSON report and optional verified engineering context, producing two strictly isolated outputs:

```mermaid
flowchart TD
    S3[Stage 3 Report<br/>Findings, Masks, Review Flags] --> S4Pipeline[Stage 4 Maintenance Prioritization Pipeline]
    VEC[Verified Engineering Context<br/>Component, Load Path, Calibrated mm, Depth, PE Sign-Off] --> S4Pipeline
    
    S4Pipeline --> PolCheck{Policy Selector}
    
    PolCheck -->|ProductionAbstentionPolicy| PPolicy[Production Decision Policy]
    PolCheck -->|ResearchDemonstrationPolicy| RPolicy[Demonstration Simulation Policy]
    
    PPolicy --> OutputA[Output A: Inspection Review Queue<br/>Provisional Visual Ranking for Field Verification]
    PPolicy --> OutputB[Output B: Engineering Maintenance Priority<br/>Strict NOT_ASSESSED Abstention + Missing Checklist]
    
    RPolicy --> OutputA
    RPolicy --> OutputB_Demo[Output B: Simulated Priority<br/>ASSESSED_RESEARCH_DEMO_ONLY + Prominent Disclaimers]
```

### 3.2 Output A: Composite Verification Scoring Formulation
Within `ProductionAbstentionPolicy`, findings are triaged for human inspection using a multi-criteria composite score $S \in [0, 1]$:

$$S = w_1 \cdot W_{\text{concern}} + w_2 \cdot \text{NormalizedExtent} + w_3 \cdot \text{Confidence} + w_4 \cdot W_{\text{criticality}}$$

Where:
- $w_1 = 0.40$ (Visual defect concern weight)
- $w_2 = 0.25$ (Normalized image footprint area $\min(\frac{\text{area}}{2500}, 1.0)$)
- $w_3 = 0.20$ (Optical segmentation model confidence score)
- $w_4 = 0.15$ (Element structural load path criticality weight)

#### Concern Weights ($W_{\text{concern}}$):
- High-consequence structural defects:
  - `ExposedRebars`: $1.00$
  - `Spalling`: $0.85$
  - `Crack`: $0.80$
  - `ACrack`: $0.80$
  - `Cavity`: $0.75$
- Medium structural defects:
  - `Rust`: $0.60$
  - `Efflorescence`: $0.50$
  - `WConccor`: $0.50$
  - `Rockpocket`: $0.45$
  - `Wetspot`: $0.35$
- Non-structural and aesthetic categories:
  - `Graffiti`: $0.05$ (Quarantined)
  - `Drainage`: $0.05$
  - `PEquipment`: $0.05$
  - `JTape`: $0.05$
  - `Restformwork`: $0.05$

#### Element Criticality Weights ($W_{\text{criticality}}$):
- `PRIMARY_LOAD_BEARING` (e.g., fracture-critical tension girder, pier column): $1.00$
- `SECONDARY_STRUCTURAL` (e.g., cross-frame, diaphragms): $0.65$
- `NON_REDUNDANT_SUBSTRUCTURE`: $0.90$
- `SUPERSTRUCTURE_DECK`: $0.50$
- `NON_STRUCTURAL` / `UNKNOWN`: $0.20$

#### Urgency Categorization:
- If finding is Non-Structural: $\implies$ `INFORMATIONAL_NON_STRUCTURAL`
- Else if $S \ge 0.65$ or contains `REBAR_EXPOSURE_REVIEW`: $\implies$ `URGENT_INSPECTION_REVIEW`
- Else if $S \ge 0.40$: $\implies$ `PRIORITY_INSPECTION_REVIEW`
- Else: $\implies$ `ROUTINE_INSPECTION_REVIEW`

### 3.3 Optical Uncertainty Categorization
Because automated segmentation masks on 2D images suffer from sensor resolution, blur, and occlusions, each finding is assigned an optical uncertainty level:
- **`HIGH_OPTICAL_UNCERTAINTY`:** Mean probability $< 0.70$ or footprint area $< 25$ pixels. Action: Requires manual close-up inspection and physical re-measurement.
- **`MODERATE_OPTICAL_UNCERTAINTY`:** Mean probability $< 0.85$.
- **`LOW_OPTICAL_UNCERTAINTY`:** High-confidence ($> 0.85$) clear visual pattern.

### 3.4 Output B: Statutory Maintenance Abstention Architecture
`ProductionAbstentionPolicy` enforces that `EngineeringMaintenancePriority.status` **must remain `NOT_ASSESSED`** whenever any required engineering prerequisite is missing.

The statutory prerequisites checklist includes:
1. Verified structural component identity (BIM or bridge inspection record).
2. Element load path criticality (AASHTO / NBI element classification).
3. Physical metric defect dimensions (calibrated optical/laser measurement in mm).
4. Internal defect depth & delamination plane (NDT sounding, GPR, or ultrasonic pulse velocity).
5. Formal validation and sign-off by a licensed Professional Engineer (PE/SE).
6. Formally adopted Department of Transportation (DOT) maintenance decision policy.

When prerequisites are missing, the system outputs:
```json
{
  "status": "NOT_ASSESSED",
  "official_repair_priority": "NOT_ASSESSED",
  "official_repair_deadline": null,
  "official_structural_safety_rating": null,
  "official_aashto_condition_state": null,
  "abstention_reason": "ENGINEERING MAINTENANCE PRIORITY ABSTENTION: Actionable bridge maintenance priority cannot be certified from monocular computer vision alone...",
  "missing_engineering_prerequisites": [
    "Verified structural component identity (BIM or inspection plan)",
    "Element load path criticality (AASHTO / NBI element classification)",
    "Physical metric defect dimensions (calibrated optical/laser measurement)",
    "Internal defect depth & delamination plane (NDT sounding, GPR, or UPV)",
    "Formal validation and sign-off by a licensed Professional Engineer (PE/SE)",
    "Formally adopted Department of Transportation (DOT) maintenance decision policy"
  ]
}
```

### 3.5 Contradiction Detection
The policy evaluates cross-domain consistency between visual observations and verified context:
- If visual observations detect `ExposedRebars` but the verified component is non-structural (e.g., sound barrier, timber walkway), a high-priority contradiction alert is raised:
  `"CONTRADICTION: Rebar exposure detected on element marked as NON_STRUCTURAL."`
- If visual extent indicates extensive spalling ($> 5,000$ px) but context records `ELEMENT_INTACT`, a severe discrepancy alert is recorded.

---

## 4. Civil Engineering Literature Grounding

The design of Stage 4 is grounded in four major civil infrastructure standards:

1. **AASHTO Manual for Bridge Element Inspection (MBEI, 2nd Edition 2019):**  
   AASHTO defines bridge element condition states (CS1 Good, CS2 Fair, CS3 Poor, CS4 Severe). Under Section 2.4, assignment to CS3 or CS4 for concrete spalling or delamination requires physical assessment of rebar section loss, bond failure, and depth relative to cover. 2D computer vision without depth measurement cannot discriminate between CS2 (minor surface spalling $< 25$ mm depth) and CS3/CS4 (deep spalling affecting reinforcing steel).
2. **FHWA Bridge Inspector's Reference Manual (BIRM):**  
   BIRM Chapter 13 emphasizes that cosmetic defects (paint deterioration, graffiti, surface efflorescence) must never be aggregated into element structural health evaluations. BIRM mandates physical sounding (chain drag, hammer sounding) to establish true delamination boundaries.
3. **DIN 1076 (Ingenieurbauwerke im Zuge von Straßen und Wegen - Überwachung und Prüfung):**  
   The German bridge inspection standard mandates strict segregation between *Erhaltungszustand* (maintenance condition) and *Tragsicherheit* (load-carrying safety). Visual defect observations trigger focused special inspections (*Sonderprüfung*) rather than immediate statutory weight or speed restrictions.
4. **FIB Model Code for Concrete Structures (2010 / 2020):**  
   FIB guidelines highlight that corrosion rate and structural capacity reduction depend non-linearly on concrete cover depth, chloride penetration profiles, and moisture availability. Monocular RGB features cannot assess these chemical-physical transport phenomena.

---

## 5. Experimental Verification & Test Outcomes

### 5.1 Unit and Integration Test Matrix
A comprehensive suite of 9 tests was authored in `tests/test_maintenance_prioritization.py` and executed against the codebase:

| Test Name | Verified Property | Status |
|:---|:---|:---:|
| `test_dual_output_contract` | Verifies Output A and Output B exist with proper types | **PASSED** |
| `test_production_abstention_without_engineering_context` | Confirms status is `NOT_ASSESSED` and lists 6 prerequisites | **PASSED** |
| `test_graffiti_isolation_cannot_dominate_queue` | Confirms 12% Graffiti is ranked below 0.3% Spalling | **PASSED** |
| `test_optical_uncertainty_identification` | Verifies low confidence / small area triggers high uncertainty | **PASSED** |
| `test_contradiction_detection` | Detects rebar exposure on non-structural component | **PASSED** |
| `test_human_override_workflow` | Records and applies PE human override and audit trail | **PASSED** |
| `test_research_demonstration_policy_flagged` | Verifies clear simulation disclaimer and isolated policy | **PASSED** |
| `test_clean_scan_no_defects` | Verifies empty queue on defect-free scans | **PASSED** |
| `test_set_theoretic_coverage_consistency_regression` | Proves $\|A \cup B\| = \|A\| + \|B\| - \|A \cap B\|$ exactly | **PASSED** |

### 5.2 Full Repository Regression
Running `python3.11 -m unittest discover -s tests -v`:
$$\mathbf{68 \text{ out of } 68 \text{ tests PASSED in } 8.15\text{s}}$$
All legacy Stage 1 (segmentation losses, dataset pipeline), Stage 2 (measurement), Stage 3 (explainable visual severity), and Stage 4 tests remain 100% green and backward-compatible.

### 5.3 Real-Image Verification: Graffiti + Spalling Case
We executed `scripts/run_maintenance_prioritization.py` on the real bridge inspection image containing 12.06% Graffiti and 0.30% Spalling:

**Output A (Inspection Review Queue):**
- **Rank #1:** `[FINDING-SPALL-002]` `Spalling`
  - Extent: $797$ px ($0.304\%$)
  - Verification Score: $0.7365$
  - Queue Urgency: `URGENT_INSPECTION_REVIEW`
  - Optical Uncertainty: `LOW`
  - Recommended Inspector Action: *"Conduct physical hammer sounding to locate subsurface delaminations; measure depth and inspect for exposed rebar."*
- **Rank #2:** `[FINDING-GRAFF-001]` `Graffiti`
  - Extent: $31,626$ px ($12.064\%$)
  - Verification Score: $0.0490$
  - Queue Urgency: `INFORMATIONAL_NON_STRUCTURAL`
  - Optical Uncertainty: `MODERATE`
  - Recommended Inspector Action: *"Aesthetic / non-structural surface marking. Route to routine cosmetic maintenance queue if applicable."*

**Output B (Engineering Maintenance Priority):**
- Status: `NOT_ASSESSED`
- Official Repair Priority: `NOT_ASSESSED`
- Official Repair Deadline: `None`
- Official AASHTO State: `None`
- Missing Prerequisites: 6 items (in production mode without verified context).

---

## 6. Software Architecture & API Specifications

The Stage 4 module is packaged cleanly in `src/prioritization/`:
- `src/prioritization/schema.py`: Pydantic/dataclass schema definitions (`ReviewQueueItem`, `InspectionReviewQueue`, `EngineeringMaintenancePriority`, `VerifiedEngineeringContext`, `HumanOverrideRecord`, `Stage4PrioritizationResult`).
- `src/prioritization/policies.py`: Abstract `BasePrioritizationPolicy`, `ProductionAbstentionPolicy`, and `ResearchDemonstrationPolicy`.
- `src/prioritization/engine.py`: `MaintenancePrioritizationPipeline` orchestrating execution, parsing, and artifact persistence.
- `scripts/run_maintenance_prioritization.py`: Production CLI runner supporting `--stage3-json`, `--engineering-context`, `--demo-policy`, and `--output-dir`.

---

## 7. Conclusions & Research Roadmap

Stage 4 establishes a methodologically sound foundation for bridge maintenance prioritization. By decoupling provisional visual triage from statutory engineering work orders, the pipeline eliminates false liability, protects against cosmetic defect distortions, and respects the non-negotiable boundaries of monocular computer vision.

Future research expansions (Stage 5) will explore:
1. Integration with terrestrial laser scanning (TLS) and photogrammetric point clouds for automated metric depth calibration.
2. Direct connection to municipal bridge inventory databases via standardized OpenBridge / IFC-Bridge schemas.
3. Multi-temporal change detection across consecutive annual inspection flight passes to measure physical progression rates.
