# Stage 5: Interactive Explainable AI Bridge Inspection Dashboard

**Project:** Explainable AI-Based Bridge Defect Severity Assessment and Risk-Based Maintenance Prioritization  
**Repository:** `bridge-ai-project`  
**Framework:** Streamlit (Python 3.11)  
**Dataset:** DACL10K-v2 (19 semantic classes, multi-label segmentation)  
**Upstream Modules:**
- Stage 1: Segmentation Models (DeepLabV3+ ResNet-34)
- Stage 2: Geometric Defect Measurement Pipeline (`src.measurement`)
- Stage 3: Explainable Visual Severity Assessment Foundation (`src.measurement.severity`)
- Stage 4: Maintenance Prioritization & Statutory Abstention Foundation (`src.prioritization`)
**Date:** November 2026  

---

## 1. Executive Summary & Objective

Civil infrastructure inspection requires intuitive, explainable, and accountable tools. While Stage 1 through Stage 4 established the theoretical, mathematical, and algorithmic foundations, practical deployment requires an interactive interface where bridge inspectors, structural engineers, and asset managers can:
1. Visually explore multi-label defect overlays and localize specific damage findings.
2. Review quantitative pixel-domain geometric measurements without confusing uncalibrated pixels with metric millimeters.
3. Inspect deterministic, rule-based explanations for why specific findings are prioritized over others.
4. Interact with the **Inspection Review Queue (Output A)** while respecting the **Statutory Maintenance Abstention (Output B)** boundaries.
5. Export structured machine-readable JSON data and human-readable inspection markdown reports.

Stage 5 delivers the **Interactive Explainable AI Bridge Inspection Dashboard**, built with **Streamlit** for seamless integration with our native PyTorch and OpenCV pipelines.

---

## 2. Dashboard Architecture & Directory Structure

The Stage 5 codebase is organized modularly to decouple presentation from pipeline logic:

```
bridge-ai-project/
├── app.py                            # Root launcher forwarder (streamlit run app.py)
├── requirements-dashboard.txt        # Dashboard dependencies
├── scripts/
│   └── launch_dashboard.sh           # Local shell launcher with configurable port
├── demo_data/                        # Self-contained saved demonstration cases
│   ├── dacl10k_v2_train_0722/        # Real bridge image with 19 findings (7 struct, 12 non-struct)
│   ├── dacl10k_v2_validation_0000/   # Benchmark validation case
│   ├── graffiti_spalling_case/       # 12% Graffiti vs 0.3% Spalling case
│   └── synthetic_demo_image/         # Controlled crack, spall, rebar benchmark
├── src/
│   └── dashboard/
│       ├── __init__.py               # Public API exports
│       ├── app.py                    # Main Streamlit application
│       ├── components.py             # UI render components (cards, tables, callouts, disclaimers)
│       └── data_loader.py            # Case discovery, disk loaders & live pipeline runner
└── tests/
    └── test_dashboard.py             # Integration test suite (6 targeted tests)
```

---

## 3. Core Functional Capabilities

### 3.1 Dual Execution Modes

The dashboard provides two primary operating modes in the sidebar:
1. **Saved-Results Demonstration Mode (No GPU Required):**  
   Instantly loads precomputed, verified inspection cases from `demo_data/`. Runs in $< 100\,\text{ms}$ on any standard CPU or laptop without requiring PyTorch model checkpoints or CUDA hardware.
2. **End-to-End Live Inference Mode (DeepLabV3+):**  
   Accepts user-uploaded bridge images (JPG/PNG) or selected benchmark images. Runs letterbox preprocessing, model inference on configurable devices (`auto`, `mps`, `cuda`, `cpu`), connected component labeling, skeletonization, and prioritization in real time.

---

### 3.2 Real-Image Integration & Demonstration Cases

The dashboard includes four ready-to-run demonstration cases:

| Case ID | Type | Findings | Structural | Non-Structural | Key Demonstration Highlight |
|:---|:---:|:---:|:---:|:---:|:---|
| **`dacl10k_v2_train_0722`** | Real Image | **19** | **7** | **12** | **Full 19-finding stress test:** Real bridge image exhibiting structural degradation (spalling, weathering, rust) alongside extensive non-structural markings (graffiti, restformwork). |
| **`dacl10k_v2_validation_0000`** | Real Image | **1** | **0** | **1** | **DACL10K Validation Benchmark:** Non-structural marking on bridge abutment demonstrating informational classification without structural alarm. |
| **`graffiti_spalling_case`** | Real Image | **2** | **1** | **1** | **Non-structural isolation:** 12.06% Graffiti vs. 0.30% Spalling. The 0.30% Spall is prioritized at Rank #1; Graffiti is quarantined at Rank #2. |
| **`synthetic_demo_image`** | Synthetic | **3** | **3** | **0** | **Multi-defect spatial overlap:** Spall with exposed rebar cavity and crack centerline skeletonization. |

---

### 3.3 Visual Inspection & High-Resolution Callouts

The image viewer section provides three interactive tabs:
- **🎨 Multi-Class Defect Overlay:** Renders semi-transparent color-coded masks ($45\%$ opacity) for all active classes, overlaid with bounding boxes and component ID tags.
- **📷 Original Bridge Image:** Displays unoccluded monocular RGB imagery.
- **🔍 Selected Finding Callout:** When a finding is selected from the review queue, the dashboard automatically crops a padded high-resolution region around the finding, highlights the bounding box in neon color (Red for structural, Yellow for non-structural), and displays finding-specific geometry and confidence metrics.

---

### 3.4 Quantitative Measurements & Metric Guardrails

The measurements panel reports:
- Total findings count, detected classes count ($K \le 19$).
- Individual class pixel areas ($A_c$), image-relative coverage ($\alpha_c = A_c / A_{\text{img}} \times 100\%$).
- Disjoint union areas: Structural union ($A_{\text{struct}}$) vs. Non-structural union ($A_{\text{non\_struct}}$).
- Overlapping defect area ($A_{\text{overlap}}$).
- Crack morphological geometry: Axis-aligned vs. oriented bounding box aspect ratios, circularity ($4\pi A / P^2$), solidity ($A / A_{\text{hull}}$), centerline skeleton length (`px`), and average estimated width (`px`).

> [!IMPORTANT]
> **Metric Calibration Guardrail:**  
> All geometric measurements are strictly reported in **pixel units (`px`)**. The dashboard explicitly disclaims metric millimeter (`mm`) claims unless external optical calibration, laser rangefinder depth, or physical scale targets are provided.

---

### 3.5 Rule-Based Explainability & Field Guidance

The explainability inspector provides deterministic explanations of pipeline decisions:
- **Rule-Based Justifications:** Decoupled from internal black-box saliency maps (e.g., Grad-CAM, SHAP). Displays explicit policy rule triggers (e.g., `RULE_HIGH_CONCERN_RELIABLE`, `RULE_NON_STRUCTURAL_QUARANTINE`).
- **Composite Score Breakdown:** Dissects the multi-criteria ranking formula:
  $$S = 0.40 \cdot W_{\text{concern}} + 0.25 \cdot \text{NormalizedExtent} + 0.20 \cdot \text{Confidence} + 0.15 \cdot W_{\text{criticality}}$$
- **Optical Uncertainty Analysis:** Evaluates whether model confidence ($< 0.70$) or pixel footprint ($< 25\,\text{px}$) warrants a `HIGH` optical uncertainty warning.
- **Recommended Field Inspector Actions:** Actionable NDT procedures (e.g., "Acoustic percussion hammer sounding to map delaminations", "Vernier caliper measurement of residual bar diameter", "Optical crack comparator survey").

---

### 3.6 Output B: Statutory Maintenance Priority Abstention

The maintenance assessment panel reinforces structural engineering ethics:
- **Status Display:** Strictly defaults to **`NOT_ASSESSED`** under production policy.
- **Abstention Justification:** Explains why 2D monocular vision cannot certify structural safety ratings or repair deadlines.
- **Interactive Prerequisites Checklist:** Tracks 6 mandatory prerequisites:
  1. Verified structural component identity (BIM / inspection plan).
  2. Element load path criticality (AASHTO / NBI classification).
  3. Calibrated physical metric defect dimensions (mm).
  4. Internal defect depth & delamination plane (NDT sounding, GPR, UPV).
  5. Formal validation and sign-off by a licensed Professional Engineer (PE/SE).
  6. Formally adopted Department of Transportation (DOT) maintenance decision policy.
- **Research Simulation Mode Toggle:** Allows researchers to test parameterized what-if simulations with prominent disclaimers (`RESEARCH_DEMONSTRATION_SIMULATION_ONLY`).

---

### 3.7 Data & Report Export

Provides one-click downloads:
- **`*_full_inspection_data.json`:** Unified JSON payload containing Stage 2 measurements, Stage 3 visual severity, and Stage 4 review queues.
- **`*_inspection_report.md`:** Publication-quality Markdown report with executive summary tables, review queue triage rankings, and mandatory legal disclaimers.

---

## 4. Verification & Testing

The Stage 5 dashboard was validated through `tests/test_dashboard.py` alongside the full project regression suite:

```bash
$ python3.11 -m unittest discover -s tests -v
----------------------------------------------------------------------
Ran 74 tests in 9.726s

OK
```

### Verified Test Cases:
1. `test_demo_cases_discovery_and_keys`: All 4 demonstration directories exist and load cleanly.
2. `test_train_0722_19_findings_contract`: Proves that `dacl10k_v2_train_0722` contains exactly 19 findings (7 structural, 12 non-structural).
3. `test_validation_0000_saved_outputs`: Proves that `dacl10k_v2_validation_0000` loads without invented data and maintains `NOT_ASSESSED`.
4. `test_end_to_end_live_pipeline_synthetic`: Proves live pipeline processes images and returns all required dictionaries.
5. `test_metric_calibration_guardrail`: Proves no metric millimeter claims exist in output dictionaries.
6. `test_research_demonstration_mode_isolation`: Proves simulation policy is isolated with mandatory disclaimers.
7. **Full Project Regression:** All 68 prior tests (checkpoint resume, loss functions, measurement geometry, explainable severity, prioritization) remain 100% green.

---

## 5. Launch Instructions

### 5.1 Local Launch (Mac / Linux)

```bash
# 1. Navigate to repository root
cd "/Users/kundankumar/Documents/untitled folder 2/bridge-ai-project"

# 2. Install dashboard dependencies (if not already installed)
python3.11 -m pip install -r requirements-dashboard.txt

# 3. Launch via shell script (default port 8501)
./scripts/launch_dashboard.sh

# Or launch directly with custom port:
python3.11 -m streamlit run app.py --server.port 8501
```

Access the dashboard in your web browser at: `http://localhost:8501`

---

### 5.2 Google Colab / Kaggle Launch

In a Google Colab notebook cell:

```python
# Cell 1: Pull latest repository code
!git pull origin main

# Cell 2: Install dependencies
!pip install -r requirements-dashboard.txt
!npm install -g localtunnel

# Cell 3: Launch Streamlit with localtunnel tunnel
import urllib
print("Your public IP for localtunnel password is:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())

!streamlit run app.py --server.headless true --server.port 8501 & npx localtunnel --port 8501
```

Click the generated `*.loca.lt` link and enter the printed external IP to open the live dashboard.
