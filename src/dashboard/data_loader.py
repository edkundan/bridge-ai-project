"""
Module: src.dashboard.data_loader
Description:
    Data loading and pipeline orchestration helpers for the Stage 5 Streamlit dashboard.
    Supports both Saved-Results Demonstration Mode (zero-GPU, instant) and
    End-to-End Live Inference Mode (with DeepLabV3+ segmentation model).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Repository root discovery
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from src.data.dataset import DACL10K_CLASSES
from src.measurement.pipeline import DefectMeasurementPipeline
from src.measurement.schema import ImageMeasurementResult
from src.measurement.severity import ExplainableVisualSeverityEvaluator
from src.measurement.visualization import DefectVisualizer
from src.prioritization.engine import MaintenancePrioritizationPipeline
from src.prioritization.policies import ProductionAbstentionPolicy, ResearchDemonstrationPolicy
from src.prioritization.schema import (
    Stage4PrioritizationResult,
    VerifiedEngineeringContext,
)

logger = logging.getLogger(__name__)

AVAILABLE_DEMO_CASES = {
    "dacl10k_v2_train_0722": {
        "title": "DACL10K Train 0722 (19 Findings: 7 Structural, 12 Non-Structural)",
        "description": "Real bridge inspection image exhibiting 19 distinct defect findings across structural concrete degradation and non-structural markings.",
        "dir": REPO_ROOT / "demo_data" / "dacl10k_v2_train_0722",
        "has_real_image": True,
        "image_file": "dacl10k_v2_train_0722.jpg",
        "findings_count": 19,
        "structural_count": 7,
        "non_structural_count": 12,
    },
    "dacl10k_v2_validation_0000": {
        "title": "DACL10K Validation 0000 (Benchmark Case)",
        "description": "Standard DACL10K-v2 validation set image containing non-structural surface marking on bridge abutment.",
        "dir": REPO_ROOT / "demo_data" / "dacl10k_v2_validation_0000",
        "has_real_image": True,
        "image_file": "dacl10k_v2_validation_0000.jpg",
        "findings_count": 1,
        "structural_count": 0,
        "non_structural_count": 1,
    },
    "graffiti_spalling_case": {
        "title": "Graffiti vs. Spalling Abutment (12% Cosmetic vs 0.3% Structural)",
        "description": "Real inspection case demonstrating non-structural isolation: extensive graffiti paint (12.06%) does not outrank localized concrete spalling (0.30%).",
        "dir": REPO_ROOT / "demo_data" / "graffiti_spalling_case",
        "has_real_image": False,
        "overlay_file": "graffiti_spalling_case_overlay.png",
        "findings_count": 2,
        "structural_count": 1,
        "non_structural_count": 1,
    },
    "synthetic_demo_image": {
        "title": "Multi-Defect Synthetic Benchmark (Crack, Spall, Rebar)",
        "description": "Controlled multi-defect verification scan with known crack geometry, spalling cavity, and overlapping exposed rebar.",
        "dir": REPO_ROOT / "demo_data" / "synthetic_demo_image",
        "has_real_image": True,
        "image_file": "synthetic_demo_image.jpg",
        "findings_count": 3,
        "structural_count": 3,
        "non_structural_count": 0,
    },
    "uploaded_test_cases": {
        "title": "Uploaded Inspection Case (images.jpeg: 32 Findings, 10 Classes)",
        "description": "Real bridge inspection image (images.jpeg) evaluated with Experiment 02 checkpoint (tau=0.50): 32 findings (24 structural, 8 non-structural) across 10 classes.",
        "dir": REPO_ROOT / "demo_data" / "uploaded_test_cases",
        "has_real_image": True,
        "image_file": "images.jpeg",
        "findings_count": 32,
        "structural_count": 24,
        "non_structural_count": 8,
    },
}


def load_saved_demo_case(case_id: str) -> Dict[str, Any]:
    """
    Loads machine-persisted demonstration outputs from disk without running neural net inference.

    Args:
        case_id: Key in AVAILABLE_DEMO_CASES.

    Returns:
        Dictionary containing image arrays, overlay, and structured JSON results.
    """
    if case_id not in AVAILABLE_DEMO_CASES:
        raise KeyError(f"Unknown demo case ID: {case_id}")

    meta = AVAILABLE_DEMO_CASES[case_id]
    case_dir = Path(meta["dir"])

    out: Dict[str, Any] = {
        "case_id": case_id,
        "meta": meta,
        "image_rgb": None,
        "overlay_rgb": None,
        "probability_maps": None,
        "measurement_dict": None,
        "severity_dict": None,
        "prioritization_dict": None,
        "review_queue_dict": None,
        "maintenance_priority_dict": None,
    }

    # Load image
    if meta.get("image_file"):
        img_p = case_dir / meta["image_file"]
        if img_p.exists():
            bgr = cv2.imread(str(img_p))
            if bgr is not None:
                out["image_rgb"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Load overlay
    overlay_candidates = [
        case_dir / f"{case_id}_overlay.png",
        case_dir / meta.get("overlay_file", ""),
    ]
    for oc in overlay_candidates:
        if str(oc) and oc.exists():
            bgr = cv2.imread(str(oc))
            if bgr is not None:
                out["overlay_rgb"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if out["image_rgb"] is None:
                    out["image_rgb"] = out["overlay_rgb"].copy()
                break

    # Load probability maps if present
    npz_candidates = list(case_dir.glob("*.npz"))
    if npz_candidates:
        try:
            data = np.load(str(npz_candidates[0]))
            if "probability_maps" in data:
                out["probability_maps"] = data["probability_maps"]
        except Exception as e:
            logger.warning(f"Could not load npz probability map: {e}")

    # Load JSON artifacts
    for json_file in case_dir.glob("*.json"):
        fname = json_file.name.lower()
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)

            if "inspection_review_queue" in fname:
                out["review_queue_dict"] = content
            elif "engineering_maintenance_priority" in fname:
                out["maintenance_priority_dict"] = content
            elif "stage4_prioritization" in fname:
                out["prioritization_dict"] = content
            elif "measurement" in fname:
                out["measurement_dict"] = content
            elif "severity" in fname:
                out["severity_dict"] = content
        except Exception as e:
            logger.warning(f"Failed parsing {json_file}: {e}")

    # Fallback extraction from unified Stage 4 report if separate files missing
    if out["prioritization_dict"]:
        p_dict = out["prioritization_dict"]
        if not out["review_queue_dict"] and "output_a_inspection_review_queue" in p_dict:
            out["review_queue_dict"] = p_dict["output_a_inspection_review_queue"]
        if not out["maintenance_priority_dict"] and "output_b_engineering_maintenance_priority" in p_dict:
            out["maintenance_priority_dict"] = p_dict["output_b_engineering_maintenance_priority"]

    # Normalize measurement dictionary top-level keys for backward compatibility
    if out["measurement_dict"]:
        m_dict = out["measurement_dict"]
        summary = m_dict.get("summary") or {}
        cov = summary.get("coverage_metrics") or {}
        for k in ["total_components_count", "total_detected_classes", "summed_class_pixel_area", "summed_class_image_area_pct"]:
            if k not in m_dict and k in summary:
                m_dict[k] = summary[k]
        for k in [
            "structural_union_image_area_pct",
            "structural_union_pixel_area",
            "non_structural_union_image_area_pct",
            "non_structural_union_pixel_area",
            "union_defect_image_area_pct",
            "union_defect_pixel_area",
            "overlapping_defect_image_area_pct",
            "overlapping_defect_pixel_area",
        ]:
            if k not in m_dict and k in cov:
                m_dict[k] = cov[k]

    # Normalize maintenance priority dictionary for backward compatibility
    if out["maintenance_priority_dict"]:
        m_dict = out["maintenance_priority_dict"]
        p_dec = m_dict.get("priority_decision") or {}
        for k in ["status", "official_repair_priority", "official_repair_deadline", "official_structural_safety_rating", "official_aashto_condition_state", "abstention_reason"]:
            if k not in m_dict and k in p_dec:
                m_dict[k] = p_dec[k]
        gov = m_dict.get("governance_and_compliance") or {}
        if "missing_engineering_prerequisites" not in m_dict and "missing_engineering_prerequisites" in gov:
            m_dict["missing_engineering_prerequisites"] = gov["missing_engineering_prerequisites"]
        if "contradiction_warnings" not in m_dict and "contradiction_warnings" in gov:
            m_dict["contradiction_warnings"] = gov["contradiction_warnings"]

    return out


def run_live_pipeline(
    image_rgb: np.ndarray,
    image_id: str = "custom_inspection_image",
    checkpoint_path: Optional[str] = None,
    probability_maps: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_component_area_pixels: int = 10,
    roi_mask: Optional[np.ndarray] = None,
    engineering_context: Optional[VerifiedEngineeringContext] = None,
    use_research_demo_policy: bool = False,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Executes end-to-end Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 pipeline.

    Args:
        image_rgb: Input image array (H, W, 3) in RGB format.
        image_id: Identifier string for image.
        checkpoint_path: Optional path to DeepLabV3+ checkpoint (.pth).
        probability_maps: Optional precomputed 19-channel probability tensor (19, H, W).
        threshold: Binary threshold in [0.0, 1.0].
        min_component_area_pixels: Minimum connected component area.
        roi_mask: Optional binary structural element ROI mask (H, W).
        engineering_context: Optional VerifiedEngineeringContext instance.
        use_research_demo_policy: If True, uses ResearchDemonstrationPolicy.
        device: 'auto', 'mps', 'cuda', or 'cpu'.

    Returns:
        Dictionary containing all structured measurements, queue rankings, and overlays.
    """
    h, w = image_rgb.shape[:2]

    # 1. Obtain 19-channel probability maps
    if probability_maps is not None:
        probs = probability_maps
    elif checkpoint_path and os.path.exists(checkpoint_path):
        try:
            from src.segmentation.predict import BridgeDefectPredictor
            predictor = BridgeDefectPredictor(
                checkpoint_path=checkpoint_path,
                device=device,
                threshold=threshold,
            )
            pred_dict = predictor.predict_image(image_rgb, return_orig_size=True)
            probs = pred_dict["probability_maps"]
        except Exception as e:
            logger.error(f"Inference failed with checkpoint: {e}. Falling back to zero map.")
            probs = np.zeros((19, h, w), dtype=np.float32)
    else:
        # Clean scan fallback when no checkpoint is supplied
        probs = np.zeros((19, h, w), dtype=np.float32)

    # 2. Stage 2: Geometric Measurement Pipeline
    meas_pipeline = DefectMeasurementPipeline(
        threshold=threshold,
        min_component_area_pixels=min_component_area_pixels,
    )
    measurement_result: ImageMeasurementResult = meas_pipeline.measure_probabilities(
        probabilities=probs,
        image_id=image_id,
        roi_mask=roi_mask,
    )

    # 3. Stage 3: Explainable Visual Severity Assessment
    sev_evaluator = ExplainableVisualSeverityEvaluator()
    stage3_report = sev_evaluator.evaluate(measurement_result)

    # 4. Stage 4: Maintenance Prioritization Pipeline
    p_pipeline = MaintenancePrioritizationPipeline(use_research_demo=use_research_demo_policy)
    stage4_result: Stage4PrioritizationResult = p_pipeline.run(
        stage3_report.to_dict(),
        engineering_context=engineering_context,
    )

    # 5. Visualization Overlay (Default clean view: masks blended, labels and bboxes hidden)
    visualizer = DefectVisualizer()
    overlay_rgb = visualizer.render_overlay(
        image_rgb=image_rgb,
        measurement=measurement_result,
        probability_maps=probs,
        roi_mask=roi_mask,
        alpha=0.45,
        show_masks=True,
        show_bboxes=False,
        show_labels=False,
    )

    return {
        "image_id": image_id,
        "image_rgb": image_rgb,
        "overlay_rgb": overlay_rgb,
        "probability_maps": probs,
        "measurement_result": measurement_result,
        "measurement_dict": measurement_result.to_dict(),
        "stage3_report": stage3_report,
        "severity_dict": stage3_report.to_dict(),
        "stage4_result": stage4_result,
        "prioritization_dict": stage4_result.to_dict(),
        "review_queue_dict": stage4_result.inspection_review_queue.to_dict(),
        "maintenance_priority_dict": stage4_result.engineering_maintenance_priority.to_dict(),
    }


# Backward-compatible alias
run_live_inference = run_live_pipeline
