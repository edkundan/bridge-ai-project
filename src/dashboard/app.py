"""
Application: src.dashboard.app
Description:
    Stage 5 Streamlit Interactive Explainable AI Bridge Inspection Dashboard.
    Provides visual triage, defect quantification, rule-based explainability,
    inspection review queue management, and statutory maintenance abstention.

Usage:
    streamlit run src/dashboard/app.py
    # or via root launcher:
    streamlit run app.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import streamlit as st

from src.dashboard.components import (
    render_explainability_panel,
    render_export_panel,
    render_header_and_disclaimers,
    render_image_and_overlay_viewer,
    render_maintenance_assessment_panel,
    render_measurements_panel,
    render_inspection_review_queue,
)
from src.dashboard.data_loader import (
    AVAILABLE_DEMO_CASES,
    load_saved_demo_case,
    run_live_pipeline,
)
from src.prioritization.schema import (
    ElementCriticality,
    EnvironmentalExposure,
    DefectProgressionState,
    VerifiedEngineeringContext,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dashboard_app")

# Page configuration
st.set_page_config(
    page_title="Explainable AI Bridge Inspection Dashboard",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def build_engineering_context_from_sidebar() -> Tuple[Optional[VerifiedEngineeringContext], bool]:
    """Builds VerifiedEngineeringContext from sidebar form inputs."""
    with st.sidebar.expander("🛠️ Verified Physical Engineering Context (Optional)", expanded=False):
        st.caption("Provide physical engineering parameters verified by a licensed Professional Engineer (PE).")
        has_context = st.checkbox("Supply Verified Context Data", value=False)
        if not has_context:
            use_demo_policy = st.checkbox(
                "🧪 Enable Research Demonstration Simulation Mode",
                value=False,
                help="For academic illustration only. Not valid for actual bridge maintenance.",
            )
            return None, use_demo_policy

        bridge_id = st.text_input("Bridge Asset ID", value="BRIDGE-US-I80-0442")
        structural_comp = st.text_input("Structural Component", value="Main Prestressed Girder G3, Midspan")
        crit_options = [e.name for e in ElementCriticality]
        crit_choice = st.selectbox("Element Criticality", options=crit_options, index=crit_options.index("PRIMARY_LOAD_BEARING"))

        col_w, col_l, col_d = st.columns(3)
        with col_w:
            w_mm = st.number_input("Width (mm)", min_value=0.0, value=210.0, step=5.0)
        with col_l:
            l_mm = st.number_input("Length (mm)", min_value=0.0, value=480.0, step=5.0)
        with col_d:
            depth_mm = st.number_input("Depth (mm)", min_value=0.0, value=45.0, step=1.0)

        prog_options = [e.name for e in DefectProgressionState]
        prog_choice = st.selectbox("Progression State", options=prog_options, index=prog_options.index("ACTIVE_PROPAGATION"))

        exp_options = [e.name for e in EnvironmentalExposure]
        exp_choice = st.selectbox("Environmental Exposure", options=exp_options, index=exp_options.index("SEVERE_DEICING_SALTS"))

        pe_verified = st.checkbox("Formally Validated by Licensed PE/SE", value=True)
        pe_id = st.text_input("PE License ID", value="PE-CA-C88392")

        use_demo_policy = st.checkbox(
            "🧪 Enable Research Demonstration Simulation Mode",
            value=False,
            help="For academic illustration only. Not valid for actual bridge maintenance.",
        )

        ctx = VerifiedEngineeringContext(
            bridge_id=bridge_id,
            structural_component=structural_comp,
            element_criticality=ElementCriticality[crit_choice],
            calibrated_dimensions={"width_mm": w_mm, "length_mm": l_mm, "depth_mm": depth_mm},
            defect_depth_mm=depth_mm if depth_mm > 0 else None,
            progression_rate=DefectProgressionState[prog_choice],
            exposure_conditions=EnvironmentalExposure[exp_choice],
            is_verified_by_pe=pe_verified,
            pe_license_id=pe_id if pe_verified else None,
        )
        return ctx, use_demo_policy


def main():
    # 1. Header & Disclaimers
    render_header_and_disclaimers()

    # 2. Sidebar Controls
    st.sidebar.title("🎛️ Inspection Controls")

    exec_mode = st.sidebar.radio(
        "Execution Mode:",
        options=[
            "Saved-Results Demonstration (No GPU)",
            "End-to-End Live Inference (DeepLabV3+)",
        ],
        index=0,
    )

    engineering_context, use_demo_policy = build_engineering_context_from_sidebar()

    # State containers
    data_bundle = None
    image_id = "inspection_image"

    if exec_mode == "Saved-Results Demonstration (No GPU)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Saved Demonstration Cases")
        case_keys = list(AVAILABLE_DEMO_CASES.keys())
        case_labels = [AVAILABLE_DEMO_CASES[k]["title"] for k in case_keys]
        selected_idx = st.sidebar.selectbox(
            "Select Inspection Demonstration:",
            options=range(len(case_keys)),
            format_func=lambda i: case_labels[i],
            index=0,
        )
        selected_case_id = case_keys[selected_idx]
        st.sidebar.caption(AVAILABLE_DEMO_CASES[selected_case_id]["description"])

        # Load case data
        try:
            data_bundle = load_saved_demo_case(selected_case_id)
            image_id = selected_case_id
        except Exception as e:
            st.error(f"Error loading demonstration case {selected_case_id}: {e}")
            return

    else:
        # End-to-End Live Inference Mode
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔬 Live Segmentation Settings")

        img_source = st.sidebar.selectbox(
            "Image Input Source:",
            options=[
                "Upload Custom Image",
                "Load Benchmark Image (Validation 0000)",
                "Load Benchmark Image (Train 0722)",
                "Load Benchmark Image (Uploaded Case: images.jpeg)",
            ],
            index=0,
        )

        input_image_rgb = None
        if img_source == "Upload Custom Image":
            uploaded = st.sidebar.file_uploader("Upload Bridge Inspection Image", type=["jpg", "jpeg", "png"])
            if uploaded:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if bgr is not None:
                    input_image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    image_id = Path(uploaded.name).stem
        elif img_source == "Load Benchmark Image (Validation 0000)":
            p = REPO_ROOT / "demo_data" / "dacl10k_v2_validation_0000" / "dacl10k_v2_validation_0000.jpg"
            if p.exists():
                bgr = cv2.imread(str(p))
                input_image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image_id = "dacl10k_v2_validation_0000"
        elif img_source == "Load Benchmark Image (Uploaded Case: images.jpeg)":
            p = REPO_ROOT / "demo_data" / "uploaded_test_cases" / "images.jpeg"
            if p.exists():
                bgr = cv2.imread(str(p))
                input_image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image_id = "images"
        else:
            p = REPO_ROOT / "demo_data" / "dacl10k_v2_train_0722" / "dacl10k_v2_train_0722.jpg"
            if p.exists():
                bgr = cv2.imread(str(p))
                input_image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image_id = "dacl10k_v2_train_0722"

        exp02_ckpt = "experiments/exp_02_sqrt_weighted_bcedice/checkpoints/best_model.pth"
        base_ckpt = "experiments/segmentation_baseline/checkpoints/best_model.pth"
        ckpt_default = exp02_ckpt if (REPO_ROOT / exp02_ckpt).exists() else base_ckpt
        ckpt_input = st.sidebar.text_input("Model Checkpoint Path:", value=ckpt_default)
        threshold = st.sidebar.slider("Detection Threshold (tau):", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
        min_area = st.sidebar.slider("Min Component Area (px):", min_value=5, max_value=300, value=10, step=5)
        device = st.sidebar.selectbox("Compute Device:", options=["auto", "mps", "cuda", "cpu"], index=0)

        run_btn = st.sidebar.button("🚀 Run Live Pipeline", type="primary", use_container_width=True)

        if input_image_rgb is not None and (run_btn or "live_bundle" in st.session_state):
            if run_btn:
                with st.spinner("Executing Stage 1-4 pipeline on image..."):
                    try:
                        live_res = run_live_pipeline(
                            image_rgb=input_image_rgb,
                            image_id=image_id,
                            checkpoint_path=str(REPO_ROOT / ckpt_input) if ckpt_input else None,
                            threshold=threshold,
                            min_component_area_pixels=min_area,
                            engineering_context=engineering_context,
                            use_research_demo_policy=use_demo_policy,
                            device=device,
                        )
                        st.session_state["live_bundle"] = live_res
                    except Exception as e:
                        st.error(f"Live pipeline execution failed: {e}")
                        logger.exception("Live pipeline failure")
                        return

            data_bundle = st.session_state.get("live_bundle")
        elif input_image_rgb is None:
            st.info("👈 Please upload an image or choose a benchmark image from the sidebar, then click 'Run Live Pipeline'.")
            return

    if not data_bundle:
        st.warning("No inspection data loaded.")
        return

    # Extract components
    image_rgb = data_bundle.get("image_rgb")
    overlay_rgb = data_bundle.get("overlay_rgb")
    measurement_result = data_bundle.get("measurement_result")
    measurement_dict = data_bundle.get("measurement_dict")
    measurement = measurement_result if measurement_result is not None else measurement_dict
    severity_dict = data_bundle.get("severity_dict")
    prioritization_dict = data_bundle.get("prioritization_dict")
    review_queue_dict = data_bundle.get("review_queue_dict")
    maintenance_priority_dict = data_bundle.get("maintenance_priority_dict")

    # 3. Main Dashboard Sections
    # A. Inspection Review Queue (Rendered first so user can select finding for callout)
    selected_finding_id = render_inspection_review_queue(review_queue_dict)

    st.markdown("---")

    # B. Image and Defect Overlays (with finding callout)
    render_image_and_overlay_viewer(
        image_rgb=image_rgb,
        overlay_rgb=overlay_rgb,
        measurement=measurement,
        measurement_dict=measurement_dict,
        selected_finding_id=selected_finding_id,
        probability_maps=data_bundle.get("probability_maps"),
    )

    st.markdown("---")

    # C. Quantitative Defect Measurements
    render_measurements_panel(measurement=measurement, measurement_dict=measurement_dict)

    st.markdown("---")

    # D. Explainability & Field Guidance
    render_explainability_panel(selected_finding_id, review_queue_dict)

    st.markdown("---")

    # E. Statutory Maintenance Priority Assessment
    render_maintenance_assessment_panel(maintenance_priority_dict, engineering_context=engineering_context)

    st.markdown("---")

    # F. Export Artifacts
    render_export_panel(
        measurement=measurement,
        measurement_dict=measurement_dict,
        severity_dict=severity_dict,
        prioritization_dict=prioritization_dict,
        review_queue_dict=review_queue_dict,
        maintenance_priority_dict=maintenance_priority_dict,
        image_id=image_id,
    )


if __name__ == "__main__":
    main()
