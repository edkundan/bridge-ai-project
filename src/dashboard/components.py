"""
Module: src.dashboard.components
Description:
    Modular UI presentation components for the Stage 5 Streamlit Dashboard.
    Enforces publication-grade engineering terminology, calibration disclaimers,
    strict abstention disclosures, and clear rule-based explainability.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.data.dataset import DACL10K_CLASSES
from src.measurement.schema import CALIBRATION_DISCLAIMER, NON_STRUCTURAL_CLASSES
from src.prioritization.schema import (
    RESEARCH_DEMO_POLICY_DISCLAIMER,
    ElementCriticality,
    EnvironmentalExposure,
    DefectProgressionState,
    ReviewQueueUrgency,
    VerifiedEngineeringContext,
)


def safe_render_dataframe(
    df: pd.DataFrame,
    use_container_width: bool = True,
    hide_index: bool = True,
) -> None:
    """
    Safely render a pandas DataFrame in Streamlit.
    Attempts st.dataframe (with PyArrow engine), and seamlessly falls back
    to a pure HTML table if PyArrow or binary serialization encounters an environment issue.
    """
    try:
        try:
            st.dataframe(df, width="stretch", hide_index=hide_index)
        except (TypeError, ValueError):
            st.dataframe(df, use_container_width=True, hide_index=hide_index)
    except Exception as exc:
        st.caption(f"ℹ️ Rendering standard table view (backend fallback: {type(exc).__name__})")
        # Pure HTML fallback does not call st.table() and has ZERO dependency on pyarrow
        html_table = df.to_html(index=not hide_index, classes="dataframe", border=0)
        styled_html = f"""
        <div style="overflow-x: auto; max-height: 480px; border: 1px solid #334155; border-radius: 6px; padding: 8px; margin-bottom: 1rem;">
            {html_table}
        </div>
        """
        st.markdown(styled_html, unsafe_allow_html=True)


def render_header_and_disclaimers():
    """Renders dashboard header, scientific badges, and prominent guardrail alerts."""
    st.markdown(
        """
        <div style="padding: 1.2rem; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    border-radius: 10px; border: 1px solid #334155; margin-bottom: 1.5rem;">
            <h1 style="color: #f8fafc; margin: 0; font-size: 1.85rem; font-weight: 700;">
                🌉 Explainable AI Bridge Inspection & Maintenance Prioritization
            </h1>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.0rem;">
                Stage 5 Interactive Research Dashboard | Multi-Label Semantic Vision (DACL10K-v2) & Engineering Abstention Foundation
            </p>
            <div style="margin-top: 0.8rem; display: flex; gap: 0.6rem; flex-wrap: wrap;">
                <span style="background: #0284c7; color: white; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">
                    🔬 Research Prototype (v0.5.0)
                </span>
                <span style="background: #334155; color: #e2e8f0; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">
                    🏛️ AASHTO MBEI / FHWA BIRM Decoupled Architecture
                </span>
                <span style="background: #334155; color: #e2e8f0; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">
                    🛡️ Strict Production Abstention (NOT_ASSESSED)
                </span>
                <span style="background: #334155; color: #e2e8f0; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">
                    🎯 19 DACL10K Defect & Component Classes
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚠️ Critical Terminology & Engineering Guardrails (Click to Read)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                r"""
                **1. Pixel-Domain Measurement Boundary:**  
                All lengths, widths, perimeters, and areas reported herein are in **pixels (`px`)** or **image-relative percentages (`%`)**.  
                *Never label pixel measurements as metric millimeters without physical camera calibration, rangefinder depth, or optical fiducial markers.*
                
                **2. Uncalibrated Model Confidence:**  
                Segmentation confidence reflects the neural network's per-pixel activation probability $P \in [0, 1]$.  
                *High optical confidence ($P = 0.95$) signifies clear visual contrast, NOT a high structural failure probability or immediate risk.*
                """
            )
        with col2:
            st.markdown(
                r"""
                **3. Provisional Review Tiers vs. Statutory Orders:**  
                Labels such as `URGENT_INSPECTION_REVIEW` are **prototype triage flags** to prioritize human field inspector attention.  
                *They are NOT certified emergency orders, speed restrictions, or bridge closure declarations.*

                **4. Statutory Maintenance Abstention:**  
                Official bridge condition states and maintenance work orders strictly require **load path hierarchy, defect depth, and licensed PE sign-off**.  
                *The system explicitly abstains (`NOT_ASSESSED`) when physical engineering context is missing.*
                """
            )


def render_image_and_overlay_viewer(
    image_rgb: Optional[np.ndarray],
    overlay_rgb: Optional[np.ndarray],
    measurement_dict: Optional[Dict[str, Any]],
    selected_finding_id: Optional[str] = None,
):
    """
    Renders high-resolution visual comparison tabs: Composite Overlay, Original Image, and Finding Callout.
    """
    st.subheader("🖼️ Visual Inspection & Defect Overlays")

    if image_rgb is None and overlay_rgb is None:
        st.info("No image available for visual display.")
        return

    tab1, tab2, tab3 = st.tabs(["🎨 Multi-Class Defect Overlay", "📷 Original Bridge Image", "🔍 Selected Finding Callout"])

    with tab1:
        if overlay_rgb is not None:
            st.image(
                overlay_rgb,
                caption="Overlay: Multi-label defect contours with component bounding boxes and class color codes",
                use_container_width=True,
            )
        elif image_rgb is not None:
            st.image(image_rgb, caption="Original image (overlay pending)", use_container_width=True)

    with tab2:
        if image_rgb is not None:
            st.image(image_rgb, caption="Raw monocular inspection image", use_container_width=True)
        else:
            st.warning("Original image RGB array unavailable; displaying overlay.")
            if overlay_rgb is not None:
                st.image(overlay_rgb, use_container_width=True)

    with tab3:
        if not selected_finding_id:
            st.info("👈 Select a finding from the Inspection Review Queue below to view its localized crop and spatial callout.")
        elif image_rgb is not None and measurement_dict:
            # Locate component matching selected_finding_id
            target_comp = None
            target_class = None
            class_measurements = measurement_dict.get("class_measurements", {})
            for cname, cdata in class_measurements.items():
                for comp in cdata.get("components", []):
                    fid = comp.get("finding_id", f"FINDING-{cname[:5].upper()}-{comp.get('component_id', 1):03d}")
                    if fid == selected_finding_id or comp.get("finding_id") == selected_finding_id:
                        target_comp = comp
                        target_class = cname
                        break
                if target_comp:
                    break

            if target_comp and "bbox" in target_comp:
                bbox = target_comp["bbox"]
                h_img, w_img = image_rgb.shape[:2]
                ymin = max(0, bbox["ymin"] - 25)
                ymax = min(h_img, bbox["ymax"] + 25)
                xmin = max(0, bbox["xmin"] - 25)
                xmax = min(w_img, bbox["xmax"] + 25)

                crop = image_rgb[ymin:ymax, xmin:xmax].copy()
                # Draw neon highlight box
                box_x1 = max(0, bbox["xmin"] - xmin)
                box_y1 = max(0, bbox["ymin"] - ymin)
                box_x2 = min(crop.shape[1] - 1, bbox["xmax"] - xmin)
                box_y2 = min(crop.shape[0] - 1, bbox["ymax"] - ymin)

                is_struct = target_class not in NON_STRUCTURAL_CLASSES
                box_color = (255, 50, 50) if is_struct else (255, 200, 0)
                cv2.rectangle(crop, (box_x1, box_y1), (box_x2, box_y2), box_color, 3)

                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.image(
                        crop,
                        caption=f"High-Resolution Callout: {selected_finding_id} ({target_class})",
                        use_container_width=True,
                    )
                with col_b:
                    st.markdown(f"### Finding Details: `{selected_finding_id}`")
                    st.markdown(f"**Class:** `{target_class}` ({'Structural Defect' if is_struct else 'Non-Structural Marking'})")
                    st.markdown(f"**Pixel Area:** `{target_comp.get('pixel_area', 0):,}` px ({target_comp.get('image_relative_area_pct', 0.0):.4f}% of image)")
                    st.markdown(f"**Bounding Box:** `[X: {bbox['xmin']}→{bbox['xmax']}, Y: {bbox['ymin']}→{bbox['ymax']}]` ({bbox['width']} × {bbox['height']} px)")
                    conf = target_comp.get("confidence", {})
                    st.markdown(f"**Mean Optical Confidence:** `{conf.get('mean', 0.0):.4f}`")
                    shape = target_comp.get("shape_features", {})
                    if shape:
                        st.markdown(f"**Circularity:** `{shape.get('circularity', 0.0):.3f}` | **Solidity:** `{shape.get('solidity', 0.0):.3f}` | **Aspect Ratio:** `{shape.get('aspect_ratio', 1.0):.2f}`")
                        if shape.get("skeleton_length_pixels"):
                            st.markdown(f"**Centerline Skeleton Length:** `{shape.get('skeleton_length_pixels')} px` | **Avg Width:** `{shape.get('estimated_avg_width_pixels', 0.0):.2f} px`")
            else:
                st.warning(f"Could not find bounding box coordinates for {selected_finding_id}.")


def render_measurements_panel(measurement_dict: Optional[Dict[str, Any]]):
    """
    Renders the quantitative defect measurement section.
    Distinguishes individual class areas from union area, and structural from non-structural damage.
    """
    st.subheader("📐 Quantitative Defect Measurements (Stage 2)")

    if not measurement_dict:
        st.info("No measurement data available.")
        return

    # Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Total Findings",
            value=f"{measurement_dict.get('total_components_count', 0)}",
            help="Total connected components across all detected classes.",
        )
    with col2:
        st.metric(
            label="Detected Classes",
            value=f"{measurement_dict.get('total_detected_classes', 0)} / 19",
            help="Number of active semantic classes with foreground pixels.",
        )
    with col3:
        struct_pct = measurement_dict.get("structural_union_image_area_pct", 0.0)
        struct_px = measurement_dict.get("structural_union_pixel_area", 0)
        st.metric(
            label="Structural Union",
            value=f"{struct_pct:.3f}%",
            delta=f"{struct_px:,} px",
            delta_color="inverse",
            help="Spatial union of all structural defect classes (excludes cosmetic markings).",
        )
    with col4:
        non_struct_pct = measurement_dict.get("non_structural_union_image_area_pct", 0.0)
        non_struct_px = measurement_dict.get("non_structural_union_pixel_area", 0)
        st.metric(
            label="Non-Structural Union",
            value=f"{non_struct_pct:.3f}%",
            delta=f"{non_struct_px:,} px",
            delta_color="off",
            help="Spatial union of aesthetic markings (e.g., Graffiti, Restformwork).",
        )
    with col5:
        total_union_pct = measurement_dict.get("union_defect_image_area_pct", 0.0)
        total_union_px = measurement_dict.get("union_defect_pixel_area", 0)
        st.metric(
            label="Total Defect Union",
            value=f"{total_union_pct:.3f}%",
            delta=f"{total_union_px:,} px",
            delta_color="off",
            help="Overall spatial union footprint across all 19 classes.",
        )

    # Class-Level Table
    st.markdown("#### Detected Class Summary Table")
    class_measurements = measurement_dict.get("class_measurements", {})
    rows = []
    for cname, cdata in class_measurements.items():
        if cdata.get("detected", False):
            is_struct = cname not in NON_STRUCTURAL_CLASSES
            conf = cdata.get("confidence") or {}
            rows.append({
                "Class Name": cname,
                "Domain Category": "Structural Defect" if is_struct else "Non-Structural Marking",
                "Component Count": cdata.get("component_count", 0),
                "Total Pixels": f"{cdata.get('total_pixel_area', 0):,}",
                "Image Coverage (%)": f"{cdata.get('image_relative_area_pct', 0.0):.4f}%",
                "Mean Confidence": f"{conf.get('mean', 0.0):.4f}",
                "Min Confidence": f"{conf.get('min', 0.0):.4f}",
                "Max Confidence": f"{conf.get('max', 0.0):.4f}",
            })

    if rows:
        df_classes = pd.DataFrame(rows)
        safe_render_dataframe(df_classes, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Clean scan: No defect components detected above probability threshold.")

    # Morphological and Crack Geometry Table
    comp_rows = []
    for cname, cdata in class_measurements.items():
        for comp in cdata.get("components", []):
            shape = comp.get("shape_features") or {}
            conf = comp.get("confidence") or {}
            fid = comp.get("finding_id", f"FINDING-{cname[:5].upper()}-{comp.get('component_id', 1):03d}")
            comp_rows.append({
                "Finding ID": fid,
                "Class": cname,
                "Pixel Area": comp.get("pixel_area", 0),
                "Image Area (%)": comp.get("image_relative_area_pct", 0.0),
                "Aspect Ratio": shape.get("aspect_ratio", 1.0),
                "Circularity": shape.get("circularity", 0.0),
                "Solidity": shape.get("solidity", 0.0),
                "Skeleton Length (px)": shape.get("skeleton_length_pixels"),
                "Mean Width (px)": shape.get("estimated_avg_width_pixels"),
                "Mean Conf": conf.get("mean", 0.0),
            })

    if comp_rows:
        with st.expander("🔬 Component Morphological Features & Centerline Geometry", expanded=False):
            st.caption("All geometric metrics are strictly in pixel units (px). Metric width in millimeters requires metric optical calibration.")
            df_comp = pd.DataFrame(comp_rows)
            safe_render_dataframe(df_comp, use_container_width=True, hide_index=True)


def render_inspection_review_queue(
    review_queue_dict: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Renders Output A: Inspection Review Queue table with urgency badges and filters.
    Returns selected finding_id or None.
    """
    st.subheader("📋 Output A: Inspection Review Queue (Provisional Field Triage)")
    st.caption("Provisional ranking of visual findings to guide human inspector field verification. Does not represent statutory work orders.")

    if not review_queue_dict:
        st.info("No review queue available.")
        return None

    queue_summary = review_queue_dict.get("queue_summary", {})
    ranked_items = review_queue_dict.get("ranked_queue_items", [])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total in Queue", queue_summary.get("total_findings_count", len(ranked_items)))
    with c2:
        st.metric("Structural Findings", queue_summary.get("structural_findings_count", 0))
    with c3:
        st.metric("Non-Structural Findings", queue_summary.get("non_structural_findings_count", 0))
    with c4:
        highest_urg = queue_summary.get("highest_queue_urgency", "NO_ACTION_REQUIRED")
        st.metric("Highest Urgency", highest_urg.replace("_", " "))

    # Filters
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        category_filter = st.radio(
            "Filter Observations:",
            options=["All Findings", "Structural Only", "Non-Structural Only"],
            horizontal=True,
        )
    with col_f2:
        urgency_filter = st.selectbox(
            "Filter by Urgency Level:",
            options=["All Urgency Levels", "URGENT_INSPECTION_REVIEW", "PRIORITY_INSPECTION_REVIEW", "ROUTINE_INSPECTION_REVIEW", "INFORMATIONAL_NON_STRUCTURAL"],
        )

    # Filter ranked items
    filtered_items = []
    for it in ranked_items:
        if category_filter == "Structural Only" and not it.get("is_structural", True):
            continue
        if category_filter == "Non-Structural Only" and it.get("is_structural", True):
            continue
        if urgency_filter != "All Urgency Levels" and it.get("queue_urgency") != urgency_filter:
            continue
        filtered_items.append(it)

    if not filtered_items:
        st.info("No findings match the selected filter criteria.")
        return None

    # Table display
    table_rows = []
    for it in filtered_items:
        urgency = it.get("queue_urgency", "ROUTINE")
        badge = "🔴 URGENT" if "URGENT" in urgency else ("🟠 PRIORITY" if "PRIORITY" in urgency else ("🔵 ROUTINE" if "ROUTINE" in urgency else "⚪ NON-STRUCT"))
        table_rows.append({
            "Rank": f"#{it.get('rank', 1)}",
            "Finding ID": it.get("finding_id", "FINDING"),
            "Class Name": it.get("class_name", "Unknown"),
            "Domain": "Structural" if it.get("is_structural", True) else "Non-Structural",
            "Urgency Level": badge,
            "Verification Score": f"{it.get('composite_verification_score', 0.0):.4f}",
            "Coverage (%)": f"{it.get('image_relative_extent_pct', 0.0):.4f}%",
            "Optical Conf": f"{it.get('model_confidence', 0.0):.4f}",
            "Uncertainty": it.get("optical_uncertainty_level", "LOW"),
        })

    safe_render_dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Selector
    finding_options = [it.get("finding_id") for it in filtered_items]
    selected_fid = st.selectbox(
        "🔎 Select Finding ID to Inspect Rule-Based Explanations & Field Guidance:",
        options=finding_options,
        index=0 if finding_options else None,
    )

    return selected_fid


def render_explainability_panel(
    selected_finding_id: Optional[str],
    review_queue_dict: Optional[Dict[str, Any]],
):
    """
    Renders rule-based explainability, ranking factor decomposition, and inspector guidance.
    """
    st.subheader("💡 Rule-Based Decision Explanations & Inspector Guidance")
    st.caption("Explicit explanations of ranking factors, optical uncertainties, and field actions. These are deterministic rules, not model attributions.")

    if not selected_finding_id or not review_queue_dict:
        st.info("Select a finding from the review queue above to view its explainability breakdown.")
        return

    # Find item
    item = None
    for it in review_queue_dict.get("ranked_queue_items", []):
        if it.get("finding_id") == selected_finding_id:
            item = it
            break

    if not item:
        st.warning(f"Finding `{selected_finding_id}` not found in queue.")
        return

    st.markdown(f"### Finding: `{item.get('finding_id')}` — {item.get('class_name')}")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 1. Visual Evidence & Observational Summary")
        st.info(item.get("visual_evidence_summary", "No visual evidence recorded."))

        st.markdown("#### 2. Triggered Policy Rules")
        rules = item.get("triggered_policy_rules", [])
        if rules:
            for r in rules:
                st.markdown(f"- 📜 **`{r}`**")
        else:
            st.markdown("- *Standard baseline triage rule applied.*")

        st.markdown("#### 3. Optical Uncertainty Analysis")
        unc_level = item.get("optical_uncertainty_level", "LOW")
        if unc_level == "HIGH":
            st.error(f"⚠️ **{unc_level} OPTICAL UNCERTAINTY:** Model confidence or pixel area is near detection threshold. Direct tactile re-measurement required.")
        elif unc_level == "MODERATE":
            st.warning(f"ℹ️ **{unc_level} OPTICAL UNCERTAINTY:** Moderate confidence or non-structural category.")
        else:
            st.success(f"✅ **{unc_level} OPTICAL UNCERTAINTY:** High-contrast, well-defined defect region.")

        factors = item.get("uncertainty_factors", [])
        if factors:
            st.caption("Contributing factors: " + "; ".join(factors))

    with col2:
        st.markdown("#### 4. Recommended Field Inspector Actions")
        actions = item.get("recommended_inspector_action", [])
        if actions:
            for act in actions:
                st.markdown(f"- 🔨 {act}")
        else:
            st.markdown("- *Conduct routine visual inspection during next cycle.*")

        st.markdown("#### 5. Engineering Limitations & Missing Prerequisites")
        limits = item.get("decision_limitations", [])
        if limits:
            for lim in limits:
                st.markdown(f"- 🔒 *{lim}*")


def render_maintenance_assessment_panel(
    maintenance_priority_dict: Optional[Dict[str, Any]],
    engineering_context: Optional[VerifiedEngineeringContext] = None,
):
    """
    Renders Output B: Engineering Maintenance Priority with strict NOT_ASSESSED disclosure.
    """
    st.subheader("🛡️ Output B: Engineering Maintenance Priority (Statutory Work Orders)")

    if not maintenance_priority_dict:
        st.info("No maintenance priority data available.")
        return

    status = maintenance_priority_dict.get("status", "NOT_ASSESSED")
    is_abstaining = status == "NOT_ASSESSED"

    if is_abstaining:
        st.markdown(
            """
            <div style="background: #450a0a; border: 2px solid #ef4444; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem;">
                <h3 style="color: #fca5a5; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                    🛑 STATUTORY MAINTENANCE PRIORITY: NOT_ASSESSED
                </h3>
                <p style="color: #fecaca; margin-bottom: 0.5rem; font-size: 0.95rem;">
                    <strong>Mandatory Abstention Enforced:</strong> Automated computer vision systems operating on monocular 2D images
                    cannot observe internal load paths, rebar section loss, or volumetric defect depth. Consequently, issuing actionable
                    statutory work orders or AASHTO condition state ratings is strictly prohibited.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: #1e1b4b; border: 2px solid #6366f1; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem;">
                <h3 style="color: #a5b4fc; margin-top: 0;">
                    🧪 RESEARCH DEMONSTRATION SIMULATION ONLY: {status}
                </h3>
                <p style="color: #c7d2fe; margin-bottom: 0.5rem; font-size: 0.95rem;">
                    {maintenance_priority_dict.get('abstention_reason', '')}
                </p>
                <p style="color: #f87171; font-weight: bold; margin-bottom: 0;">
                    ⚠️ DISCLAIMER: Results are for academic what-if simulation and are NOT approved by any highway agency or DOT.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Official Repair Priority", maintenance_priority_dict.get("official_repair_priority", "NOT_ASSESSED"))
    with col2:
        st.metric("Repair Deadline", str(maintenance_priority_dict.get("official_repair_deadline", "None")))
    with col3:
        st.metric("Structural Safety Rating", str(maintenance_priority_dict.get("official_structural_safety_rating", "None")))
    with col4:
        st.metric("AASHTO Condition State", str(maintenance_priority_dict.get("official_aashto_condition_state", "None")))

    # Missing prerequisites checklist
    st.markdown("#### Statutory Engineering Prerequisites Checklist")
    prereqs = maintenance_priority_dict.get("missing_engineering_prerequisites", [])

    all_prereqs = [
        ("Verified structural component identity (BIM or inspection plan)", bool(engineering_context and engineering_context.structural_component)),
        ("Element load path criticality (AASHTO / NBI element classification)", bool(engineering_context and engineering_context.element_criticality != ElementCriticality.UNKNOWN)),
        ("Physical metric defect dimensions (calibrated optical/laser measurement)", bool(engineering_context and engineering_context.calibrated_dimensions)),
        ("Internal defect depth & delamination plane (NDT sounding, GPR, or UPV)", bool(engineering_context and engineering_context.defect_depth_mm)),
        ("Formal validation and sign-off by a licensed Professional Engineer (PE/SE)", bool(engineering_context and engineering_context.is_verified_by_pe)),
        ("Formally adopted Department of Transportation (DOT) maintenance decision policy", False),
    ]

    for label, satisfied in all_prereqs:
        if satisfied:
            st.markdown(f"✅ ~~{label}~~ *(Provided)*")
        else:
            st.markdown(f"❌ **{label}** *(Missing)*")


def render_export_panel(
    measurement_dict: Optional[Dict[str, Any]],
    severity_dict: Optional[Dict[str, Any]],
    prioritization_dict: Optional[Dict[str, Any]],
    image_id: str = "inspection_image",
):
    """
    Renders export and download buttons for JSON data and inspection reports.
    """
    st.subheader("💾 Export Inspection Artifacts")

    col1, col2 = st.columns(2)

    with col1:
        # Full combined JSON
        combined_payload = {
            "image_id": image_id,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage_2_measurements": measurement_dict,
            "stage_3_visual_severity": severity_dict,
            "stage_4_prioritization": prioritization_dict,
        }
        json_str = json.dumps(combined_payload, indent=2)
        st.download_button(
            label="📥 Download Full Inspection JSON",
            data=json_str,
            file_name=f"{image_id}_full_inspection_data.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        # Markdown Report Generation
        md_report = f"""# Bridge Defect Inspection & Prioritization Report

**Image ID:** `{image_id}`  
**Report Generated (UTC):** {datetime.now(timezone.utc).isoformat()}  
**System Version:** Stage 5 Interactive Dashboard v0.5.0  

---

## 1. Executive Summary
- **Total Defect Findings:** {measurement_dict.get('total_components_count', 0) if measurement_dict else 'N/A'}
- **Detected Classes:** {measurement_dict.get('total_detected_classes', 0) if measurement_dict else 'N/A'}
- **Structural Union Footprint:** {measurement_dict.get('structural_union_pixel_area', 0) if measurement_dict else 'N/A'} px ({measurement_dict.get('structural_union_image_area_pct', 0.0) if measurement_dict else 'N/A'}%)
- **Non-Structural Union Footprint:** {measurement_dict.get('non_structural_union_pixel_area', 0) if measurement_dict else 'N/A'} px ({measurement_dict.get('non_structural_union_image_area_pct', 0.0) if measurement_dict else 'N/A'}%)

---

## 2. Output A: Inspection Review Queue
*(Provisional observation-level ranking for human field verification)*

| Rank | Finding ID | Class | Urgency | Score | Uncertainty |
| :---: | :--- | :--- | :--- | :---: | :---: |
"""
        queue_items = (prioritization_dict or {}).get("output_a_inspection_review_queue", {}).get("ranked_queue_items", [])
        for it in queue_items:
            md_report += f"| #{it.get('rank')} | `{it.get('finding_id')}` | {it.get('class_name')} | {it.get('queue_urgency')} | {it.get('composite_verification_score', 0.0):.4f} | {it.get('optical_uncertainty_level')} |\n"

        md_report += f"""
---

## 3. Output B: Engineering Maintenance Priority
- **Status:** `{((prioritization_dict or {}).get('output_b_engineering_maintenance_priority', {})).get('status', 'NOT_ASSESSED')}`
- **Official Repair Priority:** `NOT_ASSESSED`
- **Repair Deadline:** None
- **Official AASHTO State:** None

### Mandatory Disclaimers:
1. {CALIBRATION_DISCLAIMER}
2. Monocular computer vision observations are decoupled from certified bridge load-bearing safety ratings.
"""
        st.download_button(
            label="📄 Download Inspection Summary Report (.md)",
            data=md_report,
            file_name=f"{image_id}_inspection_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
