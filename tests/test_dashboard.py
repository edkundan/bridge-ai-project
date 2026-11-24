"""
Unit tests for Stage 5 Interactive Explainable AI Bridge Inspection Dashboard.

Verifies:
1. Discovery and loading of all demonstration cases (dacl10k_v2_train_0722, dacl10k_v2_validation_0000, etc.)
2. Real-image 19-findings integration contract (7 structural, 12 non-structural)
3. End-to-end live pipeline execution without errors
4. Strict abstention and calibration disclaimer invariants
5. Research demonstration simulation mode isolation
6. Export artifact consistency
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from src.dashboard.data_loader import (
    AVAILABLE_DEMO_CASES,
    load_saved_demo_case,
    run_live_pipeline,
)
from src.measurement.schema import CALIBRATION_DISCLAIMER
from src.prioritization.schema import (
    ElementCriticality,
    MaintenancePriorityStatus,
    VerifiedEngineeringContext,
)


class TestDashboardIntegration(unittest.TestCase):
    """Test suite for Stage 5 dashboard data loading, execution, and guardrails."""

    def test_demo_cases_discovery_and_keys(self):
        """Verify all required demonstration cases are registered."""
        required_cases = [
            "dacl10k_v2_train_0722",
            "dacl10k_v2_validation_0000",
            "graffiti_spalling_case",
            "synthetic_demo_image",
        ]
        for cid in required_cases:
            self.assertIn(cid, AVAILABLE_DEMO_CASES, f"Missing demo case registration: {cid}")
            meta = AVAILABLE_DEMO_CASES[cid]
            self.assertTrue(Path(meta["dir"]).exists(), f"Directory not found: {meta['dir']}")

    def test_train_0722_19_findings_contract(self):
        """
        Requirement 10: Real-image integration with 19 findings,
        7 structural-relevant and 12 non-structural.
        """
        data = load_saved_demo_case("dacl10k_v2_train_0722")
        self.assertIsNotNone(data["measurement_dict"])
        self.assertIsNotNone(data["review_queue_dict"])
        self.assertIsNotNone(data["maintenance_priority_dict"])

        queue = data["review_queue_dict"]
        summary = queue.get("queue_summary", {})
        total = summary.get("total_findings_count")
        struct = summary.get("structural_findings_count")
        non_struct = summary.get("non_structural_findings_count")

        self.assertEqual(total, 19, f"Expected 19 total findings, got {total}")
        self.assertEqual(struct, 7, f"Expected 7 structural findings, got {struct}")
        self.assertEqual(non_struct, 12, f"Expected 12 non-structural findings, got {non_struct}")
        self.assertEqual(struct + non_struct, total)

        # Check ranking items
        items = queue.get("ranked_queue_items", [])
        self.assertEqual(len(items), 19)
        for i, it in enumerate(items):
            self.assertEqual(it["rank"], i + 1)
            self.assertTrue(it["finding_id"].startswith("FINDING-"))
            self.assertIn("composite_verification_score", it)
            self.assertIn("optical_uncertainty_level", it)

    def test_validation_0000_saved_outputs(self):
        """
        Requirement 10: Validate dacl10k_v2_validation_0000 saved demonstration outputs.
        """
        data = load_saved_demo_case("dacl10k_v2_validation_0000")
        self.assertIsNotNone(data["image_rgb"])
        self.assertIsNotNone(data["overlay_rgb"])
        self.assertIsNotNone(data["measurement_dict"])
        self.assertIsNotNone(data["review_queue_dict"])
        self.assertIsNotNone(data["maintenance_priority_dict"])

        # Check abstention
        maint = data["maintenance_priority_dict"]
        self.assertEqual(maint.get("status"), "NOT_ASSESSED")
        self.assertEqual(maint.get("official_repair_priority"), "NOT_ASSESSED")
        self.assertIsNone(maint.get("official_repair_deadline"))

    def test_end_to_end_live_pipeline_synthetic(self):
        """
        Requirement 1 & 9: Verify live pipeline runs cleanly on synthetic array without checkpoint.
        """
        synth_rgb = np.zeros((200, 200, 3), dtype=np.uint8)
        synth_rgb[50:100, 50:100] = [180, 180, 180]

        # Simulate precomputed probability map with a crack and a spall
        probs = np.zeros((19, 200, 200), dtype=np.float32)
        probs[3, 60:80, 60:62] = 0.92   # Crack (idx 3)
        probs[15, 120:150, 120:150] = 0.89  # Spalling (idx 15)

        res = run_live_pipeline(
            image_rgb=synth_rgb,
            image_id="test_synth_live",
            probability_maps=probs,
            threshold=0.5,
            min_component_area_pixels=5,
        )

        self.assertIn("measurement_result", res)
        self.assertIn("stage3_report", res)
        self.assertIn("stage4_result", res)
        self.assertIn("overlay_rgb", res)

        m = res["measurement_result"]
        self.assertEqual(m.total_detected_classes, 2)
        self.assertEqual(m.total_components_count, 2)

        # Output B must abstain
        self.assertEqual(res["stage4_result"].engineering_maintenance_priority.status, MaintenancePriorityStatus.NOT_ASSESSED)

    def test_metric_calibration_guardrail(self):
        """
        Requirement 3: Never label pixel widths as millimetres.
        """
        data = load_saved_demo_case("dacl10k_v2_train_0722")
        m_dict = data["measurement_dict"]
        self.assertIn("disclaimer", m_dict.get("metadata", {}))
        self.assertIn("pixel-domain", m_dict["metadata"]["disclaimer"].lower())

        # Verify no component shape feature reports 'width_mm'
        for cname, cdata in m_dict.get("class_measurements", {}).items():
            for comp in cdata.get("components", []):
                shapes = comp.get("shape_features", {})
                self.assertNotIn("width_mm", shapes)
                self.assertNotIn("length_mm", shapes)
                if shapes.get("estimated_avg_width_pixels") is not None:
                    self.assertIsInstance(shapes["estimated_avg_width_pixels"], (int, float))

    def test_research_demonstration_mode_isolation(self):
        """
        Requirement 6: Verify simulation mode is clearly flagged and isolated.
        """
        synth_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        probs = np.zeros((19, 100, 100), dtype=np.float32)
        probs[15, 20:40, 20:40] = 0.90  # Spalling

        ctx = VerifiedEngineeringContext(
            bridge_id="TEST-01",
            structural_component="Pier Column P2",
            element_criticality=ElementCriticality.PRIMARY_LOAD_BEARING,
            calibrated_dimensions={"width_mm": 100.0, "length_mm": 200.0},
            defect_depth_mm=30.0,
            is_verified_by_pe=True,
        )

        res = run_live_pipeline(
            image_rgb=synth_rgb,
            image_id="test_demo_policy",
            probability_maps=probs,
            threshold=0.5,
            engineering_context=ctx,
            use_research_demo_policy=True,
        )

        maint = res["stage4_result"].engineering_maintenance_priority
        self.assertEqual(maint.status, MaintenancePriorityStatus.ASSESSED_RESEARCH_DEMO_ONLY)
        self.assertIn("RESEARCH SIMULATION", maint.abstention_reason)
        self.assertIsNotNone(maint.disclaimer)

    def test_safe_render_dataframe_and_fallback(self):
        """Verify safe_render_dataframe uses st.dataframe and falls back gracefully to HTML rendering."""
        import streamlit as st
        from src.dashboard.components import safe_render_dataframe

        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})

        # Test normal path
        with patch.object(st, "dataframe") as mock_df, patch.object(st, "markdown") as mock_md:
            safe_render_dataframe(df)
            mock_df.assert_called_once()
            mock_md.assert_not_called()

        # Test fallback path when pyarrow/dataframe serialization fails (must NOT call st.table)
        with patch.object(st, "dataframe", side_effect=ImportError("pyarrow requires NumPy 2.0 or newer")), patch.object(st, "markdown") as mock_md:
            safe_render_dataframe(df)
            mock_md.assert_called_once()

    def test_measurement_summary_extraction_and_consistency(self):
        """
        Regression test: Verify extract_measurement_summary properly parses
        nonzero counts and areas from nested summary, coverage_metrics, and top-level.
        """
        from src.dashboard.components import extract_measurement_summary

        data = load_saved_demo_case("dacl10k_v2_train_0722")
        m_dict = data["measurement_dict"]
        metrics = extract_measurement_summary(m_dict)

        # Must not be 0 when defects exist!
        self.assertEqual(metrics["total_findings"], 19, f"Expected 19 findings, got {metrics['total_findings']}")
        self.assertEqual(metrics["total_classes"], 2, f"Expected 2 detected classes, got {metrics['total_classes']}")
        self.assertGreater(metrics["structural_union_pixel_area"], 0)
        self.assertGreater(metrics["non_structural_union_pixel_area"], 0)
        self.assertGreater(metrics["union_defect_pixel_area"], 0)
        self.assertGreater(metrics["structural_union_image_area_pct"], 0.0)

        # Verify Stage 2, Stage 3, and Stage 4 consistency
        queue = data["review_queue_dict"]
        self.assertEqual(
            queue["queue_summary"]["total_findings_count"],
            metrics["total_findings"],
            "Review queue total must match Stage 2 total findings",
        )

    def test_clean_overlay_and_selective_highlighting(self):
        """
        Regression test: Verify DefectVisualizer defaults show_labels=False (clean overlay)
        and supports selective highlighting for finding and class.
        """
        from src.measurement.visualization import DefectVisualizer

        img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        probs = np.zeros((19, 100, 100), dtype=np.float32)
        probs[3, 20:40, 20:30] = 0.95  # Crack
        probs[15, 60:80, 60:80] = 0.88  # Spalling

        res = run_live_pipeline(
            image_rgb=img_rgb,
            image_id="test_overlay_clean",
            probability_maps=probs,
            threshold=0.5,
            min_component_area_pixels=5,
        )

        visualizer = DefectVisualizer()
        # 1. Clean overlay by default
        clean_overlay = visualizer.render_overlay(
            image_rgb=img_rgb,
            measurement=res["measurement_dict"],
            probability_maps=probs,
            show_labels=False,
        )
        self.assertEqual(clean_overlay.shape[1], 100)

        # 2. Selective highlight of a specific finding
        highlight_overlay = visualizer.render_overlay(
            image_rgb=img_rgb,
            measurement=res["measurement_dict"],
            probability_maps=probs,
            show_labels=False,
            highlight_finding_id="FINDING-CRACK-001",
        )
        self.assertIsNotNone(highlight_overlay)

        # 3. Selective highlight of a specific class
        class_overlay = visualizer.render_overlay(
            image_rgb=img_rgb,
            measurement=res["measurement_dict"],
            probability_maps=probs,
            show_labels=False,
            highlight_class="Crack",
        )
        self.assertIsNotNone(class_overlay)

    def test_images_jpeg_regression_real_image(self):
        """
        Regression test: Verify uploaded inspection case (images.jpeg) with Experiment 02
        checkpoint at threshold 0.50 and min_area 10 produces:
        - Exactly 32 findings in queue (24 structural + 8 non-structural = 32)
        - Measurement component count agrees with queue (32 == 32)
        - Exactly 10 detected classes
        - Nonzero union areas (structural > 0, non-structural > 0, union > 0)
        - extract_measurement_summary produces identical results from measurement_result and measurement_dict
        - Exported report strings and JSON contain matching metrics
        """
        import cv2
        from src.dashboard.components import extract_measurement_summary

        img_path = Path("demo_data/uploaded_test_cases/images.jpeg")
        ckpt_path = Path("experiments/exp_02_sqrt_weighted_bcedice/checkpoints/best_model.pth")
        if not img_path.exists() or not ckpt_path.exists():
            self.skipTest("images.jpeg or exp_02 checkpoint not available")

        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        res = run_live_pipeline(
            image_rgb=rgb,
            image_id="images_jpeg_regression",
            checkpoint_path=str(ckpt_path),
            threshold=0.50,
            min_component_area_pixels=10,
            device="cpu",
        )

        m_res = res["measurement_result"]
        m_dict = res["measurement_dict"]
        q_dict = res["review_queue_dict"]
        summary_res = extract_measurement_summary(m_res)
        summary_dict = extract_measurement_summary(m_dict)

        # 1. Queue contains 32 findings (24 structural + 8 non-structural = 32)
        q_sum = q_dict["queue_summary"]
        self.assertEqual(q_sum["total_findings_count"], 32)
        self.assertEqual(q_sum["structural_findings_count"], 24)
        self.assertEqual(q_sum["non_structural_findings_count"], 8)
        self.assertEqual(q_sum["structural_findings_count"] + q_sum["non_structural_findings_count"], 32)
        self.assertEqual(len(q_dict["ranked_queue_items"]), 32)

        # 2. Measurement component count agrees with queue
        self.assertEqual(m_res.total_components_count, 32)
        self.assertEqual(summary_res["total_findings"], 32)
        self.assertEqual(summary_dict["total_findings"], 32)
        self.assertEqual(summary_dict["total_findings"], q_sum["total_findings_count"])

        # 3. Detected-class count is nonzero and matches class summary (10 classes)
        self.assertEqual(m_res.total_detected_classes, 10)
        self.assertEqual(summary_res["total_classes"], 10)
        self.assertEqual(summary_dict["total_classes"], 10)
        detected_in_table = [c for c, d in m_dict["class_measurements"].items() if d.get("detected")]
        self.assertEqual(len(detected_in_table), 10)

        # 4. Nonzero union area is never displayed as zero
        self.assertGreater(summary_res["structural_union_pixel_area"], 0)
        self.assertGreater(summary_res["structural_union_image_area_pct"], 0.0)
        self.assertGreater(summary_res["non_structural_union_pixel_area"], 0)
        self.assertGreater(summary_res["non_structural_union_image_area_pct"], 0.0)
        self.assertGreater(summary_res["union_defect_pixel_area"], 0)
        self.assertGreater(summary_res["union_defect_image_area_pct"], 0.0)

        # 5. Consistency between measurement_result and measurement_dict
        for k in ["total_findings", "total_classes", "structural_union_pixel_area", "non_structural_union_pixel_area", "union_defect_pixel_area"]:
            self.assertEqual(summary_res[k], summary_dict[k], f"Mismatch for key {k}")

        # 6. Engineering maintenance priority must abstain (NOT_ASSESSED)
        maint = res["maintenance_priority_dict"]
        self.assertEqual(maint["status"], "NOT_ASSESSED")
        self.assertEqual(maint["official_repair_priority"], "NOT_ASSESSED")

    def test_empty_prediction_handling_reports_zero(self):
        """
        Regression test: Verify that an all-zero prediction correctly produces
        0 findings, 0 detected classes, and 0.0% coverage without errors.
        """
        from src.dashboard.components import extract_measurement_summary

        clean_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        clean_probs = np.zeros((19, 100, 100), dtype=np.float32)

        res = run_live_pipeline(
            image_rgb=clean_rgb,
            image_id="clean_scan",
            probability_maps=clean_probs,
            threshold=0.50,
            min_component_area_pixels=10,
        )

        m_res = res["measurement_result"]
        m_dict = res["measurement_dict"]
        summary = extract_measurement_summary(m_dict)

        self.assertEqual(m_res.total_components_count, 0)
        self.assertEqual(m_res.total_detected_classes, 0)
        self.assertEqual(summary["total_findings"], 0)
        self.assertEqual(summary["total_classes"], 0)
        self.assertEqual(summary["structural_union_pixel_area"], 0)
        self.assertEqual(summary["structural_union_image_area_pct"], 0.0)
        self.assertEqual(summary["union_defect_pixel_area"], 0)
        self.assertEqual(summary["union_defect_image_area_pct"], 0.0)

        q_dict = res["review_queue_dict"]
        self.assertEqual(q_dict["queue_summary"]["total_findings_count"], 0)
        self.assertEqual(len(q_dict["ranked_queue_items"]), 0)

    def test_overlay_toggles_and_clean_default(self):
        """
        Verify DefectVisualizer render_overlay supports:
        - Clean default: show_bboxes=False, show_labels=False
        - Toggling show_masks=False preserves image without mask blending
        - Selective target finding prominence
        """
        from src.measurement.visualization import DefectVisualizer

        img_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
        probs = np.zeros((19, 100, 100), dtype=np.float32)
        probs[3, 20:40, 20:40] = 0.95  # Crack

        res = run_live_pipeline(
            image_rgb=img_rgb,
            image_id="test_toggles",
            probability_maps=probs,
            threshold=0.5,
            min_component_area_pixels=5,
        )

        visualizer = DefectVisualizer()

        # 1. Masks off
        no_masks = visualizer.render_overlay(
            image_rgb=img_rgb,
            measurement=res["measurement_result"],
            probability_maps=probs,
            show_masks=False,
            show_banner=False,
        )
        # Without banner and masks, pixels outside annotations should remain untouched
        np.testing.assert_array_equal(no_masks, img_rgb)

        # 2. Masks on (default clean view)
        with_masks = visualizer.render_overlay(
            image_rgb=img_rgb,
            measurement=res["measurement_result"],
            probability_maps=probs,
            show_masks=True,
            show_bboxes=False,
            show_labels=False,
            show_banner=False,
        )
        self.assertFalse(np.array_equal(with_masks, img_rgb))


if __name__ == "__main__":
    unittest.main()
