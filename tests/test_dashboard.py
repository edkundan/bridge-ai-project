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


if __name__ == "__main__":
    unittest.main()
