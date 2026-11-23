"""
Unit tests for Stage 3: Explainable Visual Severity Assessment Foundation.

Verifies:
1. Segregation of structural vs. non-structural defect coverage (e.g. 11.92% Graffiti vs 0.32% Spalling).
2. Distinction between summed per-class pixel coverage and union pixel coverage.
3. Explicit abstention of official structural condition state (NOT_ASSESSED).
4. Explicit distinction between model segmentation confidence and structural severity confidence.
5. Completeness of missing-data checklist (component identity, importance, depth, section loss, etc.).
6. Generation of structured human-in-the-loop review flags (POSSIBLE_SPALLING_REVIEW, etc.).
7. Per-finding XAI explanations, physical mechanisms, and authoritative literature citations.
8. Machine-readable Stage 4 prioritization contract with automated prioritization abstention.
9. ROI unavailability handling and clean scan (no defects) behavior.
10. Full JSON schema validity and round-trip serialization.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from src.data.mask_converter import DACL10K_CLASSES
from src.measurement import (
    ABSTENTION_JUSTIFICATION,
    CONFIDENCE_DISTINCTION_NOTICE,
    NON_STRUCTURAL_CLASSES,
    DefectMeasurementPipeline,
    ExplainableSeverityReport,
    ExplainableVisualSeverityEvaluator,
    MissingDataChecklist,
    ReviewFlag,
    VisualConcernLevel,
)


class TestExplainableVisualSeverity(unittest.TestCase):
    """Test suite for Stage 3 Explainable Visual Severity Assessment."""

    def setUp(self):
        self.h = 512
        self.w = 512
        self.total_pixels = self.h * self.w  # 262,144 px
        self.num_classes = len(DACL10K_CLASSES)
        self.pipeline = DefectMeasurementPipeline(
            threshold=0.5,
            min_component_area_pixels=10,
        )
        self.evaluator = ExplainableVisualSeverityEvaluator()

    def test_graffiti_and_spalling_coexistence_segregation(self):
        """
        Problem 2 Verification:
        In an inspection image with 11.92% Graffiti coverage and 0.32% Spalling coverage,
        non-structural categories must NOT distort the structural defect quantification.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)

        spall_idx = DACL10K_CLASSES.index("Spalling")
        graffiti_idx = DACL10K_CLASSES.index("Graffiti")

        # 0.3217% of 262,144 pixels is ~843 pixels (e.g. 29 x 29 = 841 px)
        spall_h, spall_w = 29, 29
        probs[spall_idx, 50:50+spall_h, 50:50+spall_w] = 0.90

        # 11.92415% of 262,144 pixels is ~31,258 pixels (e.g. 177 x 177 = 31,329 px)
        graf_h, graf_w = 177, 177
        probs[graffiti_idx, 150:150+graf_h, 150:150+graf_w] = 0.95

        measurement = self.pipeline.measure_probabilities(probs, image_id="spall_graffiti_test")
        report = self.evaluator.evaluate(measurement)

        # Verify area segregation
        expected_spall_area = spall_h * spall_w  # 841
        expected_graf_area = graf_h * graf_w    # 31,329

        self.assertEqual(measurement.structural_union_pixel_area, expected_spall_area)
        self.assertEqual(measurement.non_structural_union_pixel_area, expected_graf_area)

        spall_pct = (expected_spall_area / float(self.total_pixels)) * 100.0
        graf_pct = (expected_graf_area / float(self.total_pixels)) * 100.0

        self.assertAlmostEqual(measurement.structural_union_image_area_pct, spall_pct, places=3)
        self.assertAlmostEqual(measurement.non_structural_union_image_area_pct, graf_pct, places=3)

        # Structural coverage is NOT inflated by Graffiti
        self.assertLess(measurement.structural_union_image_area_pct, 1.0)
        self.assertGreater(measurement.non_structural_union_image_area_pct, 10.0)

        # Visual concern level reflects Spalling (High Concern for cover breach), NOT Graffiti
        self.assertEqual(report.visual_concern_level, VisualConcernLevel.HIGH_VISUAL_CONCERN.value)

        # Both review flags are appropriately raised
        self.assertIn(ReviewFlag.POSSIBLE_SPALLING_REVIEW.value, report.review_flags)
        self.assertIn(ReviewFlag.NON_STRUCTURAL_SURFACE_COVERAGE.value, report.review_flags)

        # Stage 4 contract segregates them cleanly
        s4 = report.stage_4_prioritization_contract
        self.assertAlmostEqual(s4["structural_union_coverage_pct"], spall_pct, places=3)
        self.assertAlmostEqual(s4["non_structural_union_coverage_pct"], graf_pct, places=3)
        self.assertIn("Spalling", s4["primary_visual_concerns"])
        self.assertNotIn("Graffiti", s4["primary_visual_concerns"])

    def test_summed_vs_union_pixel_coverage_overlapping_defects(self):
        """
        Problem 3 Verification:
        When Spalling and ExposedRebars overlap, verify that summed_class_pixel_area
        and union_defect_pixel_area are accurately distinguished.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)

        spall_idx = DACL10K_CLASSES.index("Spalling")
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")

        # Spalling: 20 x 40 = 800 pixels [100:120, 100:140]
        probs[spall_idx, 100:120, 100:140] = 0.88

        # ExposedRebars: 10 x 30 = 300 pixels [110:120, 110:140]
        # Overlaps completely inside the spalling region (10 rows x 30 cols = 300 px overlap)
        probs[rebar_idx, 110:120, 110:140] = 0.94

        measurement = self.pipeline.measure_probabilities(probs, image_id="overlap_test")

        # Summed area = 800 + 300 = 1100
        self.assertEqual(measurement.summed_class_pixel_area, 1100)

        # Union area = 800 (since rebar is fully inside spall)
        self.assertEqual(measurement.union_defect_pixel_area, 800)

        # Overlapping area = 300
        self.assertEqual(measurement.overlapping_defect_pixel_area, 300)

        # Summed area must strictly exceed union area
        self.assertGreater(measurement.summed_class_pixel_area, measurement.union_defect_pixel_area)

    def test_explicit_abstention_of_official_structural_condition(self):
        """
        Problem 4 & Problem 1 Verification:
        Official condition state MUST be NOT_ASSESSED, with explicit justification
        citing unobservable parameters. No numerical safety rating is fabricated.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        spall_idx = DACL10K_CLASSES.index("Spalling")
        probs[spall_idx, 100:130, 100:130] = 0.92

        measurement = self.pipeline.measure_probabilities(probs, image_id="abstention_test")
        report = self.evaluator.evaluate(measurement)

        # Official condition state must abstain
        self.assertEqual(report.official_structural_condition_state, "NOT_ASSESSED")
        self.assertIn("NOT_ASSESSED", report.abstention_reason)
        self.assertIn("AASHTO", report.abstention_reason)

        # Calibrated severity confidence must be None
        self.assertIsNone(report.calibrated_structural_severity_confidence)

        # Model confidence mean is populated from neural network probabilities
        self.assertIsNotNone(report.segmentation_model_confidence_mean)
        self.assertAlmostEqual(report.segmentation_model_confidence_mean, 0.92, places=2)

        # Contract explicitly states model confidence is NOT severity confidence
        self.assertIn("CONFIDENCE CONTRACT", report.confidence_distinction_notice)
        self.assertIn("NOT calibrated engineering confidence", report.confidence_distinction_notice)

    def test_missing_data_checklist_completeness(self):
        """Verify all 7 required missing structural parameters are documented."""
        checklist = MissingDataChecklist()
        chk_dict = checklist.to_dict()

        required_keys = [
            "structural_component_identity",
            "element_importance",
            "physical_defect_dimensions",
            "internal_defect_depth",
            "reinforcement_section_loss",
            "inspection_history",
            "engineering_assessment",
        ]

        for key in required_keys:
            self.assertIn(key, chk_dict)
            self.assertEqual(chk_dict[key]["status"], "MISSING")
            self.assertTrue(len(chk_dict[key]["required_source"]) > 5)
            self.assertTrue(len(chk_dict[key]["description"]) > 5)

    def test_review_flags_triggering_and_actions(self):
        """Verify that specific defect classes trigger corresponding human review flags."""
        test_cases = [
            ("Spalling", ReviewFlag.POSSIBLE_SPALLING_REVIEW.value),
            ("ExposedRebars", ReviewFlag.REBAR_EXPOSURE_REVIEW.value),
            ("Crack", ReviewFlag.CRACK_DELAMINATION_REVIEW.value),
            ("Bearing", ReviewFlag.BEARING_ARTICULATION_REVIEW.value),
            ("EJoint", ReviewFlag.EXPANSION_JOINT_REVIEW.value),
            ("Rockpocket", ReviewFlag.HONEYCOMBING_CONCRETE_REVIEW.value),
            ("Efflorescence", ReviewFlag.SURFACE_MOISTURE_LEACHING_REVIEW.value),
        ]

        for class_name, expected_flag in test_cases:
            probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
            c_idx = DACL10K_CLASSES.index(class_name)
            probs[c_idx, 50:80, 50:80] = 0.90

            meas = self.pipeline.measure_probabilities(probs, image_id=f"test_{class_name}")
            rep = self.evaluator.evaluate(meas)

            self.assertIn(
                expected_flag, rep.review_flags,
                f"Class {class_name} failed to trigger {expected_flag}"
            )
            self.assertTrue(len(rep.observations) >= 1)
            obs = rep.observations[0]
            self.assertTrue(len(obs.inspector_action_items) >= 1)

    def test_per_finding_xai_explanations_and_literature(self):
        """Verify that every observation has complete XAI explanation, mechanism, and citation."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        crack_idx = DACL10K_CLASSES.index("Crack")
        # Elongated vertical crack
        probs[crack_idx, 100:250, 200:202] = 0.92

        meas = self.pipeline.measure_probabilities(probs, image_id="xai_crack_test")
        rep = self.evaluator.evaluate(meas)

        self.assertEqual(len(rep.observations), 1)
        obs = rep.observations[0]

        self.assertTrue(obs.finding_id.startswith("FINDING-CRACK-"))
        self.assertTrue(obs.is_structural)
        self.assertIn("Crack", obs.visual_evidence)
        self.assertIn("skeleton", obs.visual_evidence.lower())
        self.assertIn("tensile", obs.explanation.lower())
        self.assertIn("AASHTO MBEI", obs.literature_reference)
        self.assertIn("FIB Model Code", obs.literature_reference)

        # Action items must be concrete
        action_text = " ".join(obs.inspector_action_items)
        self.assertIn("crack comparator", action_text)

    def test_stage_4_prioritization_contract_abstention(self):
        """Verify Stage 4 contract explicitly abstains from automated scheduling."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")
        probs[rebar_idx, 100:150, 100:150] = 0.95

        meas = self.pipeline.measure_probabilities(probs, image_id="s4_contract_test")
        rep = self.evaluator.evaluate(meas)

        s4 = rep.stage_4_prioritization_contract
        self.assertFalse(s4["ready_for_automated_prioritization"])
        self.assertIn("abstention_reason", s4)
        self.assertIn(ReviewFlag.REBAR_EXPOSURE_REVIEW.value, s4["candidate_review_flags"])
        self.assertIn("ExposedRebars", s4["primary_visual_concerns"])
        self.assertEqual(s4["actionable_observations_count"], 1)

    def test_roi_unavailable_indication(self):
        """Verify that when no ROI is provided, roi_available is False and ROI metrics are None."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        meas = self.pipeline.measure_probabilities(probs, image_id="no_roi_test")
        rep = self.evaluator.evaluate(meas)

        self.assertFalse(meas.roi_present)
        self.assertIsNone(meas.roi_pixel_area)
        self.assertIsNone(meas.roi_relative_area_pct)

        roi_cov = rep.coverage_breakdown["roi_coverage"]
        self.assertFalse(roi_cov["roi_available"])
        self.assertIsNone(roi_cov["union_defect_roi_area_pct"])
        self.assertIsNone(roi_cov["structural_union_roi_area_pct"])

    def test_clean_scan_no_defects(self):
        """Verify clean inspection image with no detections above threshold."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        meas = self.pipeline.measure_probabilities(probs, image_id="clean_scan")
        rep = self.evaluator.evaluate(meas)

        self.assertEqual(rep.visual_concern_level, VisualConcernLevel.NO_DEFECTS_DETECTED.value)
        self.assertEqual(rep.official_structural_condition_state, "NOT_ASSESSED")
        self.assertEqual(len(rep.observations), 0)
        self.assertEqual(len(rep.review_flags), 0)
        self.assertEqual(rep.coverage_breakdown["union_defect_pixel_area"], 0)
        self.assertEqual(rep.coverage_breakdown["structural_union_pixel_area"], 0)

    def test_json_schema_export_and_reloading(self):
        """Verify ExplainableSeverityReport exports to valid JSON and reloads cleanly."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")
        graf_idx = DACL10K_CLASSES.index("Graffiti")
        probs[rebar_idx, 30:50, 30:50] = 0.94
        probs[graf_idx, 100:150, 100:150] = 0.89

        meas = self.pipeline.measure_probabilities(probs, image_id="json_roundtrip_test")
        rep = self.evaluator.evaluate(meas)

        json_str = rep.to_json(indent=2)
        parsed = json.loads(json_str)

        self.assertIn("image_id", parsed)
        self.assertIn("visual_severity_assessment", parsed)
        self.assertIn("coverage_breakdown", parsed)
        self.assertIn("confidence_contract", parsed)
        self.assertIn("missing_data_checklist", parsed)
        self.assertIn("observations", parsed)
        self.assertIn("stage_4_prioritization_contract", parsed)
        self.assertIn("disclaimer", parsed)

        # Check nested structures
        vsa = parsed["visual_severity_assessment"]
        self.assertEqual(vsa["visual_concern_level"], "HIGH_VISUAL_CONCERN")
        self.assertEqual(vsa["official_structural_condition_state"], "NOT_ASSESSED")
        self.assertIn("REBAR_EXPOSURE_REVIEW", vsa["review_flags"])
        self.assertIn("NON_STRUCTURAL_SURFACE_COVERAGE", vsa["review_flags"])

        self.assertEqual(len(parsed["observations"]), 2)


if __name__ == "__main__":
    unittest.main()
