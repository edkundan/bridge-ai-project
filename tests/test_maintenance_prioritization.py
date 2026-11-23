"""
Unit tests for Stage 4: Explainable, Evidence-Aware Maintenance Prioritization Foundation.

Verifies:
1. Dual distinct outputs: Inspection Review Queue vs Engineering Maintenance Priority.
2. Production policy strict abstention: Engineering priority is NOT_ASSESSED with missing prerequisites.
3. Non-structural isolation: Graffiti (even at 12% coverage) cannot inflate structural urgency or outrank structural defects.
4. Regression Test for Requirement 9: Set-theoretic coverage consistency invariant:
   structural_union + non_structural_union - overlap == total_union.
5. Optical uncertainty modeling and false-positive sensitivity flagging.
6. Contradiction detection between visual findings and external engineering inputs.
7. Human inspector override record tracking and audit logging.
8. Research demonstration policy execution and explicit simulation labeling.
9. Clean inspection scans (no defects) behavior.
10. Complete machine-readable JSON schema validation and round-trip serialization.
"""

import json
import os
import unittest
from pathlib import Path

import numpy as np

from src.data.mask_converter import DACL10K_CLASSES
from src.measurement import (
    DefectMeasurementPipeline,
    ExplainableSeverityReport,
    ExplainableVisualSeverityEvaluator,
    VisualConcernLevel,
)
from src.prioritization import (
    ElementCriticality,
    EngineeringMaintenancePriority,
    HumanOverrideRecord,
    InspectionReviewQueue,
    MaintenancePrioritizationPipeline,
    MaintenancePriorityStatus,
    ProductionAbstentionPolicy,
    ResearchDemonstrationPolicy,
    ReviewQueueItem,
    ReviewQueueUrgency,
    Stage4PrioritizationResult,
    VerifiedEngineeringContext,
)


class TestMaintenancePrioritization(unittest.TestCase):
    """Test suite for Stage 4 Maintenance Prioritization Foundation."""

    def setUp(self):
        self.h = 512
        self.w = 512
        self.total_pixels = self.h * self.w
        self.num_classes = len(DACL10K_CLASSES)
        self.meas_pipeline = DefectMeasurementPipeline(
            threshold=0.5,
            min_component_area_pixels=10,
        )
        self.sev_evaluator = ExplainableVisualSeverityEvaluator()
        self.stage4_pipeline = MaintenancePrioritizationPipeline()

    def test_set_theoretic_coverage_consistency_regression(self):
        """
        REGRESSION TEST FOR REQUIREMENT 9:
        Verifies that structural_union_pixel_area + non_structural_union_pixel_area - overlap
        strictly equals union_defect_pixel_area for all mask configurations.
        """
        # Case A: Disjoint structural and non-structural defects (zero overlap)
        probs_disjoint = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        spall_idx = DACL10K_CLASSES.index("Spalling")
        graf_idx = DACL10K_CLASSES.index("Graffiti")

        probs_disjoint[spall_idx, 50:80, 50:80] = 0.90       # 30 x 30 = 900 px (structural)
        probs_disjoint[graf_idx, 150:250, 150:250] = 0.95    # 100 x 100 = 10,000 px (non-structural)

        meas_disjoint = self.meas_pipeline.measure_probabilities(probs_disjoint, image_id="disjoint_test")

        struct_a = meas_disjoint.structural_union_pixel_area
        non_struct_a = meas_disjoint.non_structural_union_pixel_area
        total_union_a = meas_disjoint.union_defect_pixel_area
        overlap_a = meas_disjoint.overlapping_defect_pixel_area

        self.assertEqual(overlap_a, 0)
        self.assertEqual(struct_a, 900)
        self.assertEqual(non_struct_a, 10000)
        # Invariant check:
        self.assertEqual(struct_a + non_struct_a - overlap_a, total_union_a)
        self.assertEqual(total_union_a, 10900)

        # Case B: Partially overlapping structural and non-structural defects
        probs_overlap = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        # Spalling: [100:150, 100:150] = 50 x 50 = 2500 px
        probs_overlap[spall_idx, 100:150, 100:150] = 0.90
        # Graffiti: [125:175, 125:175] = 50 x 50 = 2500 px
        # Overlaps on [125:150, 125:150] = 25 x 25 = 625 px
        probs_overlap[graf_idx, 125:175, 125:175] = 0.95

        meas_overlap = self.meas_pipeline.measure_probabilities(probs_overlap, image_id="overlap_test")

        struct_b = meas_overlap.structural_union_pixel_area
        non_struct_b = meas_overlap.non_structural_union_pixel_area
        total_union_b = meas_overlap.union_defect_pixel_area
        overlap_b = meas_overlap.overlapping_defect_pixel_area

        self.assertEqual(struct_b, 2500)
        self.assertEqual(non_struct_b, 2500)
        self.assertEqual(overlap_b, 625)
        # Invariant check:
        self.assertEqual(struct_b + non_struct_b - overlap_b, total_union_b)
        self.assertEqual(total_union_b, 4375)

    def test_dual_distinct_outputs_contract(self):
        """
        Requirement 1: Verify Stage 4 produces two distinct outputs:
        - Output A: InspectionReviewQueue (ranked human review queue)
        - Output B: EngineeringMaintenancePriority (strictly NOT_ASSESSED)
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")
        probs[rebar_idx, 50:80, 50:80] = 0.95

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="dual_output_test")
        sev = self.sev_evaluator.evaluate(meas)
        res = self.stage4_pipeline.run(sev)

        # Check Output A
        self.assertIsInstance(res.inspection_review_queue, InspectionReviewQueue)
        queue = res.inspection_review_queue
        self.assertEqual(queue.total_findings_count, 1)
        self.assertEqual(queue.structural_findings_count, 1)
        self.assertEqual(queue.highest_queue_urgency, ReviewQueueUrgency.URGENT_INSPECTION_REVIEW.value)

        # Check Output B
        self.assertIsInstance(res.engineering_maintenance_priority, EngineeringMaintenancePriority)
        maint = res.engineering_maintenance_priority
        self.assertEqual(maint.status, MaintenancePriorityStatus.NOT_ASSESSED)
        self.assertEqual(maint.official_repair_priority, "NOT_ASSESSED")
        self.assertIsNone(maint.official_repair_deadline)
        self.assertIsNone(maint.official_structural_safety_rating)
        self.assertIsNone(maint.official_aashto_condition_state)
        self.assertFalse(maint.decision_criteria_approved_by_agency)
        self.assertGreater(len(maint.missing_engineering_prerequisites), 3)

    def test_graffiti_isolation_and_ranking(self):
        """
        Requirement 7 & 1: Massive graffiti coverage (12%) must be categorized
        as INFORMATIONAL_NON_STRUCTURAL and must rank BELOW small structural spalling (0.3%).
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        spall_idx = DACL10K_CLASSES.index("Spalling")
        graf_idx = DACL10K_CLASSES.index("Graffiti")

        # Spall = 800 pixels (~0.3%)
        probs[spall_idx, 50:70, 50:90] = 0.90
        # Graffiti = 30,000 pixels (~11.4%)
        probs[graf_idx, 150:300, 150:350] = 0.95

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="graffiti_spall_rank")
        sev = self.sev_evaluator.evaluate(meas)
        res = self.stage4_pipeline.run(sev)

        queue = res.inspection_review_queue
        self.assertEqual(queue.total_findings_count, 2)

        # Rank 1 must be Spalling
        rank_1 = queue.queue_items[0]
        self.assertEqual(rank_1.class_name, "Spalling")
        self.assertEqual(rank_1.queue_urgency, ReviewQueueUrgency.URGENT_INSPECTION_REVIEW)
        self.assertTrue(rank_1.is_structural)

        # Rank 2 must be Graffiti
        rank_2 = queue.queue_items[1]
        self.assertEqual(rank_2.class_name, "Graffiti")
        self.assertEqual(rank_2.queue_urgency, ReviewQueueUrgency.INFORMATIONAL_NON_STRUCTURAL)
        self.assertFalse(rank_2.is_structural)

        # Score of Spalling must exceed Graffiti
        self.assertGreater(rank_1.composite_verification_score, rank_2.composite_verification_score)

    def test_optical_uncertainty_and_small_defect_handling(self):
        """
        Requirement 8 & 10: Low confidence or near-threshold defect triggers HIGH optical uncertainty.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        crack_idx = DACL10K_CLASSES.index("Crack")
        # Small crack (15 px) with borderline confidence (0.55)
        probs[crack_idx, 100:115, 100:101] = 0.55

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="uncertain_crack")
        sev = self.sev_evaluator.evaluate(meas)
        res = self.stage4_pipeline.run(sev)

        queue = res.inspection_review_queue
        self.assertEqual(queue.total_findings_count, 1)
        item = queue.queue_items[0]

        self.assertEqual(item.optical_uncertainty_level, "HIGH")
        self.assertGreater(len(item.uncertainty_factors), 0)

    def test_contradiction_detection(self):
        """
        Requirement 8: Flag contradictions between visual evidence (ExposedRebars)
        and supplied engineering context (claimed non-structural or zero depth).
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")
        probs[rebar_idx, 100:120, 100:120] = 0.92

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="contradiction_test")
        sev = self.sev_evaluator.evaluate(meas)

        # Contradictory engineering context claiming element is NON_STRUCTURAL and depth is 0
        ctx = VerifiedEngineeringContext(
            structural_component="Barrier Rail",
            element_criticality=ElementCriticality.NON_STRUCTURAL,
            defect_depth_mm=0.0,
        )

        res = self.stage4_pipeline.run(sev, engineering_context=ctx)
        maint = res.engineering_maintenance_priority

        self.assertTrue(len(maint.contradiction_warnings) >= 1)
        all_warnings = " ".join(maint.contradiction_warnings)
        self.assertIn("CONTRADICTION", all_warnings)

    def test_human_override_audit_trail(self):
        """
        Requirement 8: Certified human inspector override modifies finding urgency
        and records full audit trail.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        crack_idx = DACL10K_CLASSES.index("Crack")
        probs[crack_idx, 100:150, 100:102] = 0.90

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="override_test")
        sev = self.sev_evaluator.evaluate(meas)

        finding_id = sev.observations[0].finding_id

        # Human override elevating routine/priority to URGENT due to shear crack observation
        override = HumanOverrideRecord(
            override_id="OVR-2026-001",
            finding_id=finding_id,
            inspector_id="PE-7429",
            timestamp_utc="2026-11-23T14:00:00Z",
            original_urgency="PRIORITY_INSPECTION_REVIEW",
            overridden_urgency="URGENT_INSPECTION_REVIEW",
            justification="Field inspection confirms active 45-degree diagonal shear crack across web.",
        )

        res = self.stage4_pipeline.run(sev, human_overrides=[override])
        queue = res.inspection_review_queue
        item = queue.queue_items[0]

        self.assertEqual(item.queue_urgency, ReviewQueueUrgency.URGENT_INSPECTION_REVIEW)
        self.assertIsNotNone(item.human_override)
        self.assertEqual(item.human_override.inspector_id, "PE-7429")
        self.assertIn("RULE_HUMAN_OVERRIDE", " ".join(item.triggered_policy_rules))

    def test_research_demonstration_policy_flagging(self):
        """
        Requirement 5: Research demonstration policy is clearly separated, labeled,
        and includes prominent simulation disclaimer.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        spall_idx = DACL10K_CLASSES.index("Spalling")
        probs[spall_idx, 50:80, 50:80] = 0.90

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="demo_policy_test")
        sev = self.sev_evaluator.evaluate(meas)

        demo_pipeline = MaintenancePrioritizationPipeline(use_research_demo=True)

        ctx = VerifiedEngineeringContext(
            structural_component="Pier Column P-2",
            element_criticality=ElementCriticality.PRIMARY_LOAD_BEARING,
            calibrated_dimensions={"area_m2": 0.08},
            defect_depth_mm=30.0,
            is_verified_by_pe=True,
        )

        res = demo_pipeline.run(sev, engineering_context=ctx)
        self.assertEqual(res.policy_type, "RESEARCH_DEMONSTRATION_SIMULATION_ONLY")
        maint = res.engineering_maintenance_priority

        self.assertEqual(maint.status, MaintenancePriorityStatus.ASSESSED_RESEARCH_DEMO_ONLY)
        self.assertIn("RESEARCH SIMULATION MODE ONLY", maint.abstention_reason)
        self.assertIn("RESEARCH DEMONSTRATION SIMULATION ONLY", maint.disclaimer)

    def test_clean_image_handling(self):
        """
        Requirement 10: Clean images produce NO_ACTION_REQUIRED with 0 findings.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        meas = self.meas_pipeline.measure_probabilities(probs, image_id="clean_image_stage4")
        sev = self.sev_evaluator.evaluate(meas)
        res = self.stage4_pipeline.run(sev)

        queue = res.inspection_review_queue
        self.assertEqual(queue.total_findings_count, 0)
        self.assertEqual(queue.highest_queue_urgency, ReviewQueueUrgency.NO_ACTION_REQUIRED.value)

        maint = res.engineering_maintenance_priority
        self.assertEqual(maint.status, MaintenancePriorityStatus.NOT_ASSESSED)

    def test_json_schema_validity_and_roundtrip(self):
        """
        Requirement 10: Export to full JSON and roundtrip reload.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")
        graf_idx = DACL10K_CLASSES.index("Graffiti")
        probs[rebar_idx, 50:80, 50:80] = 0.95
        probs[graf_idx, 200:300, 200:300] = 0.90

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="stage4_roundtrip")
        sev = self.sev_evaluator.evaluate(meas)
        res = self.stage4_pipeline.run(sev)

        json_str = res.to_json(indent=2)
        parsed = json.loads(json_str)

        self.assertIn("metadata", parsed)
        self.assertIn("output_a_inspection_review_queue", parsed)
        self.assertIn("output_b_engineering_maintenance_priority", parsed)
        self.assertIn("audit_trail", parsed)

        q = parsed["output_a_inspection_review_queue"]
        self.assertEqual(q["queue_summary"]["total_findings_count"], 2)
        self.assertEqual(q["queue_summary"]["structural_findings_count"], 1)
        self.assertEqual(q["queue_summary"]["non_structural_findings_count"], 1)

        m = parsed["output_b_engineering_maintenance_priority"]
        self.assertEqual(m["priority_decision"]["status"], "NOT_ASSESSED")


if __name__ == "__main__":
    unittest.main()
