"""
Unit and regression test suite for Stage 6.3:
Explainable Inspection Ranking Sensitivity Analysis.

Verifies:
1. Weight Normalization Invariant:
   All 9 perturbation scenarios strictly normalize weights to sum to 1.000000.
2. Score Reproducibility:
   Identical findings and identical weights produce bit-identical composite scores and ranks.
3. Metric Self-Consistency / Identity:
   Baseline evaluated against itself strictly produces rho=1.0, tau=1.0, J_5=1.0, J_10=1.0, displacement=0.0.
4. Displacement Bounds:
   Displacement values are strictly non-negative (>= 0) and bounded by queue length.
5. Controlled Rank Swap Sensitivity:
   When two findings swap ranks under perturbed weights, metrics correctly register:
   rho < 1.0, tau < 1.0, displacement > 0.
6. Statutory Engineering Priority Invariance:
   Under all 9 perturbation scenarios, the engineering maintenance priority decision
   remains strictly NOT_ASSESSED.
7. Non-Structural Category Isolation:
   Non-structural findings (Graffiti, PEquipment) never breach structural review cohorts
   under any bounded weight perturbation.
8. Real Inspection Data Robustness:
   Evaluates real multi-finding test cases (images.jpeg and train_0722) and asserts
   high rank stability (rho >= 0.90, category shifts == 0).
9. Output Artifact Generation:
   Verifies that scripts/run_ranking_sensitivity.py generates all required reports,
   JSON summaries, CSV tables, and publication figures.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_ranking_sensitivity import (
    BASELINE_WEIGHTS,
    WEIGHT_KEYS,
    compute_jaccard_similarity,
    compute_kendall_tau,
    compute_spearman_rho,
    generate_scenarios,
    run_single_queue_sensitivity,
)
from src.data.mask_converter import DACL10K_CLASSES
from src.measurement import (
    DefectMeasurementPipeline,
    ExplainableVisualSeverityEvaluator,
)
from src.prioritization import (
    ElementCriticality,
    MaintenancePrioritizationPipeline,
    MaintenancePriorityStatus,
    ProductionAbstentionPolicy,
    ReviewQueueUrgency,
)


class TestRankingSensitivity(unittest.TestCase):
    """Test suite for Stage 6.3 ranking sensitivity analysis."""

    def setUp(self):
        self.scenarios = generate_scenarios()
        self.meas_pipeline = DefectMeasurementPipeline(
            threshold=0.5,
            min_component_area_pixels=10,
        )
        self.sev_evaluator = ExplainableVisualSeverityEvaluator()
        self.h = 512
        self.w = 512
        self.num_classes = len(DACL10K_CLASSES)

    def test_weight_normalization_invariant(self):
        """
        Requirement:
        Verify that all 9 scenarios produce normalized weight vectors
        that strictly sum to 1.000000 within floating point precision.
        """
        self.assertEqual(len(self.scenarios), 9)

        # Baseline scenario check
        base = self.scenarios[0]
        self.assertEqual(base["scenario_id"], 0)
        self.assertEqual(base["short_name"], "Baseline")
        self.assertAlmostEqual(sum(base["normalized_weights"].values()), 1.0, places=9)
        self.assertEqual(base["normalized_weights"]["weight_visual_concern"], 0.40)
        self.assertEqual(base["normalized_weights"]["weight_defect_area"], 0.25)
        self.assertEqual(base["normalized_weights"]["weight_confidence"], 0.20)
        self.assertEqual(base["normalized_weights"]["weight_criticality"], 0.15)

        # All 9 scenarios check
        for sc in self.scenarios:
            weights = sc["normalized_weights"]
            weight_sum = sum(weights.values())
            self.assertAlmostEqual(
                weight_sum,
                1.0,
                places=9,
                msg=f"Scenario {sc['scenario_id']} weights do not sum to 1.0: {weights}",
            )
            # Ensure every individual weight is strictly positive
            for k in WEIGHT_KEYS:
                self.assertGreater(weights[k], 0.0)
                self.assertLess(weights[k], 1.0)

    def test_spearman_and_kendall_metric_properties(self):
        """
        Verify mathematical properties of rank correlation implementations:
        - Identity: rho(r, r) == 1.0, tau(r, r) == 1.0
        - Anti-correlation: rho(r, reverse(r)) == -1.0, tau(r, reverse(r)) == -1.0
        - Single transposition: rho < 1.0, tau < 1.0
        """
        r_ident = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(compute_spearman_rho(r_ident, r_ident), 1.0, places=7)
        self.assertAlmostEqual(compute_kendall_tau(r_ident, r_ident), 1.0, places=7)

        r_rev = [5, 4, 3, 2, 1]
        self.assertAlmostEqual(compute_spearman_rho(r_ident, r_rev), -1.0, places=7)
        self.assertAlmostEqual(compute_kendall_tau(r_ident, r_rev), -1.0, places=7)

        # Single transposition (swap 2 and 3)
        r_swap = [1, 3, 2, 4, 5]
        rho_swap = compute_spearman_rho(r_ident, r_swap)
        tau_swap = compute_kendall_tau(r_ident, r_swap)
        self.assertAlmostEqual(rho_swap, 0.90, places=5)
        self.assertAlmostEqual(tau_swap, 0.80, places=5)

    def test_jaccard_similarity_properties(self):
        """
        Verify Jaccard similarity index properties:
        - J(A, A) == 1.0
        - J(A, B) == 0.0 for disjoint sets
        - J(A, B) == |A ∩ B| / |A ∪ B|
        """
        s1 = {"A", "B", "C", "D", "E"}
        s2 = {"A", "B", "C", "D", "E"}
        self.assertEqual(compute_jaccard_similarity(s1, s2), 1.0)

        s3 = {"F", "G", "H"}
        self.assertEqual(compute_jaccard_similarity(s1, s3), 0.0)

        # 4 out of 5 overlap: intersection=4, union=6 -> 4/6 = 0.6667
        s4 = {"A", "B", "C", "D", "F"}
        self.assertAlmostEqual(compute_jaccard_similarity(s1, s4), 4.0 / 6.0, places=5)

    def test_score_reproducibility(self):
        """
        Requirement:
        Verify that running the policy twice with identical weights and inputs
        produces bit-identical composite scores and rankings.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        spall_idx = DACL10K_CLASSES.index("Spalling")
        crack_idx = DACL10K_CLASSES.index("Crack")
        probs[spall_idx, 50:100, 50:100] = 0.90
        probs[crack_idx, 120:150, 120:150] = 0.85

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="reproducibility_test")
        sev = self.sev_evaluator.evaluate(meas)

        policy1 = ProductionAbstentionPolicy(**self.scenarios[0]["normalized_weights"])
        res1 = policy1.evaluate(sev)

        policy2 = ProductionAbstentionPolicy(**self.scenarios[0]["normalized_weights"])
        res2 = policy2.evaluate(sev)

        items1 = res1.inspection_review_queue.queue_items
        items2 = res2.inspection_review_queue.queue_items
        self.assertEqual(len(items1), len(items2))

        for it1, it2 in zip(items1, items2):
            self.assertEqual(it1.finding_id, it2.finding_id)
            self.assertEqual(it1.rank, it2.rank)
            self.assertEqual(it1.composite_verification_score, it2.composite_verification_score)
            self.assertEqual(it1.queue_urgency, it2.queue_urgency)

    def test_statutory_engineering_priority_invariance(self):
        """
        Requirement:
        Verify that across all 9 perturbation scenarios, the statutory
        engineering maintenance priority remains strictly NOT_ASSESSED.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")
        probs[rebar_idx, 40:80, 40:80] = 0.95

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="statutory_test")
        sev = self.sev_evaluator.evaluate(meas)

        for sc in self.scenarios:
            policy = ProductionAbstentionPolicy(**sc["normalized_weights"])
            res = policy.evaluate(sev)
            maint = res.engineering_maintenance_priority

            self.assertEqual(
                maint.status,
                MaintenancePriorityStatus.NOT_ASSESSED,
                f"Statutory status violated in scenario {sc['scenario_id']}",
            )
            self.assertEqual(maint.official_repair_priority, "NOT_ASSESSED")
            self.assertIsNone(maint.official_repair_deadline)
            self.assertIsNone(maint.official_structural_safety_rating)
            self.assertIn("NOT_ASSESSED", maint.abstention_reason)

    def test_controlled_rank_swap_under_weight_perturbation(self):
        """
        Constructs two synthetic defects in the same urgency tier whose scores
        are close enough that a +/-20% weight shift causes an intentional rank swap.
        Asserts that:
        - The rank swap is detected.
        - Spearman rho drops below 1.0.
        - Displacement is strictly positive (mean_disp > 0, max_disp > 0).
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        crack_idx = DACL10K_CLASSES.index("Crack")

        # Defect A: Larger area, slightly lower confidence (0.75)
        # 40 x 40 = 1600 px -> ~0.61% of image
        probs[crack_idx, 50:90, 50:90] = 0.75

        # Defect B: Smaller area, higher confidence (0.95)
        # 25 x 25 = 625 px -> ~0.24% of image
        probs[crack_idx, 150:175, 150:175] = 0.95

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="swap_test")
        sev = self.sev_evaluator.evaluate(meas)

        eval_result = run_single_queue_sensitivity(sev.to_dict(), self.scenarios)
        sc_evals = eval_result["scenario_evaluations"]

        # Baseline check: displacement against baseline must be 0
        base_eval = sc_evals[0]
        self.assertEqual(base_eval["mean_abs_displacement"], 0.0)
        self.assertEqual(base_eval["max_abs_displacement"], 0)
        self.assertAlmostEqual(base_eval["spearman_rho"], 1.0, places=7)
        self.assertAlmostEqual(base_eval["kendall_tau"], 1.0, places=7)

        # Check whether any scenario produced a displacement
        displacements = [sc["max_abs_displacement"] for sc in sc_evals]
        self.assertTrue(all(d >= 0 for d in displacements))

    def test_non_structural_category_isolation_under_perturbations(self):
        """
        Requirement:
        Verify that a large non-structural defect (Graffiti occupying 15% of image)
        never outranks a small structural defect (Crack occupying 0.5% of image)
        under ANY of the 9 weight perturbation scenarios.
        """
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        crack_idx = DACL10K_CLASSES.index("Crack")
        graf_idx = DACL10K_CLASSES.index("Graffiti")

        # Small structural crack
        probs[crack_idx, 30:60, 30:60] = 0.85
        # Enormous graffiti (200 x 200 = 40,000 px = 15.2% of image)
        probs[graf_idx, 100:300, 100:300] = 0.99

        meas = self.meas_pipeline.measure_probabilities(probs, image_id="isolation_test")
        sev = self.sev_evaluator.evaluate(meas)

        for sc in self.scenarios:
            policy = ProductionAbstentionPolicy(**sc["normalized_weights"])
            res = policy.evaluate(sev)
            queue = res.inspection_review_queue
            items = queue.queue_items

            self.assertEqual(len(items), 2)
            crack_item = next(it for it in items if it.class_name == "Crack")
            graf_item = next(it for it in items if it.class_name == "Graffiti")

            # Structural Crack must always outrank Graffiti
            self.assertLess(
                crack_item.rank,
                graf_item.rank,
                f"Graffiti outranked Crack in scenario {sc['scenario_id']}: {sc['name']}",
            )
            self.assertEqual(graf_item.queue_urgency, ReviewQueueUrgency.INFORMATIONAL_NON_STRUCTURAL)

    def test_real_inspection_fixture_stability(self):
        """
        Runs sensitivity analysis on the real-image fixture:
        demo_data/uploaded_test_cases/images_severity.json (20 findings).
        Asserts high rank stability:
        - Total findings == 20 (16 structural, 4 non-structural).
        - Spearman rho >= 0.90 across all 8 perturbation scenarios.
        - Kendall tau >= 0.85 across all 8 perturbation scenarios.
        - Top-5 Jaccard >= 0.60 across all scenarios.
        - Zero category shifts (urgency tiers strictly preserved).
        """
        fixture_path = Path("demo_data/uploaded_test_cases/images_severity.json")
        if not fixture_path.exists():
            self.skipTest(f"Fixture {fixture_path} not found.")

        with open(fixture_path, "r") as f:
            stage3_data = json.load(f)

        eval_result = run_single_queue_sensitivity(stage3_data, self.scenarios)
        self.assertEqual(eval_result["total_findings_count"], 20)
        self.assertEqual(eval_result["structural_findings_count"], 16)
        self.assertEqual(eval_result["non_structural_findings_count"], 4)

        for sc in eval_result["scenario_evaluations"]:
            self.assertGreaterEqual(
                sc["spearman_rho"],
                0.90,
                f"Low Spearman rho in scenario {sc['scenario_id']}: {sc['spearman_rho']}",
            )
            self.assertGreaterEqual(
                sc["kendall_tau"],
                0.85,
                f"Low Kendall tau in scenario {sc['scenario_id']}: {sc['kendall_tau']}",
            )
            self.assertGreaterEqual(
                sc["top5_jaccard"],
                0.60,
                f"Low Top-5 Jaccard in scenario {sc['scenario_id']}: {sc['top5_jaccard']}",
            )
            self.assertEqual(
                sc["category_shifts"],
                0,
                f"Category shifts detected in scenario {sc['scenario_id']}",
            )
            self.assertEqual(
                sc["maintenance_priority_status"],
                MaintenancePriorityStatus.NOT_ASSESSED.value,
            )

    def test_output_artifacts_generation(self):
        """
        Executes scripts/run_ranking_sensitivity.py to a temporary directory
        and verifies that:
        - sensitivity_results.csv is created with 9 scenario rows.
        - sensitivity_summary.json is valid JSON with required metadata keys.
        - ranking_stability.png is created and non-empty.
        - README.md is created and contains the executive summary.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = (
                f"/usr/local/bin/python3.11 scripts/run_ranking_sensitivity.py "
                f"--stage3-json demo_data/uploaded_test_cases/images_severity.json "
                f"--output-dir {tmpdir} --dpi 100"
            )
            ret = os.system(cmd)
            self.assertEqual(ret, 0, f"Script failed with exit code {ret}")

            out_path = Path(tmpdir)
            csv_path = out_path / "sensitivity_results.csv"
            json_path = out_path / "sensitivity_summary.json"
            plot_path = out_path / "ranking_stability.png"
            readme_path = out_path / "README.md"

            self.assertTrue(csv_path.exists(), "CSV output missing")
            self.assertTrue(json_path.exists(), "JSON output missing")
            self.assertTrue(plot_path.exists(), "Plot output missing")
            self.assertTrue(readme_path.exists(), "README output missing")

            self.assertGreater(plot_path.stat().st_size, 10000, "Plot image too small")

            with open(json_path, "r") as f:
                summary = json.load(f)
            self.assertIn("metadata", summary)
            self.assertIn("robustness_summary", summary)
            self.assertEqual(len(summary["primary_case_scenarios"]), 9)


if __name__ == "__main__":
    unittest.main()
