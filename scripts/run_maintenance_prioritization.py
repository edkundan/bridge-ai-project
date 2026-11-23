#!/usr/bin/env python3
"""
Script: scripts/run_maintenance_prioritization.py
Description:
    Runs Stage 4 Explainable, Evidence-Aware Maintenance Prioritization Foundation.
    Takes Stage 3 JSON (or runs end-to-end on image), parses optional verified
    engineering context, applies the decision policy, and outputs:
    1. Output A: Inspection Review Queue (provisional human verification ranking)
    2. Output B: Engineering Maintenance Priority (NOT_ASSESSED / strict abstention)

Usage:
    python scripts/run_maintenance_prioritization.py \
        --stage3-json outputs/stage3_demo/graffiti_spalling_case_severity.json \
        --output-dir outputs/stage4_prioritization
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.prioritization import (
    MaintenancePrioritizationPipeline,
    ProductionAbstentionPolicy,
    ResearchDemonstrationPolicy,
    VerifiedEngineeringContext,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_maintenance_prioritization")


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 4 Explainable Maintenance Prioritization Foundation."
    )
    parser.add_argument(
        "--stage3-json",
        type=str,
        default=None,
        help="Path to Stage 3 severity assessment JSON output.",
    )
    parser.add_argument(
        "--engineering-context",
        type=str,
        default=None,
        help="Optional path to verified engineering context JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stage4_prioritization",
        help="Directory to save Stage 4 prioritization outputs.",
    )
    parser.add_argument(
        "--demo-policy",
        action="store_true",
        help="Enable ResearchDemonstrationPolicy (simulation mode only; clearly flagged).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Stage 4 Pipeline
    pipeline = MaintenancePrioritizationPipeline(use_research_demo=args.demo_policy)

    stage3_path = args.stage3_json
    if not stage3_path or not os.path.exists(stage3_path):
        # Default fallback to existing stage 3 demo outputs
        candidate_paths = [
            "outputs/stage3_demo/graffiti_spalling_case_severity.json",
            "outputs/stage3_demo/synthetic_demo_image_severity.json",
        ]
        for cp in candidate_paths:
            if os.path.exists(cp):
                stage3_path = cp
                break

    if not stage3_path or not os.path.exists(stage3_path):
        logger.error(
            "No Stage 3 JSON file found. Run scripts/run_defect_measurement.py first "
            "or provide --stage3-json <path>."
        )
        sys.exit(1)

    logger.info(f"Loading Stage 3 Report: {stage3_path}")
    with open(stage3_path, "r") as f:
        stage3_data = json.load(f)

    # Optional verified engineering context
    ctx = None
    if args.engineering_context and os.path.exists(args.engineering_context):
        logger.info(f"Loading Verified Engineering Context: {args.engineering_context}")
        with open(args.engineering_context, "r") as f:
            ctx_dict = json.load(f)
        ctx = pipeline._parse_engineering_context(ctx_dict)

    # Run prioritization pipeline
    result = pipeline.run(stage3_data, engineering_context=ctx)
    saved_paths = pipeline.save_results(result, out_dir)

    queue = result.inspection_review_queue
    maint = result.engineering_maintenance_priority

    logger.info("=" * 72)
    logger.info(f"STAGE 4 PRIORITIZATION RESULTS: {result.image_id}")
    logger.info("=" * 72)
    logger.info(f"Policy Applied: {result.policy_name} ({result.policy_type})")
    logger.info("-" * 72)
    logger.info("OUTPUT A: INSPECTION REVIEW QUEUE (Provisional Field Verification)")
    logger.info(f"  Total Findings in Queue: {queue.total_findings_count}")
    logger.info(f"  Structural Findings: {queue.structural_findings_count}")
    logger.info(f"  Non-Structural Findings: {queue.non_structural_findings_count}")
    logger.info(f"  Highest Queue Urgency: {queue.highest_queue_urgency}")
    for item in queue.queue_items:
        logger.info(
            f"    Rank #{item.rank}: [{item.finding_id}] {item.class_name:<14} | "
            f"Urgency: {item.queue_urgency.value:<28} | Score: {item.composite_verification_score:.4f} | "
            f"Uncertainty: {item.optical_uncertainty_level}"
        )
    logger.info("-" * 72)
    logger.info("OUTPUT B: ENGINEERING MAINTENANCE PRIORITY (Statutory Work Orders)")
    logger.info(f"  Priority Status: {maint.status.value}")
    logger.info(f"  Official Repair Priority: {maint.official_repair_priority}")
    logger.info(f"  Official Repair Deadline: {maint.official_repair_deadline}")
    logger.info(f"  Official AASHTO State:    {maint.official_aashto_condition_state}")
    logger.info(f"  Abstention Justification: {maint.abstention_reason[:80]}...")
    logger.info(f"  Missing Prerequisites:    {len(maint.missing_engineering_prerequisites)} items")
    if maint.contradiction_warnings:
        for w in maint.contradiction_warnings:
            logger.warning(f"  {w}")
    logger.info("=" * 72)
    logger.info(f"Saved Full Stage 4 JSON:   {saved_paths['stage4_full']}")
    logger.info(f"Saved Review Queue JSON:   {saved_paths['review_queue']}")
    logger.info(f"Saved Maint Priority JSON: {saved_paths['maintenance_priority']}")


if __name__ == "__main__":
    main()
