"""
Module: src.dashboard
Description:
    Interactive Explainable AI Bridge Inspection Dashboard (Stage 5).
    Provides visual triage, defect quantification, rule-based explainability,
    inspection review queue management, and statutory maintenance abstention.
"""

from src.dashboard.data_loader import (
    AVAILABLE_DEMO_CASES,
    load_saved_demo_case,
    run_live_pipeline,
)

# Alias for convenience
run_live_inference = run_live_pipeline

__all__ = [
    "AVAILABLE_DEMO_CASES",
    "load_saved_demo_case",
    "run_live_pipeline",
    "run_live_inference",
]
