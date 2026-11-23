"""
Package: src.measurement
Description:
    Stage 2: Defect Measurement and Severity Assessment Foundation.
"""

from src.measurement.geometry import (
    analyze_connected_components,
    compute_confidence_stats,
    compute_morphological_skeleton,
    extract_shape_features,
)
from src.measurement.pipeline import DefectMeasurementPipeline
from src.measurement.schema import (
    CALIBRATION_DISCLAIMER,
    SEVERITY_VALIDATION_DISCLAIMER,
    BoundingBox,
    ClassMeasurementSummary,
    ConfidenceStats,
    DefectComponentMeasurement,
    ImageMeasurementResult,
    ShapeFeatures,
)
from src.measurement.severity import (
    DACL10K_TIER_MAPPING,
    BaseSeverityEvaluator,
    LiteratureGroundedSeverityEvaluator,
    SeverityAssessmentResult,
    StructuralCriticalityTier,
)
from src.measurement.visualization import DefectVisualizer

__all__ = [
    "DefectMeasurementPipeline",
    "DefectVisualizer",
    "BaseSeverityEvaluator",
    "LiteratureGroundedSeverityEvaluator",
    "StructuralCriticalityTier",
    "DACL10K_TIER_MAPPING",
    "SeverityAssessmentResult",
    "BoundingBox",
    "ConfidenceStats",
    "ShapeFeatures",
    "DefectComponentMeasurement",
    "ClassMeasurementSummary",
    "ImageMeasurementResult",
    "CALIBRATION_DISCLAIMER",
    "SEVERITY_VALIDATION_DISCLAIMER",
    "compute_morphological_skeleton",
    "compute_confidence_stats",
    "extract_shape_features",
    "analyze_connected_components",
]
