"""
Package: src.measurement
Description:
    Stage 2: Defect Measurement Foundation
    Stage 3: Explainable Visual Severity Assessment Foundation
"""

from src.measurement.geometry import (
    analyze_connected_components,
    compute_confidence_stats,
    compute_morphological_skeleton,
    extract_shape_features,
)
from src.measurement.pipeline import DefectMeasurementPipeline
from src.measurement.schema import (
    ABSTENTION_JUSTIFICATION,
    CALIBRATION_DISCLAIMER,
    CONFIDENCE_DISTINCTION_NOTICE,
    NON_STRUCTURAL_CLASSES,
    SEVERITY_VALIDATION_DISCLAIMER,
    BoundingBox,
    ClassMeasurementSummary,
    ConfidenceStats,
    DefectComponentMeasurement,
    DefectVisualObservation,
    ExplainableSeverityReport,
    ImageMeasurementResult,
    MissingDataChecklist,
    ReviewFlag,
    ShapeFeatures,
    StructuralCriticalityTier,
    VisualConcernLevel,
)
from src.measurement.severity import (
    CLASS_VISUAL_CONCERN_MAPPING,
    DACL10K_TIER_MAPPING,
    DEFECT_KNOWLEDGE_BASE,
    BaseSeverityEvaluator,
    ExplainableVisualSeverityEvaluator,
    LiteratureGroundedSeverityEvaluator,
    SeverityAssessmentResult,
)
from src.measurement.visualization import DefectVisualizer

__all__ = [
    # Pipelines & Evaluators
    "DefectMeasurementPipeline",
    "DefectVisualizer",
    "BaseSeverityEvaluator",
    "ExplainableVisualSeverityEvaluator",
    "LiteratureGroundedSeverityEvaluator",
    "SeverityAssessmentResult",
    # Enums
    "VisualConcernLevel",
    "StructuralCriticalityTier",
    "ReviewFlag",
    # Constants & Mappings
    "NON_STRUCTURAL_CLASSES",
    "CLASS_VISUAL_CONCERN_MAPPING",
    "DACL10K_TIER_MAPPING",
    "DEFECT_KNOWLEDGE_BASE",
    "CALIBRATION_DISCLAIMER",
    "SEVERITY_VALIDATION_DISCLAIMER",
    "ABSTENTION_JUSTIFICATION",
    "CONFIDENCE_DISTINCTION_NOTICE",
    # Schemas & Dataclasses
    "BoundingBox",
    "ConfidenceStats",
    "ShapeFeatures",
    "DefectComponentMeasurement",
    "ClassMeasurementSummary",
    "ImageMeasurementResult",
    "MissingDataChecklist",
    "DefectVisualObservation",
    "ExplainableSeverityReport",
    # Geometric Functions
    "compute_morphological_skeleton",
    "compute_confidence_stats",
    "extract_shape_features",
    "analyze_connected_components",
]
