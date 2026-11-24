"""
Module: src.measurement.schema
Description:
    Standard dataclasses, types, and JSON serialization schemas for:
    - Stage 2: Defect Measurement Foundation
    - Stage 3: Explainable Visual Severity Assessment Foundation

    Key Design Principles:
    1. Distinguishes summed per-class pixel coverage from union pixel coverage.
    2. Segregates structural defect evidence from non-structural surface features (e.g. Graffiti).
    3. Strictly decouples visual concern levels from certified structural condition states.
    4. Enforces explicit abstention (NOT_ASSESSED) when structural context is missing.
    5. Provides structured review flags and per-finding XAI explanations for inspector verification.
    6. Maintains backward compatibility with Stage 2 consumers and test suites.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

CALIBRATION_DISCLAIMER = (
    "OPTICAL MEASUREMENT NOTICE: All lengths, widths, perimeters, and areas "
    "are reported strictly in pixel-domain units (px, px^2) or image/ROI-relative "
    "dimensionless percentages (%). Physical metric measurements (mm, cm, m) "
    "cannot be claimed without metric camera calibration, sensor distance measurements, "
    "or physical reference scale markers in the image plane."
)

SEVERITY_VALIDATION_DISCLAIMER = (
    "ENGINEERING SEVERITY DISCLAIMER: This severity assessment is an uncalibrated "
    "research prototype grounded in literature principles (AASHTO MBEI, FHWA BIRM, "
    "DIN 1076, FIB Model Code). It provides visual evidence categorization and "
    "review flagging only and does NOT constitute a certified structural safety rating, "
    "official AASHTO element condition state (CS1-CS4), remaining service life prediction, "
    "or replacement for certified structural bridge engineer inspection."
)

ABSTENTION_JUSTIFICATION = (
    "OFFICIAL CONDITION STATE ABSTENTION: An official bridge condition state (e.g. "
    "AASHTO MBEI CS1-CS4 or NBIS Item 58-60 ratings) CANNOT be established from 2D optical "
    "imagery alone. Grounded structural assessment requires unobservable parameters: "
    "structural component identity, primary/secondary load path importance, defect depth, "
    "rebar section loss, and physical metric calibration. Consequently, the official "
    "structural condition state is marked NOT_ASSESSED."
)

CONFIDENCE_DISTINCTION_NOTICE = (
    "CONFIDENCE CONTRACT: The reported confidence score is the mean pixel-level "
    "classification probability produced by the neural network segmentation model. "
    "This indicates optical pattern clarity, NOT calibrated engineering confidence "
    "in structural severity or safety risk."
)

NON_STRUCTURAL_CLASSES = {
    "Graffiti", "Drainage", "JTape", "PEquipment", "Restformwork"
}


# ============================================================================
# Enums for Stage 3 Explainable Severity Assessment
# ============================================================================

class VisualConcernLevel(str, Enum):
    """
    Visual concern level based strictly on observable defect morphology and material state.
    This is an OBSERVATION-LEVEL concern category, NOT an official bridge condition state.
    """
    HIGH_VISUAL_CONCERN = "HIGH_VISUAL_CONCERN"
    """Direct visual evidence of concrete cover breach, reinforcement exposure,
    rockpocket voiding, or bridge articulation distress (ExposedRebars, Spalling,
    Rockpocket, Bearing, EJoint)."""

    MODERATE_VISUAL_CONCERN = "MODERATE_VISUAL_CONCERN"
    """Linear cracking or material degradation pathways that may facilitate moisture
    or chloride ingress (Crack, ACrack, WConccor)."""

    LOW_VISUAL_CONCERN = "LOW_VISUAL_CONCERN"
    """Surface-level moisture, weathering, or chemical leaching without apparent
    structural section loss (Efflorescence, Rust, Weathering, Wetspot, Cavity, Hollowareas)."""

    NEGLIGIBLE_NON_STRUCTURAL = "NEGLIGIBLE_NON_STRUCTURAL"
    """Non-structural appurtenances, formwork remnants, or aesthetic surface markings
    (Graffiti, Drainage, JTape, PEquipment, Restformwork)."""

    NO_DEFECTS_DETECTED = "NO_DEFECTS_DETECTED"
    """No defect classes detected above active probability thresholds."""


class StructuralCriticalityTier(str, Enum):
    """
    [DEPRECATED - UNVALIDATED HEURISTIC]
    Legacy Stage 2 tier classification preserved solely for backward-compatible deserialization.
    These heuristic tiers (such as TIER_1_CRITICAL) assign criticality based solely on pixel presence
    without physical depth or load path verification, and must NOT be exposed as a validated
    structural condition rating or maintenance priority.
    """
    TIER_1_CRITICAL = "TIER_1_CRITICAL"
    TIER_2_PROGRESSIVE = "TIER_2_PROGRESSIVE"
    TIER_3_SUPERFICIAL = "TIER_3_SUPERFICIAL"
    TIER_4_NON_STRUCTURAL = "TIER_4_NON_STRUCTURAL"


class ReviewFlag(str, Enum):
    """
    Structured flags indicating observations that warrant human-in-the-loop verification
    without asserting structural failure or ungrounded urgency.
    """
    POSSIBLE_SPALLING_REVIEW = "POSSIBLE_SPALLING_REVIEW"
    """Visual spalling detected. Human verification needed: sounding for delamination
    and checking for concealed rebar corrosion."""

    REBAR_EXPOSURE_REVIEW = "REBAR_EXPOSURE_REVIEW"
    """Exposed reinforcing steel detected. Human verification needed: physical measurement
    of bar diameter loss and corrosion pitting depth."""

    CRACK_DELAMINATION_REVIEW = "CRACK_DELAMINATION_REVIEW"
    """Linear or alligator cracking detected. Human verification needed: optical comparator
    measurement of metric crack width against the 0.3mm durability limit."""

    BEARING_ARTICULATION_REVIEW = "BEARING_ARTICULATION_REVIEW"
    """Bearing component detected. Human verification needed: alignment, elastomeric
    bulging, anchor bolt integrity, and debris accumulation."""

    EXPANSION_JOINT_REVIEW = "EXPANSION_JOINT_REVIEW"
    """Expansion joint detected. Human verification needed: watertightness, seal integrity,
    and thermal movement clearance."""

    HONEYCOMBING_CONCRETE_REVIEW = "HONEYCOMBING_CONCRETE_REVIEW"
    """Rockpocket or weak concrete detected. Human verification needed: void depth,
    aggregate segregation, and compaction quality."""

    SURFACE_MOISTURE_LEACHING_REVIEW = "SURFACE_MOISTURE_LEACHING_REVIEW"
    """Efflorescence, rust staining, or wetspots detected. Human verification needed:
    tracing water ponding or drainage leakage pathways."""

    NON_STRUCTURAL_SURFACE_COVERAGE = "NON_STRUCTURAL_SURFACE_COVERAGE"
    """Extensive non-structural markings (e.g. Graffiti) detected. Noted to prevent
    non-structural coverage from skewing maintenance prioritization."""


# ============================================================================
# Basic Geometry & Feature Dataclasses
# ============================================================================

@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box coordinates.
    Provides [xmin, ymin, xmax, ymax], [ymin, xmin, ymax, xmax], and [xmin, ymin, width, height].
    """
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def xyxy(self) -> List[int]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def yxyx(self) -> List[int]:
        return [self.ymin, self.xmin, self.ymax, self.xmax]

    @property
    def xywh(self) -> List[int]:
        return [self.xmin, self.ymin, self.width, self.height]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "width": self.width,
            "height": self.height,
            "xyxy": self.xyxy,
            "xywh": self.xywh,
            "bbox_area_pixels": self.area,
        }


@dataclass
class ConfidenceStats:
    """
    Statistical distribution of neural network segmentation probabilities over defect pixels.
    NOTE: Represents model prediction probability, NOT structural severity confidence.
    """
    mean: float
    min: float
    max: float
    median: float
    std: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": round(float(self.mean), 4),
            "min": round(float(self.min), 4),
            "max": round(float(self.max), 4),
            "median": round(float(self.median), 4),
            "std": round(float(self.std), 4),
        }


@dataclass
class ShapeFeatures:
    """
    Pixel-domain shape and morphology descriptors.
    """
    perimeter_pixels: float
    circularity: float
    aspect_ratio: float
    solidity: float
    extent: float
    orientation_deg: float
    is_linear_defect: bool = False
    skeleton_length_pixels: Optional[int] = None
    estimated_avg_width_pixels: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perimeter_pixels": round(float(self.perimeter_pixels), 2),
            "circularity": round(float(self.circularity), 4),
            "aspect_ratio": round(float(self.aspect_ratio), 4),
            "solidity": round(float(self.solidity), 4),
            "extent": round(float(self.extent), 4),
            "orientation_deg": round(float(self.orientation_deg), 2),
            "is_linear_defect": self.is_linear_defect,
            "skeleton_length_pixels": self.skeleton_length_pixels,
            "estimated_avg_width_pixels": (
                round(float(self.estimated_avg_width_pixels), 2)
                if self.estimated_avg_width_pixels is not None
                else None
            ),
        }


@dataclass
class DefectComponentMeasurement:
    """
    Detailed measurement of a single connected component of a defect class.
    """
    component_id: int
    class_id: int
    class_name: str
    pixel_area: int
    image_relative_area_pct: float
    centroid_xy: Tuple[float, float]
    bbox: BoundingBox
    confidence: ConfidenceStats
    shape_features: ShapeFeatures
    roi_relative_area_pct: Optional[float] = None
    pixels_inside_roi: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "pixel_area": self.pixel_area,
            "image_relative_area_pct": round(float(self.image_relative_area_pct), 5),
            "roi_relative_area_pct": (
                round(float(self.roi_relative_area_pct), 5)
                if self.roi_relative_area_pct is not None
                else None
            ),
            "pixels_inside_roi": self.pixels_inside_roi,
            "centroid_xy": [round(float(self.centroid_xy[0]), 2), round(float(self.centroid_xy[1]), 2)],
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence.to_dict(),
            "shape_features": self.shape_features.to_dict(),
        }


@dataclass
class ClassMeasurementSummary:
    """
    Summary of defect measurements for a specific class across all its components.
    """
    class_id: int
    class_name: str
    detected: bool
    component_count: int
    total_pixel_area: int
    image_relative_area_pct: float
    confidence: Optional[ConfidenceStats] = None
    roi_relative_area_pct: Optional[float] = None
    total_pixels_inside_roi: Optional[int] = None
    components: List[DefectComponentMeasurement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "detected": self.detected,
            "component_count": self.component_count,
            "total_pixel_area": self.total_pixel_area,
            "image_relative_area_pct": round(float(self.image_relative_area_pct), 5),
            "roi_relative_area_pct": (
                round(float(self.roi_relative_area_pct), 5)
                if self.roi_relative_area_pct is not None
                else None
            ),
            "total_pixels_inside_roi": self.total_pixels_inside_roi,
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "components": [c.to_dict() for c in self.components],
        }


# ============================================================================
# Stage 2 & 3 Image Measurement Result
# ============================================================================

@dataclass
class ImageMeasurementResult:
    """
    Complete measurement result for a single image, encompassing all 19 DACL10K classes.
    Distinguishes summed per-class area from union area, and structural from non-structural defects.
    """
    image_id: str
    image_height: int
    image_width: int
    threshold: Union[float, Dict[str, float]]
    min_component_area_pixels: int
    roi_present: bool
    roi_pixel_area: Optional[int]
    roi_relative_area_pct: Optional[float]
    total_detected_classes: int
    total_components_count: int

    # Area metrics
    summed_class_pixel_area: int
    summed_class_image_area_pct: float
    union_defect_pixel_area: int
    union_defect_image_area_pct: float
    structural_union_pixel_area: int
    structural_union_image_area_pct: float
    non_structural_union_pixel_area: int
    non_structural_union_image_area_pct: float
    overlapping_defect_pixel_area: int
    overlapping_defect_image_area_pct: float

    # ROI relative metrics (None if ROI absent)
    summed_class_roi_pixels: Optional[int]
    union_defect_roi_area_pct: Optional[float]
    structural_union_roi_area_pct: Optional[float]
    non_structural_union_roi_area_pct: Optional[float]

    class_measurements: Dict[str, ClassMeasurementSummary]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    disclaimer: str = CALIBRATION_DISCLAIMER

    # Backward compatibility properties
    @property
    def total_defect_pixel_area(self) -> int:
        """Backward compatibility: returns summed_class_pixel_area."""
        return self.summed_class_pixel_area

    @property
    def total_defect_image_area_pct(self) -> float:
        """Backward compatibility: returns summed_class_image_area_pct."""
        return self.summed_class_image_area_pct

    @property
    def total_defect_roi_area_pct(self) -> Optional[float]:
        """Backward compatibility: returns union_defect_roi_area_pct."""
        return self.union_defect_roi_area_pct

    @property
    def total_image_pixels(self) -> int:
        return self.image_height * self.image_width

    def get_detected_classes(self) -> List[str]:
        return [
            name for name, summary in self.class_measurements.items()
            if summary.detected
        ]

    def get_detected_structural_classes(self) -> List[str]:
        return [
            name for name in self.get_detected_classes()
            if name not in NON_STRUCTURAL_CLASSES
        ]

    def get_detected_non_structural_classes(self) -> List[str]:
        return [
            name for name in self.get_detected_classes()
            if name in NON_STRUCTURAL_CLASSES
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "image_id": self.image_id,
                "image_dimensions": {
                    "height": self.image_height,
                    "width": self.image_width,
                    "total_pixels": self.total_image_pixels,
                },
                "timestamp_utc": self.timestamp,
                "threshold": self.threshold,
                "min_component_area_pixels": self.min_component_area_pixels,
                "roi": {
                    "present": self.roi_present,
                    "roi_pixel_area": self.roi_pixel_area,
                    "roi_relative_area_pct": (
                        round(float(self.roi_relative_area_pct), 5)
                        if self.roi_relative_area_pct is not None
                        else None
                    ),
                },
                "disclaimer": self.disclaimer,
            },
            "summary": {
                "total_detected_classes": self.total_detected_classes,
                "detected_class_names": self.get_detected_classes(),
                "detected_structural_classes": self.get_detected_structural_classes(),
                "detected_non_structural_classes": self.get_detected_non_structural_classes(),
                "total_components_count": self.total_components_count,
                # Backward-compatible fields
                "total_defect_pixel_area": self.summed_class_pixel_area,
                "total_defect_image_area_pct": round(float(self.summed_class_image_area_pct), 5),
                "total_defect_roi_area_pct": (
                    round(float(self.union_defect_roi_area_pct), 5)
                    if self.union_defect_roi_area_pct is not None
                    else None
                ),
                # Rigorous distinction between summed, union, structural, and non-structural areas
                "coverage_metrics": {
                    "summed_class_pixel_area": self.summed_class_pixel_area,
                    "summed_class_image_area_pct": round(float(self.summed_class_image_area_pct), 5),
                    "union_defect_pixel_area": self.union_defect_pixel_area,
                    "union_defect_image_area_pct": round(float(self.union_defect_image_area_pct), 5),
                    "structural_union_pixel_area": self.structural_union_pixel_area,
                    "structural_union_image_area_pct": round(float(self.structural_union_image_area_pct), 5),
                    "non_structural_union_pixel_area": self.non_structural_union_pixel_area,
                    "non_structural_union_image_area_pct": round(float(self.non_structural_union_image_area_pct), 5),
                    "overlapping_defect_pixel_area": self.overlapping_defect_pixel_area,
                    "overlapping_defect_image_area_pct": round(float(self.overlapping_defect_image_area_pct), 5),
                    "roi_coverage": {
                        "roi_available": self.roi_present,
                        "summed_class_roi_pixels": self.summed_class_roi_pixels,
                        "union_defect_roi_area_pct": (
                            round(float(self.union_defect_roi_area_pct), 5)
                            if self.union_defect_roi_area_pct is not None
                            else None
                        ),
                        "structural_union_roi_area_pct": (
                            round(float(self.structural_union_roi_area_pct), 5)
                            if self.structural_union_roi_area_pct is not None
                            else None
                        ),
                        "non_structural_union_roi_area_pct": (
                            round(float(self.non_structural_union_roi_area_pct), 5)
                            if self.non_structural_union_roi_area_pct is not None
                            else None
                        ),
                    },
                },
            },
            "class_measurements": {
                name: summary.to_dict()
                for name, summary in self.class_measurements.items()
            },
            # Top-level direct summary bindings for dashboard and external consumers
            "total_detected_classes": self.total_detected_classes,
            "total_components_count": self.total_components_count,
            "summed_class_pixel_area": self.summed_class_pixel_area,
            "summed_class_image_area_pct": round(float(self.summed_class_image_area_pct), 5),
            "union_defect_pixel_area": self.union_defect_pixel_area,
            "union_defect_image_area_pct": round(float(self.union_defect_image_area_pct), 5),
            "structural_union_pixel_area": self.structural_union_pixel_area,
            "structural_union_image_area_pct": round(float(self.structural_union_image_area_pct), 5),
            "non_structural_union_pixel_area": self.non_structural_union_pixel_area,
            "non_structural_union_image_area_pct": round(float(self.non_structural_union_image_area_pct), 5),
            "overlapping_defect_pixel_area": self.overlapping_defect_pixel_area,
            "overlapping_defect_image_area_pct": round(float(self.overlapping_defect_image_area_pct), 5),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# Stage 3 Explainable Severity Assessment Schemas
# ============================================================================

@dataclass
class MissingDataChecklist:
    """
    Explicit checklist of missing engineering parameters required for
    certified structural condition ratings or maintenance prioritization.
    """
    structural_component_identity: str = "MISSING"
    element_importance: str = "MISSING"
    physical_defect_dimensions: str = "MISSING"
    internal_defect_depth: str = "MISSING"
    reinforcement_section_loss: str = "MISSING"
    inspection_history: str = "MISSING"
    engineering_assessment: str = "MISSING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structural_component_identity": {
                "status": self.structural_component_identity,
                "required_source": "Bridge BIM, element segmentation, or engineering inspection plan.",
                "description": "Specific structural element (e.g. pier column, fracture-critical girder, deck slab).",
            },
            "element_importance": {
                "status": self.element_importance,
                "required_source": "National Bridge Inventory (NBI) / AASHTO Element records.",
                "description": "Primary load-bearing vs. secondary vs. non-structural appurtenance.",
            },
            "physical_defect_dimensions": {
                "status": self.physical_defect_dimensions,
                "required_source": "Optical rangefinder, calibrated stereo rig, or fiducial scale.",
                "description": "Metric crack width (mm), spall surface area (m^2), or rebar span (m).",
            },
            "internal_defect_depth": {
                "status": self.internal_defect_depth,
                "required_source": "Non-destructive testing (GPR, ultrasonic pulse velocity, impact echo).",
                "description": "Subsurface void depth, delamination plane, or crack penetration depth.",
            },
            "reinforcement_section_loss": {
                "status": self.reinforcement_section_loss,
                "required_source": "Physical caliper measurement, ultrasonic thickness gauge, or electrochemistry.",
                "description": "Residual rebar diameter, percentage cross-sectional area loss.",
            },
            "inspection_history": {
                "status": self.inspection_history,
                "required_source": "Historical bridge inspection agency database (e.g. AASHTO BrM).",
                "description": "Defect progression rate, date of first observation, active vs dormant status.",
            },
            "engineering_assessment": {
                "status": self.engineering_assessment,
                "required_source": "Licensed Professional Engineer (PE / SE) field evaluation.",
                "description": "Certified structural capacity determination and maintenance work order sign-off.",
            },
        }


@dataclass
class DefectVisualObservation:
    """
    A single explainable defect finding for XAI dashboard presentation.
    """
    finding_id: str
    class_name: str
    class_id: int
    component_id: int
    visual_concern_level: str
    is_structural: bool
    pixel_area: int
    image_relative_extent_pct: float
    roi_relative_extent_pct: Optional[float]
    bbox: BoundingBox
    confidence_stats: ConfidenceStats
    shape_features: ShapeFeatures
    visual_evidence: str
    explanation: str
    literature_reference: str
    inspector_action_items: List[str]
    triggered_review_flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "component_id": self.component_id,
            "visual_concern_level": self.visual_concern_level,
            "is_structural": self.is_structural,
            "pixel_area": self.pixel_area,
            "image_relative_extent_pct": round(float(self.image_relative_extent_pct), 5),
            "roi_relative_extent_pct": (
                round(float(self.roi_relative_extent_pct), 5)
                if self.roi_relative_extent_pct is not None
                else None
            ),
            "bbox": self.bbox.to_dict(),
            "confidence_stats": self.confidence_stats.to_dict(),
            "shape_features": self.shape_features.to_dict(),
            "visual_evidence": self.visual_evidence,
            "explanation": self.explanation,
            "literature_reference": self.literature_reference,
            "inspector_action_items": self.inspector_action_items,
            "triggered_review_flags": self.triggered_review_flags,
        }


@dataclass
class ExplainableSeverityReport:
    """
    Complete Stage 3 Explainable Visual Severity Assessment Report.
    Decouples visual observations from certified structural risk.
    """
    image_id: str
    visual_concern_level: str
    official_structural_condition_state: str  # Always "NOT_ASSESSED" in pure 2D RGB
    abstention_reason: str
    review_flags: List[str]

    # Coverage breakdown distinguishing structural from non-structural and summed from union
    coverage_breakdown: Dict[str, Any]

    # Confidence contract
    segmentation_model_confidence_mean: Optional[float]
    calibrated_structural_severity_confidence: Optional[float]  # None (abstention)
    confidence_distinction_notice: str = CONFIDENCE_DISTINCTION_NOTICE

    missing_data_checklist: MissingDataChecklist = field(default_factory=MissingDataChecklist)
    observations: List[DefectVisualObservation] = field(default_factory=list)
    stage_4_prioritization_contract: Dict[str, Any] = field(default_factory=dict)

    # Literature references
    literature_grounding: List[str] = field(default_factory=lambda: [
        "AASHTO Manual for Bridge Element Inspection (MBEI, 2019)",
        "FHWA Bridge Inspection Reference Manual (BIRM, 2012)",
        "DIN 1076 / RI-EBW-PRÜF: Bridge Evaluation Guidelines",
        "FIB Model Code 2010 (Concrete Durability and Service Life Design)",
    ])
    disclaimer: str = SEVERITY_VALIDATION_DISCLAIMER

    # Backward compatibility properties matching Stage 2 SeverityAssessmentResult
    @property
    def highest_criticality_tier(self) -> Optional[str]:
        """Backward compatibility: maps VisualConcernLevel to legacy Tier name."""
        mapping = {
            VisualConcernLevel.HIGH_VISUAL_CONCERN.value: StructuralCriticalityTier.TIER_1_CRITICAL.value,
            VisualConcernLevel.MODERATE_VISUAL_CONCERN.value: StructuralCriticalityTier.TIER_2_PROGRESSIVE.value,
            VisualConcernLevel.LOW_VISUAL_CONCERN.value: StructuralCriticalityTier.TIER_3_SUPERFICIAL.value,
            VisualConcernLevel.NEGLIGIBLE_NON_STRUCTURAL.value: StructuralCriticalityTier.TIER_4_NON_STRUCTURAL.value,
            VisualConcernLevel.NO_DEFECTS_DETECTED.value: None,
        }
        return mapping.get(self.visual_concern_level, None)

    @property
    def detected_tiers(self) -> List[str]:
        """Backward compatibility: returns list of legacy tier names."""
        tiers = []
        for obs in self.observations:
            legacy_t = {
                VisualConcernLevel.HIGH_VISUAL_CONCERN.value: StructuralCriticalityTier.TIER_1_CRITICAL.value,
                VisualConcernLevel.MODERATE_VISUAL_CONCERN.value: StructuralCriticalityTier.TIER_2_PROGRESSIVE.value,
                VisualConcernLevel.LOW_VISUAL_CONCERN.value: StructuralCriticalityTier.TIER_3_SUPERFICIAL.value,
                VisualConcernLevel.NEGLIGIBLE_NON_STRUCTURAL.value: StructuralCriticalityTier.TIER_4_NON_STRUCTURAL.value,
            }.get(obs.visual_concern_level)
            if legacy_t and legacy_t not in tiers:
                tiers.append(legacy_t)
        return tiers

    @property
    def tier_summary(self) -> Dict[str, Dict[str, Any]]:
        """Backward compatibility: returns legacy tier_summary dictionary."""
        summary = {
            t.value: {
                "classes_detected": [],
                "component_count": 0,
                "total_pixel_area": 0,
                "image_relative_area_pct": 0.0,
            }
            for t in StructuralCriticalityTier
        }
        for obs in self.observations:
            legacy_t = {
                VisualConcernLevel.HIGH_VISUAL_CONCERN.value: StructuralCriticalityTier.TIER_1_CRITICAL.value,
                VisualConcernLevel.MODERATE_VISUAL_CONCERN.value: StructuralCriticalityTier.TIER_2_PROGRESSIVE.value,
                VisualConcernLevel.LOW_VISUAL_CONCERN.value: StructuralCriticalityTier.TIER_3_SUPERFICIAL.value,
                VisualConcernLevel.NEGLIGIBLE_NON_STRUCTURAL.value: StructuralCriticalityTier.TIER_4_NON_STRUCTURAL.value,
            }.get(obs.visual_concern_level)
            if legacy_t:
                if obs.class_name not in summary[legacy_t]["classes_detected"]:
                    summary[legacy_t]["classes_detected"].append(obs.class_name)
                summary[legacy_t]["component_count"] += 1
                summary[legacy_t]["total_pixel_area"] += obs.pixel_area
                summary[legacy_t]["image_relative_area_pct"] += obs.image_relative_extent_pct

        for t in summary:
            summary[t]["image_relative_area_pct"] = round(summary[t]["image_relative_area_pct"], 5)
        return summary

    @property
    def total_defect_image_area_pct(self) -> float:
        """Backward compatibility: returns union_defect_image_area_pct."""
        return float(self.coverage_breakdown.get("union_defect_image_area_pct", 0.0))

    @property
    def total_defect_roi_area_pct(self) -> Optional[float]:
        """Backward compatibility: returns union_defect_roi_area_pct."""
        return self.coverage_breakdown.get("roi_coverage", {}).get("union_defect_roi_area_pct")

    @property
    def is_calibrated(self) -> bool:
        return False

    @property
    def requires_expert_validation(self) -> bool:
        return True

    @property
    def missing_engineering_parameters(self) -> List[str]:
        return list(self.missing_data_checklist.to_dict().keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            # Top-level backward compatibility fields for Stage 2 consumers (Explicitly Deprecated)
            "provisional_criticality_tier": self.highest_criticality_tier,
            "detected_tiers": self.detected_tiers,
            "tier_summary": self.tier_summary,
            "legacy_tier_deprecation": {
                "is_deprecated": True,
                "warning": "TIER_1_CRITICAL and legacy tier heuristics are deprecated and unvalidated. Do not use for engineering work orders or structural ratings.",
                "provisional_criticality_tier": self.highest_criticality_tier,
            },
            "total_defect_image_area_pct": self.total_defect_image_area_pct,
            "total_defect_roi_area_pct": self.total_defect_roi_area_pct,
            "validation_status": {
                "is_calibrated": self.is_calibrated,
                "requires_expert_validation": self.requires_expert_validation,
                "missing_engineering_parameters": self.missing_engineering_parameters,
                "literature_grounding": self.literature_grounding,
            },
            # Stage 3 Explainable Visual Severity Assessment
            "visual_severity_assessment": {
                "visual_concern_level": self.visual_concern_level,
                "official_structural_condition_state": self.official_structural_condition_state,
                "abstention_reason": self.abstention_reason,
                "review_flags": self.review_flags,
            },
            "coverage_breakdown": self.coverage_breakdown,
            "confidence_contract": {
                "segmentation_model_confidence_mean": (
                    round(float(self.segmentation_model_confidence_mean), 4)
                    if self.segmentation_model_confidence_mean is not None
                    else None
                ),
                "calibrated_structural_severity_confidence": self.calibrated_structural_severity_confidence,
                "confidence_distinction_notice": self.confidence_distinction_notice,
            },
            "missing_data_checklist": self.missing_data_checklist.to_dict(),
            "observations": [obs.to_dict() for obs in self.observations],
            "stage_4_prioritization_contract": self.stage_4_prioritization_contract,
            "literature_grounding": self.literature_grounding,
            "disclaimer": self.disclaimer,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
