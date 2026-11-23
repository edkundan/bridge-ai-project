"""
Module: src.measurement.severity
Description:
    Literature-grounded severity assessment interface and extensible foundation.

    CRITICAL ARCHITECTURAL BOUNDARY:
    Separates measured optical evidence (Stage 2) from engineering severity
    decisions. Does NOT invent arbitrary engineering thresholds or claim
    validated structural risk ratings.

    Grounding in Structural Inspection Standards:
    - AASHTO Manual for Bridge Element Inspection (MBEI, 2019):
      Condition States CS1 (Good), CS2 (Fair), CS3 (Poor), CS4 (Severe).
    - FHWA Bridge Inspection Reference Manual (BIRM, 2012):
      National Bridge Inspection Standards (NBIS) component ratings.
    - German Standard DIN 1076 / RI-EBW-PRÜF:
      Bridge inspection defect categorization (Schadensausmaß & Schadensintensität).
    - FIB Model Code 2010 & Eurocode 2 (EN 1992-1-1):
      Durability limit states, crack width thresholds, and cover concrete spalling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.measurement.schema import (
    SEVERITY_VALIDATION_DISCLAIMER,
    ImageMeasurementResult,
)


class StructuralCriticalityTier(str, Enum):
    """
    Literature-grounded defect categorization reflecting structural risk potential.
    Based on AASHTO MBEI and FHWA BIRM defect taxonomies.
    """
    TIER_1_CRITICAL = "TIER_1_CRITICAL"
    """Directly threatens load capacity, reinforcement continuity, or articulation.
    Classes: ExposedRebars, Spalling, Rockpocket, Bearing, EJoint."""

    TIER_2_PROGRESSIVE = "TIER_2_PROGRESSIVE"
    """Progressive material degradation, moisture pathways, active shear/flexure.
    Classes: Crack, ACrack, WConccor."""

    TIER_3_SUPERFICIAL = "TIER_3_SUPERFICIAL"
    """Surface-level weathering, chemical leaching, cosmetic or early-stage defects.
    Classes: Efflorescence, Rust, Weathering, Wetspot, Cavity, Hollowareas."""

    TIER_4_NON_STRUCTURAL = "TIER_4_NON_STRUCTURAL"
    """Auxiliary hardware, formwork remnants, surface markings.
    Classes: Graffiti, Drainage, JTape, PEquipment, Restformwork."""


# Mapping DACL10K classes to literature-grounded criticality tiers
DACL10K_TIER_MAPPING: Dict[str, StructuralCriticalityTier] = {
    # Tier 1: Critical Structural Elements / Load-Bearing Articulation
    "ExposedRebars": StructuralCriticalityTier.TIER_1_CRITICAL,
    "Spalling": StructuralCriticalityTier.TIER_1_CRITICAL,
    "Rockpocket": StructuralCriticalityTier.TIER_1_CRITICAL,
    "Bearing": StructuralCriticalityTier.TIER_1_CRITICAL,
    "EJoint": StructuralCriticalityTier.TIER_1_CRITICAL,

    # Tier 2: Progressive Cracking & Concrete Degradation
    "Crack": StructuralCriticalityTier.TIER_2_PROGRESSIVE,
    "ACrack": StructuralCriticalityTier.TIER_2_PROGRESSIVE,
    "WConccor": StructuralCriticalityTier.TIER_2_PROGRESSIVE,

    # Tier 3: Environmental & Superficial Indicators
    "Efflorescence": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Rust": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Weathering": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Wetspot": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Cavity": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Hollowareas": StructuralCriticalityTier.TIER_3_SUPERFICIAL,

    # Tier 4: Non-structural / Appurtenances / Markings
    "Graffiti": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
    "Drainage": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
    "JTape": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
    "PEquipment": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
    "Restformwork": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
}


@dataclass
class SeverityAssessmentResult:
    """
    Extensible assessment result capturing visual evidence aggregation,
    provisional literature-based categorization, and explicit validation disclosures.
    """
    image_id: str
    highest_criticality_tier: Optional[str]
    detected_tiers: List[str]
    tier_summary: Dict[str, Dict[str, Any]]
    total_defect_image_area_pct: float
    total_defect_roi_area_pct: Optional[float]
    is_calibrated: bool = False
    requires_expert_validation: bool = True
    literature_grounding: List[str] = field(default_factory=lambda: [
        "AASHTO Manual for Bridge Element Inspection (MBEI, 2019)",
        "FHWA Bridge Inspection Reference Manual (BIRM, 2012)",
        "DIN 1076 / RI-EBW-PRÜF: Bridge Evaluation Guidelines",
        "FIB Model Code 2010 (Concrete Durability and Service Life Design)",
    ])
    missing_engineering_parameters: List[str] = field(default_factory=lambda: [
        "1. Metric scale calibration / sensor distance (pixels to physical mm/cm)",
        "2. Structural element hierarchy (e.g. fracture-critical girder vs secondary diaphragm)",
        "3. Internal defect depth / 3D geometry (unobservable via surface 2D RGB imagery)",
        "4. Reinforcement cross-section loss and corrosion penetration depth",
        "5. Ultrasonic pulse velocity / impact-echo / GPR non-destructive testing data",
        "6. Actual live/dead load stress state and bridge design rating",
    ])
    disclaimer: str = SEVERITY_VALIDATION_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "provisional_criticality_tier": self.highest_criticality_tier,
            "detected_tiers": self.detected_tiers,
            "tier_summary": self.tier_summary,
            "total_defect_image_area_pct": round(float(self.total_defect_image_area_pct), 5),
            "total_defect_roi_area_pct": (
                round(float(self.total_defect_roi_area_pct), 5)
                if self.total_defect_roi_area_pct is not None
                else None
            ),
            "validation_status": {
                "is_calibrated": self.is_calibrated,
                "requires_expert_validation": self.requires_expert_validation,
                "missing_engineering_parameters": self.missing_engineering_parameters,
                "literature_grounding": self.literature_grounding,
            },
            "disclaimer": self.disclaimer,
        }


class BaseSeverityEvaluator(ABC):
    """
    Abstract interface for severity assessment algorithms.
    Decouples visual measurement from engineering policy/risk scoring.
    """

    @abstractmethod
    def evaluate(self, measurement: ImageMeasurementResult) -> SeverityAssessmentResult:
        """
        Evaluates an ImageMeasurementResult to produce a structured severity assessment.

        Args:
            measurement: ImageMeasurementResult containing extracted pixel measurements.

        Returns:
            SeverityAssessmentResult instance.
        """
        pass


class LiteratureGroundedSeverityEvaluator(BaseSeverityEvaluator):
    """
    Baseline literature-grounded severity assessment evaluator.
    Categorizes detected defects according to AASHTO MBEI / FHWA BIRM tiers.

    NOTE: Explicitly flagged as uncalibrated demonstration pending engineering ground truth.
    """

    def __init__(
        self,
        tier_mapping: Optional[Dict[str, StructuralCriticalityTier]] = None,
    ):
        self.tier_mapping = tier_mapping or DACL10K_TIER_MAPPING

    def evaluate(self, measurement: ImageMeasurementResult) -> SeverityAssessmentResult:
        tier_summary: Dict[str, Dict[str, Any]] = {
            tier.value: {
                "classes_detected": [],
                "component_count": 0,
                "total_pixel_area": 0,
                "image_relative_area_pct": 0.0,
            }
            for tier in StructuralCriticalityTier
        }

        detected_tiers: List[str] = []

        for class_name, summary in measurement.class_measurements.items():
            if not summary.detected:
                continue

            tier = self.tier_mapping.get(class_name, StructuralCriticalityTier.TIER_3_SUPERFICIAL)
            tier_key = tier.value

            tier_info = tier_summary[tier_key]
            tier_info["classes_detected"].append(class_name)
            tier_info["component_count"] += summary.component_count
            tier_info["total_pixel_area"] += summary.total_pixel_area
            tier_info["image_relative_area_pct"] += summary.image_relative_area_pct

            if tier_key not in detected_tiers:
                detected_tiers.append(tier_key)

        # Round area percentages
        for tier_key in tier_summary:
            tier_summary[tier_key]["image_relative_area_pct"] = round(
                tier_summary[tier_key]["image_relative_area_pct"], 5
            )

        # Determine highest criticality tier
        tier_order = [
            StructuralCriticalityTier.TIER_1_CRITICAL.value,
            StructuralCriticalityTier.TIER_2_PROGRESSIVE.value,
            StructuralCriticalityTier.TIER_3_SUPERFICIAL.value,
            StructuralCriticalityTier.TIER_4_NON_STRUCTURAL.value,
        ]

        highest_tier = None
        for t in tier_order:
            if t in detected_tiers:
                highest_tier = t
                break

        return SeverityAssessmentResult(
            image_id=measurement.image_id,
            highest_criticality_tier=highest_tier,
            detected_tiers=detected_tiers,
            tier_summary=tier_summary,
            total_defect_image_area_pct=measurement.total_defect_image_area_pct,
            total_defect_roi_area_pct=measurement.total_defect_roi_area_pct,
            is_calibrated=False,
            requires_expert_validation=True,
        )
