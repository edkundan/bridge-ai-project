"""
Module: src.prioritization.schema
Description:
    Standard dataclasses, types, enums, and JSON serialization schemas for
    Stage 4: Explainable, Evidence-Aware Maintenance Prioritization Foundation.

    Enforces the fundamental architectural requirement:
    - Dual Distinct Outputs:
      1. Inspection Review Queue (Provisional ranking to guide human verification)
      2. Engineering Maintenance Priority (Strictly NOT_ASSESSED pending verified data)
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

PRIORITIZATION_DISCLAIMER = (
    "STAGE 4 RESEARCH DISCLAIMER: The Inspection Review Queue provides an "
    "evidence-weighted ranking of visual defect findings strictly to prioritize human "
    "inspector site visits and verification. It does NOT assert official structural risk, "
    "repair deadlines, maintenance work orders, structural failure probabilities, or "
    "certified AASHTO bridge element condition ratings."
)

MAINTENANCE_ABSTENTION_NOTICE = (
    "ENGINEERING MAINTENANCE PRIORITY ABSTENTION: Actionable bridge maintenance priority, "
    "repair work orders, and statutory condition states CANNOT be determined from 2D optical "
    "computer vision detections alone. Official engineering prioritization strictly requires: "
    "(1) verified structural component identity and load path criticality, (2) physical metric "
    "dimensions and internal defect depth from calibrated NDT tools, (3) documented historical "
    "progression rates, and (4) formal approval by a licensed Professional Engineer (PE/SE) "
    "under agency-adopted maintenance decision criteria. Consequently, official maintenance "
    "priority is marked NOT_ASSESSED."
)

RESEARCH_DEMO_POLICY_DISCLAIMER = (
    "RESEARCH DEMONSTRATION SIMULATION ONLY: This evaluation uses explicitly hypothetical "
    "inputs and parameterized heuristic rules for academic demonstration and what-if exploration. "
    "It is strictly decoupled from certified engineering decision policies and MUST NOT be used "
    "for actual bridge maintenance scheduling or public safety decisions."
)


# ============================================================================
# Enums
# ============================================================================

class ReviewQueueUrgency(str, Enum):
    """
    Provisional urgency tiers for human inspector site review and verification.
    """
    URGENT_INSPECTION_REVIEW = "URGENT_INSPECTION_REVIEW"
    """High visual concern defects directly breaching cover or articulation
    (e.g. ExposedRebars, active Spalling, Bearing seizure, Expansion Joint breach)."""

    PRIORITY_INSPECTION_REVIEW = "PRIORITY_INSPECTION_REVIEW"
    """Progressive deterioration pathways warranting timely field verification
    (e.g. extensive Crack networks, Rockpocket honeycombing, Weak Concrete)."""

    ROUTINE_INSPECTION_REVIEW = "ROUTINE_INSPECTION_REVIEW"
    """Superficial environmental indicators or minor cosmetic voiding
    (e.g. Efflorescence, surface Rust staining, Wetspots, Cavities)."""

    INFORMATIONAL_NON_STRUCTURAL = "INFORMATIONAL_NON_STRUCTURAL"
    """Non-structural surface markings or appurtenances (e.g. Graffiti, formwork).
    Documented for informational tracking; does not represent structural damage."""

    NO_ACTION_REQUIRED = "NO_ACTION_REQUIRED"
    """No defect patterns detected above active threshold."""


class MaintenancePriorityStatus(str, Enum):
    """
    Status of the statutory engineering maintenance priority decision.
    """
    NOT_ASSESSED = "NOT_ASSESSED"
    """Default and required status when structural context or engineering criteria are incomplete."""

    UNASSESSED_MISSING_DATA = "UNASSESSED_MISSING_DATA"
    """Abstention due to missing physical depth, calibrated dimensions, or element identity."""

    UNASSESSED_PENDING_ENGINEERING_APPROVAL = "UNASSESSED_PENDING_ENGINEERING_APPROVAL"
    """Abstention pending review and sign-off by a licensed Professional Engineer."""

    ASSESSED_RESEARCH_DEMO_ONLY = "ASSESSED_RESEARCH_DEMO_ONLY"
    """Clearly flagged simulation result generated exclusively under research demonstration mode."""


class ElementCriticality(str, Enum):
    """
    Structural importance of the bridge element containing the defect.
    """
    FRACTURE_CRITICAL = "FRACTURE_CRITICAL"
    """Failure of this member would cause collapse of bridge section."""

    PRIMARY_LOAD_BEARING = "PRIMARY_LOAD_BEARING"
    """Major load-carrying member (e.g. main pier column, prestressed girder, pier cap)."""

    SECONDARY = "SECONDARY"
    """Secondary structural member (e.g. cross-bracing, diaphragm, deck overhang)."""

    NON_STRUCTURAL = "NON_STRUCTURAL"
    """Appurtenance, barrier wall, architectural finish, or signage."""

    UNKNOWN = "UNKNOWN"
    """Structural element role not provided or unverified."""


class DefectProgressionState(str, Enum):
    """
    Historical progression rate of the defect based on agency records.
    """
    FIRST_OBSERVED = "FIRST_OBSERVED"
    ACTIVE_PROGRESSING = "ACTIVE_PROGRESSING"
    STABLE_DORMANT = "STABLE_DORMANT"
    PREVIOUSLY_REPAIRED = "PREVIOUSLY_REPAIRED"
    UNKNOWN = "UNKNOWN"


class EnvironmentalExposure(str, Enum):
    """
    Corrosive or environmental exposure severity.
    """
    DEICING_SALTS_SEVERE = "DEICING_SALTS_SEVERE"
    COASTAL_MARINE = "COASTAL_MARINE"
    MODERATE_WEATHERING = "MODERATE_WEATHERING"
    DRY_BENIGN = "DRY_BENIGN"
    UNKNOWN = "UNKNOWN"


# ============================================================================
# Verified Engineering Context (Optional Input)
# ============================================================================

@dataclass
class VerifiedEngineeringContext:
    """
    Optional human-verified or external engineering telemetry supplied to Stage 4.
    """
    bridge_id: Optional[str] = None
    structural_component: Optional[str] = None
    element_criticality: ElementCriticality = ElementCriticality.UNKNOWN
    calibrated_dimensions: Optional[Dict[str, float]] = None
    defect_depth_mm: Optional[float] = None
    inspection_history: DefectProgressionState = DefectProgressionState.UNKNOWN
    progression_rate_mm_per_year: Optional[float] = None
    exposure_condition: EnvironmentalExposure = EnvironmentalExposure.UNKNOWN
    professional_engineer_notes: Optional[str] = None
    is_verified_by_pe: bool = False
    verification_timestamp: Optional[str] = None

    @property
    def has_sufficient_structural_data(self) -> bool:
        """
        Determines whether external inputs meet the minimum bar for non-abstention:
        Requires identified component, non-unknown criticality, metric dimensions,
        physical depth, and PE sign-off.
        """
        return (
            bool(self.structural_component)
            and self.element_criticality not in (ElementCriticality.UNKNOWN, ElementCriticality.NON_STRUCTURAL)
            and self.calibrated_dimensions is not None
            and self.defect_depth_mm is not None
            and self.is_verified_by_pe
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "structural_component": self.structural_component,
            "element_criticality": self.element_criticality.value,
            "calibrated_dimensions": self.calibrated_dimensions,
            "defect_depth_mm": self.defect_depth_mm,
            "inspection_history": self.inspection_history.value,
            "progression_rate_mm_per_year": self.progression_rate_mm_per_year,
            "exposure_condition": self.exposure_condition.value,
            "professional_engineer_notes": self.professional_engineer_notes,
            "is_verified_by_pe": self.is_verified_by_pe,
            "verification_timestamp": self.verification_timestamp,
            "has_sufficient_structural_data": self.has_sufficient_structural_data,
        }


# ============================================================================
# Human Override & Audit Trail Dataclasses
# ============================================================================

@dataclass
class HumanOverrideRecord:
    """
    Audit log entry capturing a certified human inspector's adjustment of an AI finding.
    """
    override_id: str
    finding_id: str
    inspector_id: str
    timestamp_utc: str
    original_urgency: str
    overridden_urgency: str
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "override_id": self.override_id,
            "finding_id": self.finding_id,
            "inspector_id": self.inspector_id,
            "timestamp_utc": self.timestamp_utc,
            "original_urgency": self.original_urgency,
            "overridden_urgency": self.overridden_urgency,
            "justification": self.justification,
        }


# ============================================================================
# Output A: Inspection Review Queue
# ============================================================================

@dataclass
class ReviewQueueItem:
    """
    An individual ranked finding in the provisional Inspection Review Queue.
    """
    rank: int
    finding_id: str
    class_name: str
    queue_urgency: ReviewQueueUrgency
    composite_verification_score: float  # [0.0, 1.0]
    is_structural: bool
    visual_evidence_summary: str
    image_relative_extent_pct: float
    roi_relative_extent_pct: Optional[float]
    model_confidence: float
    optical_uncertainty_level: str  # "LOW", "MODERATE", "HIGH"
    uncertainty_factors: List[str]
    triggered_policy_rules: List[str]
    recommended_inspector_action: List[str]
    decision_limitations: List[str]
    human_override: Optional[HumanOverrideRecord] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "finding_id": self.finding_id,
            "class_name": self.class_name,
            "queue_urgency": self.queue_urgency.value,
            "composite_verification_score": round(float(self.composite_verification_score), 4),
            "is_structural": self.is_structural,
            "visual_evidence_summary": self.visual_evidence_summary,
            "image_relative_extent_pct": round(float(self.image_relative_extent_pct), 5),
            "roi_relative_extent_pct": (
                round(float(self.roi_relative_extent_pct), 5)
                if self.roi_relative_extent_pct is not None
                else None
            ),
            "model_confidence": round(float(self.model_confidence), 4),
            "optical_uncertainty_level": self.optical_uncertainty_level,
            "uncertainty_factors": self.uncertainty_factors,
            "triggered_policy_rules": self.triggered_policy_rules,
            "recommended_inspector_action": self.recommended_inspector_action,
            "decision_limitations": self.decision_limitations,
            "human_override": self.human_override.to_dict() if self.human_override else None,
        }


@dataclass
class InspectionReviewQueue:
    """
    Output A: Ranked queue of visual findings to triage human field verification.
    """
    image_id: str
    timestamp_utc: str
    total_findings_count: int
    structural_findings_count: int
    non_structural_findings_count: int
    highest_queue_urgency: str
    queue_items: List[ReviewQueueItem]
    disclaimer: str = PRIORITIZATION_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "timestamp_utc": self.timestamp_utc,
            "queue_summary": {
                "total_findings_count": self.total_findings_count,
                "structural_findings_count": self.structural_findings_count,
                "non_structural_findings_count": self.non_structural_findings_count,
                "highest_queue_urgency": self.highest_queue_urgency,
            },
            "ranked_queue_items": [item.to_dict() for item in self.queue_items],
            "disclaimer": self.disclaimer,
        }


# ============================================================================
# Output B: Engineering Maintenance Priority (Strict Abstention by Default)
# ============================================================================

@dataclass
class EngineeringMaintenancePriority:
    """
    Output B: Statutory engineering maintenance priority and work order decision.
    Strictly marked NOT_ASSESSED unless verified structural criteria are loaded.
    """
    image_id: str
    timestamp_utc: str
    status: MaintenancePriorityStatus
    official_repair_priority: str  # "NOT_ASSESSED"
    official_repair_deadline: Optional[str]  # None
    official_structural_safety_rating: Optional[str]  # None
    official_aashto_condition_state: Optional[str]  # None
    abstention_reason: str
    missing_engineering_prerequisites: List[str]
    decision_criteria_approved_by_agency: bool
    is_pe_certified: bool
    contradiction_warnings: List[str] = field(default_factory=list)
    disclaimer: str = MAINTENANCE_ABSTENTION_NOTICE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "timestamp_utc": self.timestamp_utc,
            "priority_decision": {
                "status": self.status.value,
                "official_repair_priority": self.official_repair_priority,
                "official_repair_deadline": self.official_repair_deadline,
                "official_structural_safety_rating": self.official_structural_safety_rating,
                "official_aashto_condition_state": self.official_aashto_condition_state,
                "abstention_reason": self.abstention_reason,
            },
            "governance_and_compliance": {
                "decision_criteria_approved_by_agency": self.decision_criteria_approved_by_agency,
                "is_pe_certified": self.is_pe_certified,
                "missing_engineering_prerequisites": self.missing_engineering_prerequisites,
                "contradiction_warnings": self.contradiction_warnings,
            },
            "disclaimer": self.disclaimer,
        }


# ============================================================================
# Unified Stage 4 Prioritization Result
# ============================================================================

@dataclass
class Stage4PrioritizationResult:
    """
    Combined Stage 4 output containing both Output A (Inspection Review Queue)
    and Output B (Engineering Maintenance Priority) with complete audit trail.
    """
    image_id: str
    policy_name: str
    policy_version: str
    policy_type: str  # "PRODUCTION_ABSTENTION" or "RESEARCH_DEMONSTRATION_SIMULATION_ONLY"
    inspection_review_queue: InspectionReviewQueue
    engineering_maintenance_priority: EngineeringMaintenancePriority
    verified_engineering_context: Optional[VerifiedEngineeringContext] = None
    audit_trail: Dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "image_id": self.image_id,
                "timestamp_utc": self.timestamp_utc,
                "policy_name": self.policy_name,
                "policy_version": self.policy_version,
                "policy_type": self.policy_type,
            },
            "output_a_inspection_review_queue": self.inspection_review_queue.to_dict(),
            "output_b_engineering_maintenance_priority": self.engineering_maintenance_priority.to_dict(),
            "verified_engineering_context": (
                self.verified_engineering_context.to_dict()
                if self.verified_engineering_context
                else None
            ),
            "audit_trail": self.audit_trail,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
