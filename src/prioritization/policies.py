"""
Module: src.prioritization.policies
Description:
    Extensible Decision-Policy Interface and Implementations for Stage 4:
    Maintenance Prioritization Foundation.

    Contains:
    1. BasePrioritizationPolicy (Abstract Interface)
    2. ProductionAbstentionPolicy (Strict production reference with complete abstention)
    3. ResearchDemonstrationPolicy (Clearly labeled simulation mode for what-if research)
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.measurement.schema import (
    NON_STRUCTURAL_CLASSES,
    DefectVisualObservation,
    ExplainableSeverityReport,
    VisualConcernLevel,
)
from src.prioritization.schema import (
    MAINTENANCE_ABSTENTION_NOTICE,
    PRIORITIZATION_DISCLAIMER,
    RESEARCH_DEMO_POLICY_DISCLAIMER,
    ElementCriticality,
    EngineeringMaintenancePriority,
    HumanOverrideRecord,
    InspectionReviewQueue,
    MaintenancePriorityStatus,
    ReviewQueueItem,
    ReviewQueueUrgency,
    Stage4PrioritizationResult,
    VerifiedEngineeringContext,
)


class BasePrioritizationPolicy(ABC):
    """
    Abstract interface defining the prioritization policy contract.
    """

    @abstractmethod
    def evaluate(
        self,
        stage3_report: ExplainableSeverityReport,
        engineering_context: Optional[VerifiedEngineeringContext] = None,
        human_overrides: Optional[List[HumanOverrideRecord]] = None,
    ) -> Stage4PrioritizationResult:
        """
        Executes policy evaluation producing the dual outputs:
        Output A: Inspection Review Queue
        Output B: Engineering Maintenance Priority
        """
        pass


class ProductionAbstentionPolicy(BasePrioritizationPolicy):
    """
    Production-grade reference policy.

    Guarantees:
    - Provides a ranked Inspection Review Queue to guide human field inspectors.
    - Strictly ABSTAINS (NOT_ASSESSED) from asserting certified maintenance priorities,
      repair deadlines, or structural condition states.
    - Prevents non-structural features (Graffiti) from inflating structural urgency.
    - Includes optical uncertainty modeling, contradiction detection, and audit trails.
    """

    POLICY_NAME = "ProductionAbstentionPolicy"
    POLICY_VERSION = "1.0.0"
    POLICY_TYPE = "PRODUCTION_ABSTENTION"

    def __init__(
        self,
        high_concern_weight: float = 1.0,
        moderate_concern_weight: float = 0.60,
        low_concern_weight: float = 0.30,
        non_structural_weight: float = 0.05,
        min_reliable_confidence: float = 0.70,
        min_reliable_area_pixels: int = 25,
    ):
        self.w_high = high_concern_weight
        self.w_mod = moderate_concern_weight
        self.w_low = low_concern_weight
        self.w_non_struct = non_structural_weight
        self.min_reliable_confidence = min_reliable_confidence
        self.min_reliable_area_pixels = min_reliable_area_pixels

    def _compute_concern_weight(self, visual_concern: str) -> float:
        if visual_concern == VisualConcernLevel.HIGH_VISUAL_CONCERN.value:
            return self.w_high
        elif visual_concern == VisualConcernLevel.MODERATE_VISUAL_CONCERN.value:
            return self.w_mod
        elif visual_concern == VisualConcernLevel.LOW_VISUAL_CONCERN.value:
            return self.w_low
        return self.w_non_struct

    def _compute_criticality_weight(self, criticality: ElementCriticality) -> float:
        if criticality == ElementCriticality.FRACTURE_CRITICAL:
            return 1.0
        elif criticality == ElementCriticality.PRIMARY_LOAD_BEARING:
            return 0.75
        elif criticality == ElementCriticality.SECONDARY:
            return 0.40
        elif criticality == ElementCriticality.NON_STRUCTURAL:
            return 0.10
        return 0.50  # Conservative baseline for UNKNOWN

    def _detect_contradictions(
        self,
        stage3_report: ExplainableSeverityReport,
        context: Optional[VerifiedEngineeringContext],
    ) -> List[str]:
        warnings = []
        if context is None:
            return warnings

        # Contradiction: High visual concern (e.g. ExposedRebars/Spalling) on non-structural element
        if context.element_criticality == ElementCriticality.NON_STRUCTURAL:
            for obs in stage3_report.observations:
                if obs.visual_concern_level == VisualConcernLevel.HIGH_VISUAL_CONCERN.value and obs.is_structural:
                    warnings.append(
                        f"CONTRADICTION: Finding {obs.finding_id} ({obs.class_name}) indicates "
                        f"HIGH_VISUAL_CONCERN, but external context asserts element_criticality is NON_STRUCTURAL. "
                        f"Requires human inspector resolution."
                    )

        # Contradiction: Rebar exposure claimed with 0 depth
        if context.defect_depth_mm is not None and context.defect_depth_mm <= 0.0:
            for obs in stage3_report.observations:
                if obs.class_name in ("ExposedRebars", "Spalling"):
                    warnings.append(
                        f"CONTRADICTION: Finding {obs.finding_id} shows {obs.class_name}, "
                        f"but external defect_depth_mm is {context.defect_depth_mm} mm. "
                        f"Physical loss cannot exist with zero depth."
                    )

        return warnings

    def build_review_queue(
        self,
        stage3_report: ExplainableSeverityReport,
        context: Optional[VerifiedEngineeringContext] = None,
        human_overrides: Optional[List[HumanOverrideRecord]] = None,
    ) -> InspectionReviewQueue:
        override_map = {ov.finding_id: ov for ov in (human_overrides or [])}
        queue_items: List[ReviewQueueItem] = []

        now_iso = datetime.now(timezone.utc).isoformat()
        element_crit = context.element_criticality if context else ElementCriticality.UNKNOWN
        w_crit = self._compute_criticality_weight(element_crit)

        for obs in stage3_report.observations:
            w_concern = self._compute_concern_weight(obs.visual_concern_level)
            conf = float(obs.confidence_stats.mean)

            # Normalized extent: clamp at 2.0% image coverage to prevent single huge defects from dominating
            extent_norm = min(1.0, obs.image_relative_extent_pct / 2.0)

            # Composite verification ranking score (transparent multi-criteria formula)
            # Weights: Concern (40%), Extent (25%), Model Confidence (20%), Criticality (15%)
            if obs.is_structural:
                composite_score = (
                    0.40 * w_concern
                    + 0.25 * extent_norm
                    + 0.20 * conf
                    + 0.15 * w_crit
                )
            else:
                # Non-structural items (Graffiti, etc.) capped at non-structural weight
                composite_score = self.w_non_struct * (0.5 * conf + 0.5 * extent_norm)

            composite_score = float(np.clip(composite_score, 0.0, 1.0))

            # Optical uncertainty modeling
            uncertainty_factors = []
            if conf < self.min_reliable_confidence:
                uncertainty_factors.append(
                    f"Model confidence ({conf:.2f}) below reliability threshold ({self.min_reliable_confidence:.2f})"
                )
            if obs.pixel_area < self.min_reliable_area_pixels:
                uncertainty_factors.append(
                    f"Small defect area ({obs.pixel_area} px) near noise filter boundary"
                )
            if not obs.is_structural:
                uncertainty_factors.append("Non-structural category; zero direct load capacity impact")

            if len(uncertainty_factors) >= 2 or conf < 0.60:
                uncertainty_level = "HIGH"
            elif len(uncertainty_factors) == 1:
                uncertainty_level = "MODERATE"
            else:
                uncertainty_level = "LOW"

            # Assign Provisional Urgency Tier
            triggered_rules = []
            if not obs.is_structural:
                urgency = ReviewQueueUrgency.INFORMATIONAL_NON_STRUCTURAL
                triggered_rules.append("RULE_NON_STRUCTURAL_CATEGORY: Mapped to informational tracking.")
            elif obs.visual_concern_level == VisualConcernLevel.HIGH_VISUAL_CONCERN.value:
                if conf >= self.min_reliable_confidence:
                    urgency = ReviewQueueUrgency.URGENT_INSPECTION_REVIEW
                    triggered_rules.append("RULE_HIGH_CONCERN_RELIABLE: Direct cover breach or articulation distress with reliable confidence.")
                else:
                    urgency = ReviewQueueUrgency.PRIORITY_INSPECTION_REVIEW
                    triggered_rules.append("RULE_HIGH_CONCERN_UNCERTAIN: High concern feature with moderate confidence.")
            elif obs.visual_concern_level == VisualConcernLevel.MODERATE_VISUAL_CONCERN.value:
                urgency = ReviewQueueUrgency.PRIORITY_INSPECTION_REVIEW
                triggered_rules.append("RULE_MODERATE_CONCERN: Linear cracking or progressive concrete degradation pathway.")
            else:
                urgency = ReviewQueueUrgency.ROUTINE_INSPECTION_REVIEW
                triggered_rules.append("RULE_LOW_CONCERN: Surface weathering, efflorescence, or superficial staining.")

            if element_crit == ElementCriticality.FRACTURE_CRITICAL and obs.is_structural:
                if urgency != ReviewQueueUrgency.URGENT_INSPECTION_REVIEW:
                    urgency = ReviewQueueUrgency.URGENT_INSPECTION_REVIEW
                    triggered_rules.append("RULE_FRACTURE_CRITICAL_ELEVATION: Elevated to urgent due to verified fracture-critical member role.")

            # Decision limitations
            limitations = [
                "Optical 2D estimation only; subsurface defect depth is unobserved.",
                "Reinforcement cross-section loss cannot be confirmed without caliper/NDT sounding.",
                "Physical metric dimensions require calibrated optical rangefinder.",
            ]

            # Apply human override if present
            override_rec = override_map.get(obs.finding_id)
            if override_rec:
                urgency = ReviewQueueUrgency(override_rec.overridden_urgency)
                triggered_rules.append(
                    f"RULE_HUMAN_OVERRIDE: Urgency overridden by Inspector {override_rec.inspector_id} "
                    f"with justification: {override_rec.justification}"
                )

            item = ReviewQueueItem(
                rank=0,  # Assigned after sorting
                finding_id=obs.finding_id,
                class_name=obs.class_name,
                queue_urgency=urgency,
                composite_verification_score=composite_score,
                is_structural=obs.is_structural,
                visual_evidence_summary=obs.visual_evidence,
                image_relative_extent_pct=obs.image_relative_extent_pct,
                roi_relative_extent_pct=obs.roi_relative_extent_pct,
                model_confidence=conf,
                optical_uncertainty_level=uncertainty_level,
                uncertainty_factors=uncertainty_factors,
                triggered_policy_rules=triggered_rules,
                recommended_inspector_action=obs.inspector_action_items,
                decision_limitations=limitations,
                human_override=override_rec,
            )
            queue_items.append(item)

        # Sort queue descending: composite score first, structural priority next
        urgency_priority = {
            ReviewQueueUrgency.URGENT_INSPECTION_REVIEW: 4,
            ReviewQueueUrgency.PRIORITY_INSPECTION_REVIEW: 3,
            ReviewQueueUrgency.ROUTINE_INSPECTION_REVIEW: 2,
            ReviewQueueUrgency.INFORMATIONAL_NON_STRUCTURAL: 1,
            ReviewQueueUrgency.NO_ACTION_REQUIRED: 0,
        }
        queue_items.sort(
            key=lambda x: (urgency_priority.get(x.queue_urgency, 0), x.composite_verification_score),
            reverse=True,
        )

        for idx, itm in enumerate(queue_items, start=1):
            itm.rank = idx

        # Determine queue summary metrics
        struct_count = sum(1 for itm in queue_items if itm.is_structural)
        non_struct_count = len(queue_items) - struct_count
        highest_urg = queue_items[0].queue_urgency.value if queue_items else ReviewQueueUrgency.NO_ACTION_REQUIRED.value

        return InspectionReviewQueue(
            image_id=stage3_report.image_id,
            timestamp_utc=now_iso,
            total_findings_count=len(queue_items),
            structural_findings_count=struct_count,
            non_structural_findings_count=non_struct_count,
            highest_queue_urgency=highest_urg,
            queue_items=queue_items,
        )

    def evaluate_maintenance_priority(
        self,
        stage3_report: ExplainableSeverityReport,
        context: Optional[VerifiedEngineeringContext] = None,
    ) -> EngineeringMaintenancePriority:
        now_iso = datetime.now(timezone.utc).isoformat()
        contradictions = self._detect_contradictions(stage3_report, context)

        # Check all statutory prerequisites
        missing_prereqs = []
        if context is None or not context.structural_component:
            missing_prereqs.append("Verified structural component identity (BIM or inspection plan)")
        if context is None or context.element_criticality == ElementCriticality.UNKNOWN:
            missing_prereqs.append("Element load path criticality (AASHTO / NBI element classification)")
        if context is None or context.calibrated_dimensions is None:
            missing_prereqs.append("Physical metric defect dimensions (calibrated optical/laser measurement)")
        if context is None or context.defect_depth_mm is None:
            missing_prereqs.append("Internal defect depth & delamination plane (NDT sounding, GPR, or UPV)")
        if context is None or not context.is_verified_by_pe:
            missing_prereqs.append("Formal validation and sign-off by a licensed Professional Engineer (PE/SE)")

        # Always abstains in production policy without approved agency decision rules
        missing_prereqs.append("Formally adopted Department of Transportation (DOT) maintenance decision policy")

        return EngineeringMaintenancePriority(
            image_id=stage3_report.image_id,
            timestamp_utc=now_iso,
            status=MaintenancePriorityStatus.NOT_ASSESSED,
            official_repair_priority="NOT_ASSESSED",
            official_repair_deadline=None,
            official_structural_safety_rating=None,
            official_aashto_condition_state=None,
            abstention_reason=MAINTENANCE_ABSTENTION_NOTICE,
            missing_engineering_prerequisites=missing_prereqs,
            decision_criteria_approved_by_agency=False,
            is_pe_certified=bool(context and context.is_verified_by_pe),
            contradiction_warnings=contradictions,
        )

    def evaluate(
        self,
        stage3_report: ExplainableSeverityReport,
        engineering_context: Optional[VerifiedEngineeringContext] = None,
        human_overrides: Optional[List[HumanOverrideRecord]] = None,
    ) -> Stage4PrioritizationResult:
        now_iso = datetime.now(timezone.utc).isoformat()

        queue = self.build_review_queue(stage3_report, engineering_context, human_overrides)
        maint_priority = self.evaluate_maintenance_priority(stage3_report, engineering_context)

        audit_trail = {
            "policy_name": self.POLICY_NAME,
            "policy_version": self.POLICY_VERSION,
            "policy_type": self.POLICY_TYPE,
            "evaluated_at_utc": now_iso,
            "stage3_image_id": stage3_report.image_id,
            "stage3_visual_concern": stage3_report.visual_concern_level,
            "stage3_review_flags": stage3_report.review_flags,
            "input_observations_count": len(stage3_report.observations),
            "engineering_context_supplied": engineering_context is not None,
            "human_overrides_applied_count": len(human_overrides or []),
            "abstention_enforced": True,
        }

        return Stage4PrioritizationResult(
            image_id=stage3_report.image_id,
            policy_name=self.POLICY_NAME,
            policy_version=self.POLICY_VERSION,
            policy_type=self.POLICY_TYPE,
            inspection_review_queue=queue,
            engineering_maintenance_priority=maint_priority,
            verified_engineering_context=engineering_context,
            audit_trail=audit_trail,
            timestamp_utc=now_iso,
        )


class ResearchDemonstrationPolicy(BasePrioritizationPolicy):
    """
    Clearly labeled research demonstration policy.

    Explicitly isolated from certified engineering decisions.
    Allows parameterized what-if simulation when hypothetical inputs are supplied.
    """

    POLICY_NAME = "ResearchDemonstrationPolicy"
    POLICY_VERSION = "0.9.0-demonstration"
    POLICY_TYPE = "RESEARCH_DEMONSTRATION_SIMULATION_ONLY"

    def __init__(self, allow_simulation: bool = True):
        self.allow_simulation = allow_simulation
        self.base_policy = ProductionAbstentionPolicy()

    def evaluate(
        self,
        stage3_report: ExplainableSeverityReport,
        engineering_context: Optional[VerifiedEngineeringContext] = None,
        human_overrides: Optional[List[HumanOverrideRecord]] = None,
    ) -> Stage4PrioritizationResult:
        now_iso = datetime.now(timezone.utc).isoformat()

        # Build standard review queue
        queue = self.base_policy.build_review_queue(stage3_report, engineering_context, human_overrides)

        # In research simulation mode, if verified engineering context is provided,
        # illustrate parameterized condition state mapping with strict disclaimers
        if self.allow_simulation and engineering_context and engineering_context.has_sufficient_structural_data:
            hypothetical_priority = "DEMO_HIGH_PRIORITY" if queue.highest_queue_urgency == ReviewQueueUrgency.URGENT_INSPECTION_REVIEW.value else "DEMO_ROUTINE"
            maint = EngineeringMaintenancePriority(
                image_id=stage3_report.image_id,
                timestamp_utc=now_iso,
                status=MaintenancePriorityStatus.ASSESSED_RESEARCH_DEMO_ONLY,
                official_repair_priority=f"HYPOTHETICAL_{hypothetical_priority}",
                official_repair_deadline="SIMULATED_DEMO_90_DAYS",
                official_structural_safety_rating="SIMULATED_FAIR_CONDITION",
                official_aashto_condition_state="SIMULATED_CS3_POOR",
                abstention_reason="RESEARCH SIMULATION MODE ONLY: Result is for academic illustration and is not approved by any agency.",
                missing_engineering_prerequisites=["Official DOT statutory sign-off"],
                decision_criteria_approved_by_agency=False,
                is_pe_certified=engineering_context.is_verified_by_pe,
                contradiction_warnings=[],
                disclaimer=RESEARCH_DEMO_POLICY_DISCLAIMER,
            )
        else:
            # Fall back to strict abstention
            maint = self.base_policy.evaluate_maintenance_priority(stage3_report, engineering_context)

        audit_trail = {
            "policy_name": self.POLICY_NAME,
            "policy_version": self.POLICY_VERSION,
            "policy_type": self.POLICY_TYPE,
            "evaluated_at_utc": now_iso,
            "simulation_mode": True,
            "disclaimer": RESEARCH_DEMO_POLICY_DISCLAIMER,
        }

        return Stage4PrioritizationResult(
            image_id=stage3_report.image_id,
            policy_name=self.POLICY_NAME,
            policy_version=self.POLICY_VERSION,
            policy_type=self.POLICY_TYPE,
            inspection_review_queue=queue,
            engineering_maintenance_priority=maint,
            verified_engineering_context=engineering_context,
            audit_trail=audit_trail,
            timestamp_utc=now_iso,
        )
