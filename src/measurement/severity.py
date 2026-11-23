"""
Module: src.measurement.severity
Description:
    Stage 3: Explainable Visual Severity Assessment Foundation.

    CRITICAL ARCHITECTURAL BOUNDARIES & RESEARCH PRINCIPLES:
    1. Visual Evidence vs. Structural Decisions:
       Optical 2D imagery provides visual defect footprints, morphology, and
       detection confidence. It does NOT provide certified structural condition
       ratings, remaining capacity, or maintenance priority deadlines.
    2. Segregation of Non-Structural Categories:
       Non-structural markings (e.g. Graffiti) must never dominate structural
       defect extent. Structural and non-structural coverages are strictly separated.
    3. Explicit Abstention:
       When structural component identity, element importance, physical metric scale,
       defect depth, or inspection history are missing, the official condition state
       is marked NOT_ASSESSED rather than fabricating an uncalibrated risk score.
    4. Structured Review Flags:
       Generates human-in-the-loop review flags (e.g. POSSIBLE_SPALLING_REVIEW,
       REBAR_EXPOSURE_REVIEW) with actionable field verification guidance.
    5. Literature Grounding:
       Grounded in authoritative bridge inspection standards:
       - AASHTO Manual for Bridge Element Inspection (MBEI, 2019)
       - FHWA Bridge Inspection Reference Manual (BIRM, 2012)
       - German DIN 1076 / RI-EBW-PRÜF: Bridge Evaluation Guidelines
       - FIB Model Code 2010 & Eurocode 2 (EN 1992-1-1)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.measurement.schema import (
    ABSTENTION_JUSTIFICATION,
    CONFIDENCE_DISTINCTION_NOTICE,
    NON_STRUCTURAL_CLASSES,
    SEVERITY_VALIDATION_DISCLAIMER,
    DefectVisualObservation,
    ExplainableSeverityReport,
    ImageMeasurementResult,
    MissingDataChecklist,
    ReviewFlag,
    StructuralCriticalityTier,
    VisualConcernLevel,
)

# Mapping DACL10K classes to Visual Concern Levels (Observation-level, not condition states)
CLASS_VISUAL_CONCERN_MAPPING: Dict[str, VisualConcernLevel] = {
    # High Visual Concern: Concrete cover breach, reinforcement exposure, joint/bearing distress
    "ExposedRebars": VisualConcernLevel.HIGH_VISUAL_CONCERN,
    "Spalling": VisualConcernLevel.HIGH_VISUAL_CONCERN,
    "Rockpocket": VisualConcernLevel.HIGH_VISUAL_CONCERN,
    "Bearing": VisualConcernLevel.HIGH_VISUAL_CONCERN,
    "EJoint": VisualConcernLevel.HIGH_VISUAL_CONCERN,

    # Moderate Visual Concern: Linear cracking or concrete deterioration pathways
    "Crack": VisualConcernLevel.MODERATE_VISUAL_CONCERN,
    "ACrack": VisualConcernLevel.MODERATE_VISUAL_CONCERN,
    "WConccor": VisualConcernLevel.MODERATE_VISUAL_CONCERN,

    # Low Visual Concern: Environmental/surface indicators of moisture ingress or chemical leaching
    "Efflorescence": VisualConcernLevel.LOW_VISUAL_CONCERN,
    "Rust": VisualConcernLevel.LOW_VISUAL_CONCERN,
    "Weathering": VisualConcernLevel.LOW_VISUAL_CONCERN,
    "Wetspot": VisualConcernLevel.LOW_VISUAL_CONCERN,
    "Cavity": VisualConcernLevel.LOW_VISUAL_CONCERN,
    "Hollowareas": VisualConcernLevel.LOW_VISUAL_CONCERN,

    # Negligible / Non-Structural: Appurtenances, formwork remnants, surface paint
    "Graffiti": VisualConcernLevel.NEGLIGIBLE_NON_STRUCTURAL,
    "Drainage": VisualConcernLevel.NEGLIGIBLE_NON_STRUCTURAL,
    "JTape": VisualConcernLevel.NEGLIGIBLE_NON_STRUCTURAL,
    "PEquipment": VisualConcernLevel.NEGLIGIBLE_NON_STRUCTURAL,
    "Restformwork": VisualConcernLevel.NEGLIGIBLE_NON_STRUCTURAL,
}

# Backward compatibility alias
DACL10K_TIER_MAPPING: Dict[str, StructuralCriticalityTier] = {
    "ExposedRebars": StructuralCriticalityTier.TIER_1_CRITICAL,
    "Spalling": StructuralCriticalityTier.TIER_1_CRITICAL,
    "Rockpocket": StructuralCriticalityTier.TIER_1_CRITICAL,
    "Bearing": StructuralCriticalityTier.TIER_1_CRITICAL,
    "EJoint": StructuralCriticalityTier.TIER_1_CRITICAL,
    "Crack": StructuralCriticalityTier.TIER_2_PROGRESSIVE,
    "ACrack": StructuralCriticalityTier.TIER_2_PROGRESSIVE,
    "WConccor": StructuralCriticalityTier.TIER_2_PROGRESSIVE,
    "Efflorescence": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Rust": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Weathering": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Wetspot": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Cavity": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Hollowareas": StructuralCriticalityTier.TIER_3_SUPERFICIAL,
    "Graffiti": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
    "Drainage": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
    "JTape": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
    "PEquipment": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
    "Restformwork": StructuralCriticalityTier.TIER_4_NON_STRUCTURAL,
}


# ============================================================================
# Domain Knowledge Base: Explanations, Literature Citations, Field Actions
# ============================================================================

DEFECT_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Spalling": {
        "mechanism": (
            "Spalling represents the physical loss of surface concrete cover, often driven "
            "by internal expansion pressure from corroding reinforcing steel, freeze-thaw "
            "cycles, or localized impact. When cover concrete breaks away, the underlying "
            "reinforcement or deeper matrix becomes directly exposed to atmospheric chlorides."
        ),
        "literature": (
            "AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1080 - Delamination/Spall/Patched Area); "
            "FHWA BIRM (2012) Section 7.2.2; DIN 1076 Section 5.3 (Schadensgruppe Beton)."
        ),
        "actions": [
            "Perform acoustic hammer sounding around defect boundaries to map unseparated delaminations.",
            "Inspect the base of the cavity for exposed or corroding reinforcing steel bars.",
            "Measure physical cavity depth and surface area with calibrated field gauges.",
        ],
        "flags": [ReviewFlag.POSSIBLE_SPALLING_REVIEW.value],
    },
    "ExposedRebars": {
        "mechanism": (
            "Reinforcing steel visible through breached concrete cover. Indicates that "
            "passivation alkalinity has been lost and active oxidation is occurring. "
            "Ongoing corrosion reduces cross-sectional steel area, weakens bond strength, "
            "and can lead to brittle reinforcement fracture under dynamic cyclic loads."
        ),
        "literature": (
            "AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1090 - Exposed Rebar); "
            "FHWA BIRM (2012) Section 7.2.3; Eurocode 2 (EN 1992-1-1 Section 4)."
        ),
        "actions": [
            "Measure residual bar diameter with vernier caliper to calculate percentage cross-sectional loss.",
            "Determine corrosion pitting severity and length of debonded bar length.",
            "Check adjacent concrete elements for rust staining and impending spalls.",
        ],
        "flags": [ReviewFlag.REBAR_EXPOSURE_REVIEW.value],
    },
    "Crack": {
        "mechanism": (
            "Discrete linear fracture in concrete indicating tensile stress exceeding tensile "
            "capacity. Cracks create preferential pathways for water, oxygen, and deicing salt "
            "penetration into the reinforcing cage, accelerating rebar depassivation."
        ),
        "literature": (
            "AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1130 - Cracking); "
            "FIB Model Code 2010 Section 7.7.3 (durability crack limit w_lim = 0.2-0.3 mm); "
            "Eurocode 2 (EN 1992-1-1 Section 7.3)."
        ),
        "actions": [
            "Measure physical crack width using an optical crack comparator card or gauge.",
            "Determine whether crack pattern reflects flexure, shear, torsion, or plastic shrinkage.",
            "Verify whether crack is active (dynamic thermal/live load movement) or dormant.",
        ],
        "flags": [ReviewFlag.CRACK_DELAMINATION_REVIEW.value],
    },
    "ACrack": {
        "mechanism": (
            "Interconnected map / alligator cracking pattern. Often signifies alkali-silica "
            "reaction (ASR), severe thermal gradient fatigue, or surface shrinkage matrix "
            "degradation across an entire structural face."
        ),
        "literature": (
            "AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1130 - Map Cracking); "
            "FHWA BIRM (2012) Section 7.2.1."
        ),
        "actions": [
            "Map global extent of interconnected network across the concrete element.",
            "Examine for gel exudation or petrographic signs of alkali-silica reaction (ASR).",
            "Perform rebound hammer or ultrasonic pulse velocity (UPV) to check matrix integrity.",
        ],
        "flags": [ReviewFlag.CRACK_DELAMINATION_REVIEW.value],
    },
    "Rockpocket": {
        "mechanism": (
            "Severe honeycomb voiding in hardened concrete caused by improper consolidation, "
            "honeycombed aggregate segregation, or grout leakage during original construction. "
            "Reduces compressive section and provides easy ingress for water."
        ),
        "literature": (
            "AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1120 - Honeycomb); "
            "FHWA BIRM (2012) Section 7.2.4."
        ),
        "actions": [
            "Measure depth and volume of void pocket using a calibrated probe.",
            "Check whether rebar is exposed within the honeycomb void.",
            "Determine whether the void affects a critical shear or compressive load zone.",
        ],
        "flags": [ReviewFlag.HONEYCOMBING_CONCRETE_REVIEW.value],
    },
    "Bearing": {
        "mechanism": (
            "Bridge bearing assembly transmitting superstructure loads to the substructure "
            "while permitting thermal translation and rotation. Bearing distress (corrosion, "
            "elastomeric shear bulging, frozen rockers) induces unintended moments and restrained forces."
        ),
        "literature": (
            "AASHTO MBEI (2019) Element Category: Bearings (Elements 310-316); "
            "FHWA BIRM (2012) Chapter 10 (Bridge Bearings)."
        ),
        "actions": [
            "Check alignment, tilt, and expansion movement clearance against temperature curves.",
            "Examine elastomeric pads for bulging, debonding, or cracking.",
            "Verify anchor bolt tightness and clear debris/grit accumulation from bearing seat.",
        ],
        "flags": [ReviewFlag.BEARING_ARTICULATION_REVIEW.value],
    },
    "EJoint": {
        "mechanism": (
            "Superstructure expansion joint. Joint failure allows runoff water, road salts, "
            "and grit to leak onto underlying pier caps, bearings, and beam ends, triggering "
            "accelerated substructure decay."
        ),
        "literature": (
            "AASHTO MBEI (2019) Element Category: Joints (Elements 300-305); "
            "FHWA BIRM (2012) Chapter 9 (Expansion Joints)."
        ),
        "actions": [
            "Inspect joint seal watertightness and examine substructure directly below for leakage.",
            "Measure physical joint gap against design ambient temperature range.",
            "Check nosing concrete/steel for impact fractures, loose armor, or debris impaction.",
        ],
        "flags": [ReviewFlag.EXPANSION_JOINT_REVIEW.value],
    },
    "WConccor": {
        "mechanism": (
            "Weak, degraded, or corroded concrete matrix exhibiting chemical softening, "
            "carbonation depassivation, or acid attack degradation."
        ),
        "literature": (
            "AASHTO MBEI (2019) Defect 1190 (Abrasion/Wear) & Defect 1120 (Honeycomb); "
            "DIN 1076 Section 5.3."
        ),
        "actions": [
            "Conduct phenolphthalein carbonation depth test on broken concrete surfaces.",
            "Perform Schmidt rebound hammer testing to quantify compressive surface hardness.",
            "Assess depth of softened or friable paste layer.",
        ],
        "flags": [ReviewFlag.HONEYCOMBING_CONCRETE_REVIEW.value],
    },
    "Efflorescence": {
        "mechanism": (
            "Crystalline deposit of soluble salts (typically calcium carbonate) leached from "
            "concrete paste by percolating water and deposited upon surface evaporation. "
            "Confirms active internal water flow through the structural element."
        ),
        "literature": (
            "AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1120 - Efflorescence/Rust Staining); "
            "FHWA BIRM (2012) Section 7.2.5."
        ),
        "actions": [
            "Trace moisture source (deck leakage, clogged drains, or joint failure).",
            "Check whether efflorescence is accompanied by rust staining or active dripping.",
            "Inspect underside of deck slab or diaphragm for signs of internal leaching.",
        ],
        "flags": [ReviewFlag.SURFACE_MOISTURE_LEACHING_REVIEW.value],
    },
    "Rust": {
        "mechanism": (
            "Iron oxide discoloration or staining on concrete surface. Frequently indicates "
            "subsurface reinforcement corrosion leaching rust outward through concrete pore network."
        ),
        "literature": (
            "AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1120 - Efflorescence/Rust Staining); "
            "FHWA BIRM (2012) Section 7.2.3."
        ),
        "actions": [
            "Probe rust-stained area for hollow-sounding delamination planes.",
            "Determine whether rust originates from embedded rebar vs. external metallic hardware.",
            "Measure concrete cover thickness with a magnetic covermeter.",
        ],
        "flags": [ReviewFlag.SURFACE_MOISTURE_LEACHING_REVIEW.value],
    },
    "Weathering": {
        "mechanism": (
            "Superficial atmospheric erosion, biological growth (moss/lichen), or aggregate "
            "exposure from environmental weathering. Generally cosmetic in early stages."
        ),
        "literature": (
            "AASHTO MBEI (2019) Table 2.4.3-1 (Defect 1190 - Abrasion/Wear); "
            "FHWA BIRM (2012) Section 7.2.6."
        ),
        "actions": [
            "Check if aggregate exposure has progressed into active surface paste erosion.",
            "Confirm absence of cracking or delamination beneath weathered skin.",
        ],
        "flags": [],
    },
    "Wetspot": {
        "mechanism": (
            "Localized surface dampness or standing moisture. Indicates drainage failure, "
            "leaking expansion joint, or poor surface grading allowing water ponding."
        ),
        "literature": (
            "FHWA BIRM (2012) Section 7.2.5 (Moisture Intrusion)."
        ),
        "actions": [
            "Identify origin of moisture (e.g. leaking joint, scupper overflow).",
            "Verify whether water has freeze-thaw potential during sub-zero operational periods.",
        ],
        "flags": [ReviewFlag.SURFACE_MOISTURE_LEACHING_REVIEW.value],
    },
    "Cavity": {
        "mechanism": (
            "Small localized surface void, form-tie hole, or popout without significant "
            "rebar exposure or extensive delamination."
        ),
        "literature": "AASHTO MBEI (2019) Defect 1080 (Minor Void/Popout).",
        "actions": ["Measure diameter and depth to ensure it does not compromise cover."],
        "flags": [],
    },
    "Hollowareas": {
        "mechanism": (
            "Surface region exhibiting visual contour irregularity suggestive of subsurface "
            "delamination or past shallow patch repair."
        ),
        "literature": "AASHTO MBEI (2019) Defect 1080 (Delamination / Patched Area).",
        "actions": ["Perform chain dragging or percussion hammer sounding to confirm void."],
        "flags": [],
    },
    "Graffiti": {
        "mechanism": (
            "Surface aesthetic paint or marker application on concrete or steel face. "
            "Does NOT impair structural load capacity or element durability."
        ),
        "literature": "FHWA BIRM (2012) Section 2.1 (Aesthetic / Appurtenance condition).",
        "actions": [
            "Verify that surface paint does not conceal active structural cracks or spalls.",
            "Note for municipal aesthetic cleanup (non-structural priority).",
        ],
        "flags": [ReviewFlag.NON_STRUCTURAL_SURFACE_COVERAGE.value],
    },
    "Drainage": {
        "mechanism": (
            "Bridge deck scupper, downspout, or drainage pipe. Component functionality is "
            "critical to prevent deck ponding and joint overflow."
        ),
        "literature": "AASHTO MBEI (2019) Element Category: Drainage Systems.",
        "actions": ["Inspect for sediment blockages, pipe fractures, or outfall misalignment."],
        "flags": [],
    },
    "JTape": {
        "mechanism": "Joint sealing tape or membrane remnant at expansion interface.",
        "literature": "FHWA BIRM (2012) Section 9.3.",
        "actions": ["Inspect membrane adhesion and watertightness across joint gap."],
        "flags": [],
    },
    "PEquipment": {
        "mechanism": "Inspection access gear, rigging, temporary sensors, or utility conduits.",
        "literature": "FHWA BIRM (2012) Chapter 2 (Bridge Utilities and Appurtenances).",
        "actions": ["Confirm secure mounting; ensure utility supports do not damage bridge concrete."],
        "flags": [],
    },
    "Restformwork": {
        "mechanism": "Remnant timber, plywood, or tie-wires left from original construction formwork.",
        "literature": "FHWA BIRM (2012) Section 7.1.",
        "actions": ["Check for trapped moisture behind wood formwork; verify no embedded corrosion."],
        "flags": [],
    },
}


# ============================================================================
# Evaluator Interface & Stage 3 Implementation
# ============================================================================

class BaseSeverityEvaluator(ABC):
    """
    Abstract interface for severity assessment algorithms.
    Decouples visual measurement from engineering policy/risk scoring.
    """

    @abstractmethod
    def evaluate(self, measurement: ImageMeasurementResult) -> ExplainableSeverityReport:
        """
        Evaluates an ImageMeasurementResult to produce an ExplainableSeverityReport.
        """
        pass


class ExplainableVisualSeverityEvaluator(BaseSeverityEvaluator):
    """
    Stage 3 Explainable Visual Severity Assessment Evaluator.

    Implements:
    - Segregation of structural vs non-structural defect extents.
    - Observation-level Visual Concern Levels (HIGH, MODERATE, LOW, NEGLIGIBLE, NONE).
    - Explicit abstention for official bridge condition ratings (NOT_ASSESSED).
    - Structured human-in-the-loop review flags.
    - Per-finding XAI explanations with literature grounding and inspector action items.
    - Explicit missing data checklist and Stage 4 prioritization contract.
    """

    def __init__(
        self,
        class_concern_mapping: Optional[Dict[str, VisualConcernLevel]] = None,
        non_structural_classes: Optional[Set[str]] = None,
    ):
        self.concern_mapping = class_concern_mapping or CLASS_VISUAL_CONCERN_MAPPING
        self.non_structural_classes = set(non_structural_classes or NON_STRUCTURAL_CLASSES)

    def evaluate(self, measurement: ImageMeasurementResult) -> ExplainableSeverityReport:
        detected_classes = measurement.get_detected_classes()
        detected_structural = measurement.get_detected_structural_classes()
        detected_non_structural = measurement.get_detected_non_structural_classes()

        # 1. Determine Visual Concern Level (Observation-level, not certified bridge condition)
        if not detected_classes:
            overall_concern = VisualConcernLevel.NO_DEFECTS_DETECTED
        elif any(self.concern_mapping.get(c) == VisualConcernLevel.HIGH_VISUAL_CONCERN for c in detected_classes):
            overall_concern = VisualConcernLevel.HIGH_VISUAL_CONCERN
        elif any(self.concern_mapping.get(c) == VisualConcernLevel.MODERATE_VISUAL_CONCERN for c in detected_classes):
            overall_concern = VisualConcernLevel.MODERATE_VISUAL_CONCERN
        elif any(self.concern_mapping.get(c) == VisualConcernLevel.LOW_VISUAL_CONCERN for c in detected_classes):
            overall_concern = VisualConcernLevel.LOW_VISUAL_CONCERN
        else:
            overall_concern = VisualConcernLevel.NEGLIGIBLE_NON_STRUCTURAL

        # 2. Collect Review Flags
        review_flags: List[str] = []
        for c_name in detected_classes:
            kb_entry = DEFECT_KNOWLEDGE_BASE.get(c_name, {})
            for flag in kb_entry.get("flags", []):
                if flag not in review_flags:
                    review_flags.append(flag)

        # Flag extensive non-structural coverage if Graffiti or non-structural area >= 1.0%
        if measurement.non_structural_union_image_area_pct >= 1.0 or "Graffiti" in detected_classes:
            if ReviewFlag.NON_STRUCTURAL_SURFACE_COVERAGE.value not in review_flags:
                review_flags.append(ReviewFlag.NON_STRUCTURAL_SURFACE_COVERAGE.value)

        # 3. Generate Per-Finding Observations for XAI Dashboard
        observations: List[DefectVisualObservation] = []
        finding_counter = 1

        for c_name, c_summary in measurement.class_measurements.items():
            if not c_summary.detected:
                continue

            c_concern = self.concern_mapping.get(c_name, VisualConcernLevel.LOW_VISUAL_CONCERN)
            is_struct = c_name not in self.non_structural_classes
            kb_info = DEFECT_KNOWLEDGE_BASE.get(c_name, {
                "mechanism": f"Visual observation of {c_name} on bridge structure.",
                "literature": "FHWA Bridge Inspection Reference Manual (2012).",
                "actions": ["Verify visual defect condition in the field."],
                "flags": [],
            })

            for comp in c_summary.components:
                fid = f"FINDING-{c_name.upper()[:5]}-{finding_counter:03d}"

                # Construct evidence string
                evidence_parts = [
                    f"Detected {c_name} component of {comp.pixel_area} px ({comp.image_relative_area_pct:.4f}% of image)."
                ]
                if comp.roi_relative_area_pct is not None:
                    evidence_parts.append(f"ROI-relative coverage: {comp.roi_relative_area_pct:.4f}%.")
                evidence_parts.append(
                    f"Spatial centroid at ({comp.centroid_xy[0]}, {comp.centroid_xy[1]}) "
                    f"within bounding box [{comp.bbox.xmin}, {comp.bbox.ymin}, {comp.bbox.xmax}, {comp.bbox.ymax}]."
                )
                evidence_parts.append(
                    f"Model confidence: mean={comp.confidence.mean:.2f}, min={comp.confidence.min:.2f}, max={comp.confidence.max:.2f}."
                )

                if comp.shape_features.is_linear_defect and comp.shape_features.skeleton_length_pixels:
                    evidence_parts.append(
                        f"Centerline skeleton length: {comp.shape_features.skeleton_length_pixels} px, "
                        f"estimated width: {comp.shape_features.estimated_avg_width_pixels:.1f} px, "
                        f"aspect ratio: {comp.shape_features.aspect_ratio:.1f}."
                    )
                else:
                    evidence_parts.append(
                        f"Shape descriptors: circularity={comp.shape_features.circularity:.3f}, "
                        f"solidity={comp.shape_features.solidity:.3f}, aspect_ratio={comp.shape_features.aspect_ratio:.1f}."
                    )

                obs = DefectVisualObservation(
                    finding_id=fid,
                    class_name=c_name,
                    class_id=c_summary.class_id,
                    component_id=comp.component_id,
                    visual_concern_level=c_concern.value,
                    is_structural=is_struct,
                    pixel_area=comp.pixel_area,
                    image_relative_extent_pct=comp.image_relative_area_pct,
                    roi_relative_extent_pct=comp.roi_relative_area_pct,
                    bbox=comp.bbox,
                    confidence_stats=comp.confidence,
                    shape_features=comp.shape_features,
                    visual_evidence=" ".join(evidence_parts),
                    explanation=kb_info["mechanism"],
                    literature_reference=kb_info["literature"],
                    inspector_action_items=kb_info["actions"],
                    triggered_review_flags=kb_info.get("flags", []),
                )
                observations.append(obs)
                finding_counter += 1

        # 4. Compute Model Confidence Mean across detected components
        all_means = [obs.confidence_stats.mean for obs in observations]
        model_conf_mean = float(np.mean(all_means)) if all_means else None

        # 5. Build Coverage Breakdown
        coverage_breakdown = {
            "summed_class_pixel_area": measurement.summed_class_pixel_area,
            "summed_class_image_area_pct": measurement.summed_class_image_area_pct,
            "union_pixel_area": measurement.union_defect_pixel_area,
            "union_defect_pixel_area": measurement.union_defect_pixel_area,
            "union_defect_image_area_pct": measurement.union_defect_image_area_pct,
            "structural_union_pixel_area": measurement.structural_union_pixel_area,
            "structural_union_image_area_pct": measurement.structural_union_image_area_pct,
            "non_structural_union_pixel_area": measurement.non_structural_union_pixel_area,
            "non_structural_union_image_area_pct": measurement.non_structural_union_image_area_pct,
            "overlapping_defect_pixel_area": measurement.overlapping_defect_pixel_area,
            "overlapping_defect_image_area_pct": measurement.overlapping_defect_image_area_pct,
            "roi_coverage": {
                "roi_available": measurement.roi_present,
                "summed_class_roi_pixels": measurement.summed_class_roi_pixels,
                "union_defect_roi_area_pct": measurement.union_defect_roi_area_pct,
                "structural_union_roi_area_pct": measurement.structural_union_roi_area_pct,
                "non_structural_union_roi_area_pct": measurement.non_structural_union_roi_area_pct,
            },
        }

        # 6. Build Stage 4 Prioritization Contract
        stage_4_contract = {
            "ready_for_automated_prioritization": False,
            "abstention_reason": (
                "Automated maintenance priority and repair scheduling cannot be assigned "
                "from visual features alone. Required data: structural component identity, "
                "element importance, physical metric dimensions, and internal depth."
            ),
            "candidate_review_flags": review_flags,
            "primary_visual_concerns": detected_structural,
            "structural_union_coverage_pct": measurement.structural_union_image_area_pct,
            "non_structural_union_coverage_pct": measurement.non_structural_union_image_area_pct,
            "actionable_observations_count": len(observations),
        }

        return ExplainableSeverityReport(
            image_id=measurement.image_id,
            visual_concern_level=overall_concern.value,
            official_structural_condition_state="NOT_ASSESSED",
            abstention_reason=ABSTENTION_JUSTIFICATION,
            review_flags=review_flags,
            coverage_breakdown=coverage_breakdown,
            segmentation_model_confidence_mean=model_conf_mean,
            calibrated_structural_severity_confidence=None,  # Explicit abstention
            confidence_distinction_notice=CONFIDENCE_DISTINCTION_NOTICE,
            missing_data_checklist=MissingDataChecklist(),
            observations=observations,
            stage_4_prioritization_contract=stage_4_contract,
        )


# Backward compatibility alias
class LiteratureGroundedSeverityEvaluator(ExplainableVisualSeverityEvaluator):
    """
    Backward-compatible evaluator class that delegates to ExplainableVisualSeverityEvaluator.
    Maintains existing Stage 2 method signatures and property accesses.
    """
    def __init__(self, tier_mapping: Optional[Any] = None):
        super().__init__()


# Backward compatibility type alias
SeverityAssessmentResult = ExplainableSeverityReport
