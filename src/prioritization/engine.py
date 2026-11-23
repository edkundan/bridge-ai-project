"""
Module: src.prioritization.engine
Description:
    Pipeline orchestrator for Stage 4: Maintenance Prioritization Foundation.
    Loads Stage 3 explainable severity outputs, parses optional verified
    engineering context, applies the selected decision policy, and exports
    the dual outputs (Inspection Review Queue & Engineering Maintenance Priority).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.measurement.schema import ExplainableSeverityReport
from src.prioritization.policies import (
    BasePrioritizationPolicy,
    ProductionAbstentionPolicy,
    ResearchDemonstrationPolicy,
)
from src.prioritization.schema import (
    ElementCriticality,
    EngineeringMaintenancePriority,
    HumanOverrideRecord,
    InspectionReviewQueue,
    Stage4PrioritizationResult,
    VerifiedEngineeringContext,
)


class MaintenancePrioritizationPipeline:
    """
    Orchestration engine for Stage 4 Maintenance Prioritization.
    """

    def __init__(
        self,
        policy: Optional[BasePrioritizationPolicy] = None,
        use_research_demo: bool = False,
    ):
        if policy is not None:
            self.policy = policy
        elif use_research_demo:
            self.policy = ResearchDemonstrationPolicy(allow_simulation=True)
        else:
            self.policy = ProductionAbstentionPolicy()

    def run(
        self,
        stage3_input: Union[str, Path, Dict[str, Any], ExplainableSeverityReport],
        engineering_context: Optional[Union[str, Path, Dict[str, Any], VerifiedEngineeringContext]] = None,
        human_overrides: Optional[List[Union[Dict[str, Any], HumanOverrideRecord]]] = None,
    ) -> Stage4PrioritizationResult:
        """
        Executes Stage 4 prioritization.

        Args:
            stage3_input: ExplainableSeverityReport instance, or path/dict containing Stage 3 JSON.
            engineering_context: Optional verified engineering context instance, dict, or path.
            human_overrides: Optional list of human override records or dicts.

        Returns:
            Stage4PrioritizationResult containing dual outputs and audit trail.
        """
        # Parse Stage 3 input
        report = self._parse_stage3_report(stage3_input)

        # Parse optional engineering context
        ctx = self._parse_engineering_context(engineering_context)

        # Parse optional human overrides
        overrides = self._parse_human_overrides(human_overrides)

        # Evaluate policy
        return self.policy.evaluate(
            stage3_report=report,
            engineering_context=ctx,
            human_overrides=overrides,
        )

    def _parse_stage3_report(
        self,
        inp: Union[str, Path, Dict[str, Any], ExplainableSeverityReport],
    ) -> ExplainableSeverityReport:
        if isinstance(inp, ExplainableSeverityReport):
            return inp

        if isinstance(inp, (str, Path)):
            with open(inp, "r") as f:
                data = json.load(f)
        elif isinstance(inp, dict):
            data = inp
        else:
            raise TypeError(f"Unsupported stage3_input type: {type(inp)}")

        # If data is a dictionary, extract necessary fields to reconstruct report
        # If it was saved via to_dict(), reconstruct cleanly
        from src.measurement.schema import (
            BoundingBox,
            ConfidenceStats,
            DefectVisualObservation,
            MissingDataChecklist,
            ShapeFeatures,
        )

        img_id = data.get("image_id", "unknown")
        vsa = data.get("visual_severity_assessment", {})
        concern = vsa.get("visual_concern_level", data.get("visual_concern_level", "NO_DEFECTS_DETECTED"))
        flags = vsa.get("review_flags", data.get("review_flags", []))
        abstention = vsa.get("abstention_reason", data.get("abstention_reason", ""))

        coverage = data.get("coverage_breakdown", {})
        conf_contract = data.get("confidence_contract", {})
        mean_conf = conf_contract.get("segmentation_model_confidence_mean")

        # Parse observations
        raw_obs = data.get("observations", [])
        observations: List[DefectVisualObservation] = []
        for o in raw_obs:
            bbox_dict = o.get("bbox", {})
            bbox = BoundingBox(
                xmin=bbox_dict.get("xmin", 0),
                ymin=bbox_dict.get("ymin", 0),
                xmax=bbox_dict.get("xmax", 0),
                ymax=bbox_dict.get("ymax", 0),
            )
            c_dict = o.get("confidence_stats", {})
            conf_stats = ConfidenceStats(
                mean=c_dict.get("mean", 0.0),
                min=c_dict.get("min", 0.0),
                max=c_dict.get("max", 0.0),
                median=c_dict.get("median", 0.0),
                std=c_dict.get("std", 0.0),
            )
            s_dict = o.get("shape_features", {})
            shape_feats = ShapeFeatures(
                perimeter_pixels=s_dict.get("perimeter_pixels", 0.0),
                circularity=s_dict.get("circularity", 0.0),
                aspect_ratio=s_dict.get("aspect_ratio", 1.0),
                solidity=s_dict.get("solidity", 1.0),
                extent=s_dict.get("extent", 1.0),
                orientation_deg=s_dict.get("orientation_deg", 0.0),
                is_linear_defect=s_dict.get("is_linear_defect", False),
                skeleton_length_pixels=s_dict.get("skeleton_length_pixels"),
                estimated_avg_width_pixels=s_dict.get("estimated_avg_width_pixels"),
            )

            obs = DefectVisualObservation(
                finding_id=o.get("finding_id", "FINDING-UNKNOWN"),
                class_name=o.get("class_name", "Unknown"),
                class_id=o.get("class_id", 0),
                component_id=o.get("component_id", 1),
                visual_concern_level=o.get("visual_concern_level", concern),
                is_structural=o.get("is_structural", True),
                pixel_area=o.get("pixel_area", 0),
                image_relative_extent_pct=o.get("image_relative_extent_pct", 0.0),
                roi_relative_extent_pct=o.get("roi_relative_extent_pct"),
                bbox=bbox,
                confidence_stats=conf_stats,
                shape_features=shape_feats,
                visual_evidence=o.get("visual_evidence", ""),
                explanation=o.get("explanation", ""),
                literature_reference=o.get("literature_reference", ""),
                inspector_action_items=o.get("inspector_action_items", []),
                triggered_review_flags=o.get("triggered_review_flags", []),
            )
            observations.append(obs)

        return ExplainableSeverityReport(
            image_id=img_id,
            visual_concern_level=concern,
            official_structural_condition_state="NOT_ASSESSED",
            abstention_reason=abstention,
            review_flags=flags,
            coverage_breakdown=coverage,
            segmentation_model_confidence_mean=mean_conf,
            calibrated_structural_severity_confidence=None,
            missing_data_checklist=MissingDataChecklist(),
            observations=observations,
            stage_4_prioritization_contract=data.get("stage_4_prioritization_contract", {}),
        )

    def _parse_engineering_context(
        self,
        ctx: Optional[Union[str, Path, Dict[str, Any], VerifiedEngineeringContext]],
    ) -> Optional[VerifiedEngineeringContext]:
        if ctx is None:
            return None
        if isinstance(ctx, VerifiedEngineeringContext):
            return ctx
        if isinstance(ctx, (str, Path)):
            with open(ctx, "r") as f:
                d = json.load(f)
        elif isinstance(ctx, dict):
            d = ctx
        else:
            raise TypeError(f"Unsupported context type: {type(ctx)}")

        crit_str = d.get("element_criticality", "UNKNOWN")
        try:
            crit = ElementCriticality(crit_str)
        except ValueError:
            crit = ElementCriticality.UNKNOWN

        return VerifiedEngineeringContext(
            bridge_id=d.get("bridge_id"),
            structural_component=d.get("structural_component"),
            element_criticality=crit,
            calibrated_dimensions=d.get("calibrated_dimensions"),
            defect_depth_mm=d.get("defect_depth_mm"),
            professional_engineer_notes=d.get("professional_engineer_notes"),
            is_verified_by_pe=d.get("is_verified_by_pe", False),
            verification_timestamp=d.get("verification_timestamp"),
        )

    def _parse_human_overrides(
        self,
        overrides: Optional[List[Union[Dict[str, Any], HumanOverrideRecord]]],
    ) -> List[HumanOverrideRecord]:
        if not overrides:
            return []
        parsed = []
        for o in overrides:
            if isinstance(o, HumanOverrideRecord):
                parsed.append(o)
            elif isinstance(o, dict):
                parsed.append(
                    HumanOverrideRecord(
                        override_id=o.get("override_id", "OV-001"),
                        finding_id=o["finding_id"],
                        inspector_id=o.get("inspector_id", "INSPECTOR-PE"),
                        timestamp_utc=o.get("timestamp_utc", ""),
                        original_urgency=o.get("original_urgency", ""),
                        overridden_urgency=o["overridden_urgency"],
                        justification=o.get("justification", ""),
                    )
                )
        return parsed

    def save_results(
        self,
        result: Stage4PrioritizationResult,
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """
        Exports machine-readable JSON outputs into the target output directory.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        full_json_path = out_dir / f"{result.image_id}_stage4_prioritization.json"
        with open(full_json_path, "w") as f:
            f.write(result.to_json(indent=2))

        queue_json_path = out_dir / f"{result.image_id}_inspection_review_queue.json"
        with open(queue_json_path, "w") as f:
            json.dump(result.inspection_review_queue.to_dict(), f, indent=2)

        maint_json_path = out_dir / f"{result.image_id}_engineering_maintenance_priority.json"
        with open(maint_json_path, "w") as f:
            json.dump(result.engineering_maintenance_priority.to_dict(), f, indent=2)

        return {
            "stage4_full": full_json_path,
            "review_queue": queue_json_path,
            "maintenance_priority": maint_json_path,
        }
