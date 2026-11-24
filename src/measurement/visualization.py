"""
Module: src.measurement.visualization
Description:
    Publication-quality visualization and overlay generation for bridge defect
    measurements.

    Renders:
    - Multi-class semi-transparent color masks.
    - Class-specific bounding boxes with component tags.
    - Area percentages, confidence ratings, and skeleton lengths.
    - Structural Region-of-Interest (ROI) boundary contours.
    - Informative summary metadata banner.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.measurement.schema import ImageMeasurementResult, NON_STRUCTURAL_CLASSES

# Distinct 19-class colormap (RGB format, 0-255)
# Grounded in standard qualitative color palettes
_TAB20 = plt.cm.tab20(np.linspace(0, 1, 20))[:19]
CLASS_COLOR_PALETTE_RGB = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in _TAB20]


class DefectVisualizer:
    """
    Renders visual overlays of defect measurements onto bridge inspection images.
    """

    def __init__(
        self,
        color_palette: Optional[List[Tuple[int, int, int]]] = None,
        font_scale: float = 0.5,
        line_thickness: int = 2,
    ):
        """
        Args:
            color_palette: List of 19 RGB tuples. Defaults to Matplotlib Tab20.
            font_scale: Base font scale for OpenCV putText.
            line_thickness: Bounding box line thickness.
        """
        self.color_palette = color_palette or CLASS_COLOR_PALETTE_RGB
        self.font_scale = font_scale
        self.line_thickness = line_thickness

    def get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Returns (R, G, B) tuple for given class_id."""
        return self.color_palette[class_id % len(self.color_palette)]

    def render_overlay(
        self,
        image_rgb: np.ndarray,
        measurement: Union[ImageMeasurementResult, Dict[str, Any]],
        probability_maps: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
        alpha: float = 0.45,
        show_masks: bool = True,
        show_bboxes: bool = False,
        show_labels: bool = False,
        show_roi: bool = True,
        show_banner: bool = True,
        highlight_finding_id: Optional[str] = None,
        highlight_class: Optional[str] = None,
    ) -> np.ndarray:
        """
        Renders defect masks, bounding boxes, labels, and ROI boundaries onto image.
        Supports selective highlighting of a specific finding ID or class.
        By default, renders clean class-color segmentation masks without component labels
        or cluttering bounding boxes, keeping visual inspection clear and legible.

        Args:
            image_rgb: Original RGB image array (H, W, 3) in uint8 [0, 255].
            measurement: ImageMeasurementResult or measurement_dict from pipeline.
            probability_maps: Optional (19, H, W) float array to render exact masks.
            roi_mask: Optional (H, W) ROI mask.
            alpha: Transparency weight for mask blending (0.0 to 1.0).
            show_masks: Whether to render class-color segmentation masks (default True).
            show_bboxes: Whether to render bounding boxes around unselected components (default False).
            show_labels: Whether to render text callouts (default False for clean display).
            show_roi: Whether to render ROI boundary.
            show_banner: Whether to render top summary header banner.
            highlight_finding_id: Optional finding ID to highlight prominently (e.g. 'FINDING-CRACK-001').
            highlight_class: Optional class name to highlight/filter (e.g. 'Crack').

        Returns:
            RGB array of the overlaid image (H, W, 3).
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape (H, W, 3), got {image_rgb.shape}")

        h, w = image_rgb.shape[:2]
        canvas = image_rgb.copy()
        mask_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
        has_any_mask = False

        # Normalize measurement input (supports both ImageMeasurementResult and plain Dict)
        if isinstance(measurement, dict):
            class_dict = measurement.get("class_measurements", {})
            thresh_meta = measurement.get("metadata", {}).get("threshold", 0.5)
            roi_present = measurement.get("metadata", {}).get("roi", {}).get("present", False)
            roi_pct = measurement.get("metadata", {}).get("roi", {}).get("roi_relative_area_pct")
            det_classes = [c for c, d in class_dict.items() if d.get("detected", False)]
            det_struct = [c for c in det_classes if c not in NON_STRUCTURAL_CLASSES]
            det_non_struct = [c for c in det_classes if c in NON_STRUCTURAL_CLASSES]
            struct_union_pct = measurement.get("structural_union_image_area_pct", 0.0)
            non_struct_union_pct = measurement.get("non_structural_union_image_area_pct", 0.0)
            union_defect_roi_area_pct = measurement.get("union_defect_roi_area_pct")
        else:
            class_dict = measurement.class_measurements
            thresh_meta = measurement.threshold
            roi_present = measurement.roi_present
            roi_pct = measurement.roi_relative_area_pct
            det_classes = measurement.get_detected_classes()
            det_struct = measurement.get_detected_structural_classes()
            det_non_struct = measurement.get_detected_non_structural_classes()
            struct_union_pct = measurement.structural_union_image_area_pct
            non_struct_union_pct = measurement.non_structural_union_image_area_pct
            union_defect_roi_area_pct = measurement.union_defect_roi_area_pct

        is_filtered_class = bool(highlight_class and highlight_class not in ("All", "All Classes", "All Detected Classes"))

        # 1. Render defect masks
        if show_masks and probability_maps is not None:
            for class_name, summary in class_dict.items():
                is_detected = summary.get("detected", False) if isinstance(summary, dict) else summary.detected
                if not is_detected:
                    continue
                if is_filtered_class and class_name != highlight_class:
                    continue

                cid = summary.get("class_id") if isinstance(summary, dict) else summary.class_id
                if cid is None or not (0 <= cid < len(probability_maps)):
                    continue

                color = self.get_class_color(cid)

                # Threshold for this class
                if isinstance(thresh_meta, dict):
                    tau = float(thresh_meta.get(class_name, 0.5))
                else:
                    tau = float(thresh_meta)

                c_mask = probability_maps[cid] >= tau
                if np.any(c_mask):
                    has_any_mask = True
                    mask_overlay[c_mask] = color

        # Alpha blend masks with original image
        if has_any_mask:
            canvas = cv2.addWeighted(canvas, 1.0 - alpha, mask_overlay, alpha, 0)

        # 2. Render ROI boundary if present
        if show_roi and roi_mask is not None:
            roi_u8 = (roi_mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(roi_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (0, 255, 255), thickness=max(2, self.line_thickness))

        # 3. Render bounding boxes and component annotations
        finding_counter = 1
        for class_name, summary in class_dict.items():
            is_detected = summary.get("detected", False) if isinstance(summary, dict) else summary.detected
            if not is_detected:
                continue

            cid = summary.get("class_id") if isinstance(summary, dict) else summary.class_id
            color = self.get_class_color(cid) if cid is not None else (0, 255, 0)
            components = summary.get("components", []) if isinstance(summary, dict) else summary.components

            for comp in components:
                if isinstance(comp, dict):
                    b = comp.get("bbox", {})
                    xmin, ymin, xmax, ymax = b.get("xmin", 0), b.get("ymin", 0), b.get("xmax", 0), b.get("ymax", 0)
                    comp_id = comp.get("component_id", 1)
                    fid = comp.get("finding_id", f"FINDING-{class_name[:5].upper()}-{finding_counter:03d}")
                    area_pct = comp.get("image_relative_area_pct", 0.0)
                    conf_mean = comp.get("confidence", {}).get("mean", 0.0)
                    shapes = comp.get("shape_features", {})
                    is_linear = shapes.get("is_linear_defect", False)
                    skel_len = shapes.get("skeleton_length_pixels")
                else:
                    xmin, ymin, xmax, ymax = comp.bbox.xyxy
                    comp_id = comp.component_id
                    fid = getattr(comp, "finding_id", f"FINDING-{class_name[:5].upper()}-{finding_counter:03d}")
                    area_pct = comp.image_relative_area_pct
                    conf_mean = comp.confidence.mean
                    is_linear = comp.shape_features.is_linear_defect
                    skel_len = comp.shape_features.skeleton_length_pixels

                finding_counter += 1

                # Matching conditions
                is_selected_finding = bool(highlight_finding_id and (fid == highlight_finding_id or f"{class_name} #{comp_id}" == highlight_finding_id))
                matches_class_filter = (not is_filtered_class or class_name == highlight_class)

                if is_filtered_class and not matches_class_filter and not is_selected_finding:
                    continue

                # Draw bounding box
                if is_selected_finding:
                    # Prominent neon callout with dark outer halo for maximum contrast
                    cv2.rectangle(canvas, (max(0, xmin - 2), max(0, ymin - 2)), (min(w - 1, xmax + 2), min(h - 1, ymax + 2)), (0, 0, 0), thickness=5)
                    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 245, 255), thickness=3)
                elif show_bboxes:
                    box_thick = 1 if (highlight_finding_id or is_filtered_class) else self.line_thickness
                    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, thickness=box_thick)

                # Draw annotation tag: ALWAYS drawn for selected finding, hidden by default for others
                if is_selected_finding or show_labels:
                    if is_selected_finding:
                        label_text = f"★ {fid} | {class_name} #{comp_id} | {area_pct:.2f}% | Conf:{conf_mean:.2f}"
                    else:
                        tag_parts = [f"{class_name} #{comp_id}", f"{area_pct:.2f}%", f"C:{conf_mean:.2f}"]
                        if is_linear and skel_len:
                            tag_parts.append(f"L:{skel_len}px")
                        label_text = " | ".join(tag_parts)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = max(self.font_scale, 0.45) if is_selected_finding else self.font_scale
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, scale, thickness=1)

                    label_ymin = max(text_h + 4, ymin - 4)
                    pill_top_left = (xmin, label_ymin - text_h - 4)
                    pill_bottom_right = (min(w, xmin + text_w + 6), label_ymin + baseline)

                    pill_border = (0, 245, 255) if is_selected_finding else color
                    cv2.rectangle(canvas, pill_top_left, pill_bottom_right, (20, 20, 20), -1)
                    cv2.rectangle(canvas, pill_top_left, pill_bottom_right, pill_border, 2 if is_selected_finding else 1)

                    text_color = (0, 245, 255) if is_selected_finding else (255, 255, 255)
                    cv2.putText(
                        canvas,
                        label_text,
                        (xmin + 3, label_ymin - 2),
                        font,
                        scale,
                        text_color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        # 4. Top header summary banner
        if show_banner:
            banner_h = max(36, int(h * 0.05))
            banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
            banner[:] = (25, 25, 25)

            classes_str = ", ".join(det_classes) if det_classes else "None"
            roi_str = f" | ROI: {roi_pct:.2f}%" if roi_present and roi_pct is not None else ""

            if highlight_finding_id:
                summary_text = f"🎯 Highlight Target: {highlight_finding_id} | Total Detected: {len(det_classes)} classes ({classes_str}){roi_str}"
            elif is_filtered_class:
                summary_text = f"🎯 Class Focus: {highlight_class} | Total Detected: {len(det_classes)} classes ({classes_str}){roi_str}"
            elif det_non_struct and det_struct:
                summary_text = (
                    f"Structural: {struct_union_pct:.2f}% {det_struct} | "
                    f"Non-Struct: {non_struct_union_pct:.2f}% {det_non_struct}{roi_str}"
                )
            elif det_struct:
                summary_text = f"Structural: {struct_union_pct:.2f}% {det_struct}{roi_str}"
            elif det_non_struct:
                summary_text = f"Non-Structural: {non_struct_union_pct:.2f}% {det_non_struct} (No structural defects)"
            else:
                summary_text = "No defect patterns detected above threshold (Clean Inspection Scan)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            banner_font_scale = min(0.55, max(0.38, w / 1400.0))
            cv2.putText(
                banner,
                summary_text,
                (12, int(banner_h * 0.65)),
                font,
                banner_font_scale,
                (240, 240, 240),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            canvas = np.vstack([banner, canvas])

        return canvas

    def save_overlay(
        self,
        save_path: Union[str, Path],
        image_rgb: np.ndarray,
        measurement: ImageMeasurementResult,
        probability_maps: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
        **render_kwargs,
    ) -> Path:
        """
        Renders overlay and saves image as PNG or JPEG.
        """
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rendered_rgb = self.render_overlay(
            image_rgb=image_rgb,
            measurement=measurement,
            probability_maps=probability_maps,
            roi_mask=roi_mask,
            **render_kwargs,
        )

        bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return out_path
