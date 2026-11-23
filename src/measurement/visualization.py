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

from src.measurement.schema import ImageMeasurementResult

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
        measurement: ImageMeasurementResult,
        probability_maps: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
        alpha: float = 0.45,
        show_bboxes: bool = True,
        show_labels: bool = True,
        show_roi: bool = True,
        show_banner: bool = True,
    ) -> np.ndarray:
        """
        Renders defect masks, bounding boxes, labels, and ROI boundaries onto image.

        Args:
            image_rgb: Original RGB image array (H, W, 3) in uint8 [0, 255].
            measurement: ImageMeasurementResult from DefectMeasurementPipeline.
            probability_maps: Optional (19, H, W) float array to render exact masks.
            roi_mask: Optional (H, W) ROI mask.
            alpha: Transparency weight for mask blending (0.0 to 1.0).
            show_bboxes: Whether to render bounding boxes around components.
            show_labels: Whether to render text callouts.
            show_roi: Whether to render ROI boundary.
            show_banner: Whether to render top summary header banner.

        Returns:
            RGB array of the overlaid image (H, W, 3).
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape (H, W, 3), got {image_rgb.shape}")

        h, w = image_rgb.shape[:2]
        canvas = image_rgb.copy()
        mask_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
        has_any_mask = False

        # 1. Render defect masks
        if probability_maps is not None:
            # Render directly from thresholded probability maps
            for class_name, summary in measurement.class_measurements.items():
                if not summary.detected:
                    continue
                cid = summary.class_id
                color = self.get_class_color(cid)

                # Threshold for this class
                if isinstance(measurement.threshold, dict):
                    tau = float(measurement.threshold.get(class_name, 0.5))
                else:
                    tau = float(measurement.threshold)

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
            # Draw distinct cyan boundary (R=0, G=255, B=255)
            cv2.drawContours(canvas, contours, -1, (0, 255, 255), thickness=max(2, self.line_thickness))

        # 3. Render bounding boxes and component annotations
        for class_name, summary in measurement.class_measurements.items():
            if not summary.detected:
                continue

            cid = summary.class_id
            color = self.get_class_color(cid)

            for comp in summary.components:
                xmin, ymin, xmax, ymax = comp.bbox.xyxy

                # Draw bounding box
                if show_bboxes:
                    cv2.rectangle(
                        canvas,
                        (xmin, ymin),
                        (xmax, ymax),
                        color,
                        thickness=self.line_thickness,
                    )

                # Draw annotation tag
                if show_labels:
                    # Construct text label
                    # Example: "Crack #1 | 0.08% | Conf: 0.91"
                    tag_parts = [f"{class_name} #{comp.component_id}"]
                    tag_parts.append(f"{comp.image_relative_area_pct:.2f}%")
                    tag_parts.append(f"C:{comp.confidence.mean:.2f}")

                    if comp.shape_features.is_linear_defect and comp.shape_features.skeleton_length_pixels:
                        tag_parts.append(f"L:{comp.shape_features.skeleton_length_pixels}px")

                    label_text = " | ".join(tag_parts)

                    # Text background pill for contrast
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label_text, font, self.font_scale, thickness=1
                    )

                    label_ymin = max(text_h + 4, ymin - 4)
                    pill_top_left = (xmin, label_ymin - text_h - 4)
                    pill_bottom_right = (min(w, xmin + text_w + 6), label_ymin + baseline)

                    # Dark background box
                    cv2.rectangle(canvas, pill_top_left, pill_bottom_right, (20, 20, 20), -1)
                    # Colored border indicator
                    cv2.rectangle(canvas, pill_top_left, pill_bottom_right, color, 1)

                    # White text
                    cv2.putText(
                        canvas,
                        label_text,
                        (xmin + 3, label_ymin - 2),
                        font,
                        self.font_scale,
                        (255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        # 4. Top header summary banner
        if show_banner:
            banner_h = max(32, int(h * 0.045))
            banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
            banner[:] = (30, 30, 30)  # Dark gray

            det_classes = measurement.get_detected_classes()
            classes_str = ", ".join(det_classes) if det_classes else "None (No defects detected)"

            roi_str = f" | ROI Defect: {measurement.total_defect_roi_area_pct:.2f}%" if measurement.roi_present else ""
            summary_text = (
                f"Defects: {len(det_classes)} classes | "
                f"Area: {measurement.total_defect_image_area_pct:.2f}% img{roi_str} | "
                f"Classes: [{classes_str}]"
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            banner_font_scale = min(0.6, max(0.4, w / 1600.0))
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

            # Stack banner on top of canvas
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
