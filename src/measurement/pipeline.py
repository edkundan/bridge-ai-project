"""
Module: src.measurement.pipeline
Description:
    Core pipeline for Stage 2: Defect Measurement Foundation.
    
    Transforms 19-channel segmentation probabilities into structured,
    machine-readable geometric measurements, spatial bounding boxes,
    confidence distributions, and image/ROI-relative metrics.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.data.mask_converter import DACL10K_CLASSES
from src.measurement.geometry import analyze_connected_components
from src.measurement.schema import (
    CALIBRATION_DISCLAIMER,
    NON_STRUCTURAL_CLASSES,
    ClassMeasurementSummary,
    ImageMeasurementResult,
)

logger = logging.getLogger(__name__)

# Classes with predominantly linear / elongated morphology
DEFAULT_LINEAR_CLASSES = ["Crack", "ACrack", "Efflorescence"]


class DefectMeasurementPipeline:
    """
    Modular defect measurement pipeline for multi-label bridge defect segmentation.

    Accepts 19-channel probability maps, applies thresholding, and extracts
    rigorous pixel-domain defect measurements.
    """

    def __init__(
        self,
        threshold: Union[float, Dict[str, float]] = 0.5,
        min_component_area_pixels: int = 10,
        class_names: Optional[List[str]] = None,
        linear_classes: Optional[List[str]] = None,
    ):
        """
        Args:
            threshold: Probability threshold in [0.0, 1.0]. Can be a single float
                       applied across all 19 channels (default 0.5), or a dictionary
                       mapping class names to per-class thresholds.
            min_component_area_pixels: Minimum pixel area for a connected component
                                       to be retained (filters single-pixel noise).
            class_names: Canonical list of classes. Defaults to DACL10K_CLASSES (19 classes).
            linear_classes: List of class names treated as linear defects for skeletonization.
        """
        self.class_names = list(class_names) if class_names is not None else list(DACL10K_CLASSES)
        self.num_classes = len(self.class_names)
        self.min_component_area_pixels = max(1, int(min_component_area_pixels))
        self.linear_classes = set(linear_classes or DEFAULT_LINEAR_CLASSES)

        # Validate threshold configuration
        if isinstance(threshold, (int, float)):
            val = float(threshold)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Threshold must be in [0.0, 1.0], got {val}")
            self.threshold = val
            self.threshold_map = {name: val for name in self.class_names}
        elif isinstance(threshold, dict):
            self.threshold = threshold
            self.threshold_map = {}
            for name in self.class_names:
                t = float(threshold.get(name, 0.5))
                if not 0.0 <= t <= 1.0:
                    raise ValueError(f"Threshold for {name} must be in [0.0, 1.0], got {t}")
                self.threshold_map[name] = t
        else:
            raise TypeError(f"Threshold must be float or dict, got {type(threshold)}")

    def _normalize_probabilities_tensor(
        self,
        probabilities: Union[torch.Tensor, np.ndarray],
    ) -> np.ndarray:
        """
        Normalizes input probability tensor or array to shape (num_classes, H, W)
        with values clamped to float32 [0.0, 1.0].
        """
        if isinstance(probabilities, torch.Tensor):
            probs = probabilities.detach().cpu().float().numpy()
        elif isinstance(probabilities, np.ndarray):
            probs = probabilities.astype(np.float32)
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(probabilities)}")

        # Squeeze batch dimension if present: (1, C, H, W) -> (C, H, W)
        if probs.ndim == 4 and probs.shape[0] == 1:
            probs = probs.squeeze(0)

        # Handle channels-last: (H, W, C) -> (C, H, W)
        if probs.ndim == 3 and probs.shape[2] == self.num_classes and probs.shape[0] != self.num_classes:
            probs = np.transpose(probs, (2, 0, 1))

        if probs.ndim != 3 or probs.shape[0] != self.num_classes:
            raise ValueError(
                f"Expected probability map of shape ({self.num_classes}, H, W), "
                f"got shape {probs.shape}"
            )

        # Clamp values to valid probability range
        probs = np.clip(probs, 0.0, 1.0)
        return probs

    def measure_probabilities(
        self,
        probabilities: Union[torch.Tensor, np.ndarray],
        image_id: str = "unknown",
        roi_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> ImageMeasurementResult:
        """
        Measures defects from 19-channel probability maps.

        Args:
            probabilities: Array or Tensor of shape (19, H, W), (1, 19, H, W), or (H, W, 19).
            image_id: Identifier string for tracking.
            roi_mask: Optional binary mask of shape (H, W) indicating the structural
                      region of interest (e.g. pier, girder, deck slab).

        Returns:
            ImageMeasurementResult dataclass instance.
        """
        probs = self._normalize_probabilities_tensor(probabilities)
        num_classes, h, w = probs.shape
        total_image_pixels = h * w

        # Process ROI mask if provided
        processed_roi = None
        roi_pixel_area = None
        roi_rel_pct = None
        if roi_mask is not None:
            if isinstance(roi_mask, torch.Tensor):
                roi_np = roi_mask.detach().cpu().numpy()
            else:
                roi_np = np.asarray(roi_mask)

            if roi_np.ndim == 3 and roi_np.shape[0] == 1:
                roi_np = roi_np.squeeze(0)
            elif roi_np.ndim == 3 and roi_np.shape[2] == 1:
                roi_np = roi_np.squeeze(2)

            if roi_np.shape != (h, w):
                raise ValueError(
                    f"ROI mask shape {roi_np.shape} does not match probability map shape ({h}, {w})"
                )

            processed_roi = (roi_np > 0).astype(np.uint8)
            roi_pixel_area = int(np.sum(processed_roi))
            roi_rel_pct = (roi_pixel_area / float(total_image_pixels)) * 100.0

        # Apply thresholds per class to produce binary masks and track retained masks
        class_measurements: Dict[str, ClassMeasurementSummary] = {}
        retained_masks = np.zeros((self.num_classes, h, w), dtype=bool)
        total_components_count = 0
        summed_class_pixel_area = 0
        summed_class_roi_pixels = 0 if processed_roi is not None else None
        detected_classes_count = 0

        # Process each class channel independently (supports overlapping classes)
        for class_idx, class_name in enumerate(self.class_names):
            tau = self.threshold_map[class_name]
            channel_probs = probs[class_idx]
            channel_binary = (channel_probs >= tau).astype(np.uint8)

            is_linear = class_name in self.linear_classes

            comp_measurements, class_summary, c_retained_mask = analyze_connected_components(
                binary_mask=channel_binary,
                probability_map=channel_probs,
                class_id=class_idx,
                class_name=class_name,
                image_shape=(h, w),
                roi_mask=processed_roi,
                min_area_pixels=self.min_component_area_pixels,
                is_linear_class=is_linear,
                return_retained_mask=True,
            )

            class_measurements[class_name] = class_summary
            retained_masks[class_idx] = c_retained_mask

            if class_summary.detected:
                detected_classes_count += 1
                total_components_count += class_summary.component_count
                summed_class_pixel_area += class_summary.total_pixel_area
                if processed_roi is not None and class_summary.total_pixels_inside_roi is not None:
                    summed_class_roi_pixels += class_summary.total_pixels_inside_roi

        # -------------------------------------------------------------------------
        # Area quantification: Summed vs Union vs Structural vs Non-structural
        # -------------------------------------------------------------------------
        summed_class_img_pct = (summed_class_pixel_area / float(total_image_pixels)) * 100.0

        # Overall union mask (pixels with >= 1 detected class)
        union_mask = np.any(retained_masks, axis=0)
        union_pixel_area = int(np.sum(union_mask))
        union_img_pct = (union_pixel_area / float(total_image_pixels)) * 100.0

        # Structural vs non-structural partition
        structural_indices = [
            i for i, name in enumerate(self.class_names)
            if name not in NON_STRUCTURAL_CLASSES
        ]
        non_structural_indices = [
            i for i, name in enumerate(self.class_names)
            if name in NON_STRUCTURAL_CLASSES
        ]

        if structural_indices:
            structural_mask = np.any(retained_masks[structural_indices], axis=0)
            structural_union_pixel_area = int(np.sum(structural_mask))
        else:
            structural_mask = np.zeros((h, w), dtype=bool)
            structural_union_pixel_area = 0
        structural_union_img_pct = (structural_union_pixel_area / float(total_image_pixels)) * 100.0

        if non_structural_indices:
            non_structural_mask = np.any(retained_masks[non_structural_indices], axis=0)
            non_structural_union_pixel_area = int(np.sum(non_structural_mask))
        else:
            non_structural_mask = np.zeros((h, w), dtype=bool)
            non_structural_union_pixel_area = 0
        non_structural_union_img_pct = (non_structural_union_pixel_area / float(total_image_pixels)) * 100.0

        # Overlap mask (pixels with >= 2 classes active simultaneously)
        overlap_mask = np.sum(retained_masks, axis=0) >= 2
        overlap_pixel_area = int(np.sum(overlap_mask))
        overlap_img_pct = (overlap_pixel_area / float(total_image_pixels)) * 100.0

        # ROI relative union calculations
        union_roi_pct = None
        structural_roi_pct = None
        non_structural_roi_pct = None
        if processed_roi is not None and roi_pixel_area is not None and roi_pixel_area > 0:
            roi_bool = processed_roi > 0
            union_roi_pct = (float(np.sum(union_mask & roi_bool)) / float(roi_pixel_area)) * 100.0
            structural_roi_pct = (float(np.sum(structural_mask & roi_bool)) / float(roi_pixel_area)) * 100.0
            non_structural_roi_pct = (float(np.sum(non_structural_mask & roi_bool)) / float(roi_pixel_area)) * 100.0
        elif processed_roi is not None:
            union_roi_pct = 0.0
            structural_roi_pct = 0.0
            non_structural_roi_pct = 0.0

        return ImageMeasurementResult(
            image_id=str(image_id),
            image_height=h,
            image_width=w,
            threshold=self.threshold,
            min_component_area_pixels=self.min_component_area_pixels,
            roi_present=processed_roi is not None,
            roi_pixel_area=roi_pixel_area,
            roi_relative_area_pct=roi_rel_pct,
            total_detected_classes=detected_classes_count,
            total_components_count=total_components_count,
            summed_class_pixel_area=summed_class_pixel_area,
            summed_class_image_area_pct=summed_class_img_pct,
            union_defect_pixel_area=union_pixel_area,
            union_defect_image_area_pct=union_img_pct,
            structural_union_pixel_area=structural_union_pixel_area,
            structural_union_image_area_pct=structural_union_img_pct,
            non_structural_union_pixel_area=non_structural_union_pixel_area,
            non_structural_union_image_area_pct=non_structural_union_img_pct,
            overlapping_defect_pixel_area=overlap_pixel_area,
            overlapping_defect_image_area_pct=overlap_img_pct,
            summed_class_roi_pixels=summed_class_roi_pixels,
            union_defect_roi_area_pct=union_roi_pct,
            structural_union_roi_area_pct=structural_roi_pct,
            non_structural_union_roi_area_pct=non_structural_roi_pct,
            class_measurements=class_measurements,
            timestamp=datetime.now(timezone.utc).isoformat(),
            disclaimer=CALIBRATION_DISCLAIMER,
        )

    def measure_from_binary_masks(
        self,
        binary_masks: Union[torch.Tensor, np.ndarray],
        probability_maps: Optional[Union[torch.Tensor, np.ndarray]] = None,
        image_id: str = "ground_truth",
        roi_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> ImageMeasurementResult:
        """
        Convenience method to measure directly from binary masks (e.g. ground truth
        annotations or pre-thresholded predictions). If probability_maps are omitted,
        binary masks are used as probabilities (1.0 for defect, 0.0 for background).
        """
        if isinstance(binary_masks, torch.Tensor):
            masks_np = binary_masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(binary_masks)

        if masks_np.ndim == 4 and masks_np.shape[0] == 1:
            masks_np = masks_np.squeeze(0)
        elif masks_np.ndim == 3 and masks_np.shape[2] == self.num_classes and masks_np.shape[0] != self.num_classes:
            masks_np = np.transpose(masks_np, (2, 0, 1))

        if probability_maps is not None:
            probs = probability_maps
        else:
            probs = (masks_np > 0).astype(np.float32)

        return self.measure_probabilities(
            probabilities=probs,
            image_id=image_id,
            roi_mask=roi_mask,
        )
