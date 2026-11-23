"""
Module: src.measurement.geometry
Description:
    Geometric and morphological measurement algorithms for bridge defect regions.
    
    Provides:
    - Robust morphological skeletonization via iterative erosion & opening.
    - Connected component analysis with configurable minimum area filtering.
    - Pixel-level shape descriptors: perimeter, circularity, aspect ratio, solidity, extent.
    - Crack / linear defect centerline length and estimated average pixel width.
    - Distributional confidence statistics (mean, min, max, median, std).
    - Image-relative and ROI-relative area calculations.
"""

import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from src.measurement.schema import (
    BoundingBox,
    ClassMeasurementSummary,
    ConfidenceStats,
    DefectComponentMeasurement,
    ShapeFeatures,
)


def compute_morphological_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Computes a 1-pixel-wide topological centerline (skeleton) of a binary mask
    using standard iterative morphological erosion and opening with a cross structuring element.

    Args:
        binary_mask: 2D uint8 binary array where defect pixels are > 0.

    Returns:
        2D uint8 binary array containing only the skeleton pixels (values in {0, 255}).
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {binary_mask.shape}")

    # Ensure mask is strict binary uint8 (0 or 255)
    img = (binary_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(img) == 0:
        return np.zeros_like(img)

    skel = np.zeros(img.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = img.copy()

    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        subset = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, subset)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # If the skeleton ended up empty for a tiny region, keep at least one central pixel
    if cv2.countNonZero(skel) == 0 and cv2.countNonZero(img) > 0:
        skel = (cv2.morphologyEx(img, cv2.MORPH_CLOSE, element) > 0).astype(np.uint8) * 255

    return skel


def compute_confidence_stats(
    probability_map_2d: np.ndarray,
    mask_2d: np.ndarray,
) -> ConfidenceStats:
    """
    Computes statistical distribution of predicted probabilities over positive defect pixels.

    Args:
        probability_map_2d: 2D float array in [0.0, 1.0].
        mask_2d: 2D boolean or uint8 mask (> 0 indicates defect).

    Returns:
        ConfidenceStats dataclass instance.
    """
    defect_pixels = probability_map_2d[mask_2d > 0]
    if len(defect_pixels) == 0:
        return ConfidenceStats(mean=0.0, min=0.0, max=0.0, median=0.0, std=0.0)

    return ConfidenceStats(
        mean=float(np.mean(defect_pixels)),
        min=float(np.min(defect_pixels)),
        max=float(np.max(defect_pixels)),
        median=float(np.median(defect_pixels)),
        std=float(np.std(defect_pixels)),
    )


def extract_shape_features(
    component_mask: np.ndarray,
    pixel_area: int,
    bbox: BoundingBox,
    is_linear_defect: bool = False,
) -> ShapeFeatures:
    """
    Extracts shape, orientation, and morphological descriptors for a single connected component.

    Args:
        component_mask: 2D uint8 mask (0 or 255) containing only the single component.
        pixel_area: Total foreground pixels in this component.
        bbox: Precomputed BoundingBox of the component.
        is_linear_defect: If True (or if elongated), extracts skeleton length and average width.

    Returns:
        ShapeFeatures dataclass instance.
    """
    if pixel_area <= 0:
        return ShapeFeatures(
            perimeter_pixels=0.0,
            circularity=0.0,
            aspect_ratio=1.0,
            solidity=0.0,
            extent=0.0,
            orientation_deg=0.0,
            is_linear_defect=is_linear_defect,
            skeleton_length_pixels=0 if is_linear_defect else None,
            estimated_avg_width_pixels=0.0 if is_linear_defect else None,
        )

    # 1. External contour
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return ShapeFeatures(
            perimeter_pixels=0.0,
            circularity=0.0,
            aspect_ratio=1.0,
            solidity=1.0,
            extent=1.0,
            orientation_deg=0.0,
            is_linear_defect=is_linear_defect,
            skeleton_length_pixels=pixel_area if is_linear_defect else None,
            estimated_avg_width_pixels=1.0 if is_linear_defect else None,
        )

    # Find the largest contour in this component
    main_contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(main_contour, closed=True))

    # 2. Circularity / Isoperimetric Quotient: 4 * pi * Area / (Perimeter^2)
    # 1.0 for a smooth circle; very low (<0.1) for jagged, branched, or elongated cracks.
    circularity = 0.0
    if perimeter > 0:
        circularity = min(1.0, (4.0 * math.pi * pixel_area) / (perimeter ** 2))

    # 3. Minimum Area Bounding Box (Oriented Bounding Box)
    # Provides true aspect ratio regardless of diagonal orientation
    aspect_ratio = 1.0
    orientation_deg = 0.0
    if len(main_contour) >= 5:
        # minAreaRect returns ((cx, cy), (w, h), angle)
        rect = cv2.minAreaRect(main_contour)
        dim1, dim2 = rect[1]
        major_axis = max(dim1, dim2)
        minor_axis = min(dim1, dim2)
        if minor_axis > 0:
            aspect_ratio = max(1.0, major_axis / minor_axis)
        orientation_deg = float(rect[2])
    else:
        # Fallback to axis-aligned box
        w, h = bbox.width, bbox.height
        if min(w, h) > 0:
            aspect_ratio = max(1.0, max(w, h) / min(w, h))

    # 4. Solidity: Area / Convex Hull Area
    # Measures boundary concavity and branching
    hull = cv2.convexHull(main_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = 1.0
    if hull_area > 0:
        solidity = min(1.0, max(0.0, pixel_area / hull_area))

    # 5. Extent: Area / Axis-Aligned Bounding Box Area
    bbox_area = bbox.area
    extent = 1.0
    if bbox_area > 0:
        extent = min(1.0, max(0.0, pixel_area / bbox_area))

    # 6. Linear defect features (crack centerline skeleton and average width)
    # Trigger if explicitly flagged as linear class OR if highly elongated (aspect_ratio >= 3.0)
    should_extract_skeleton = is_linear_defect or (aspect_ratio >= 3.0 and pixel_area >= 20)
    skeleton_length = None
    estimated_avg_width = None

    if should_extract_skeleton:
        # Crop to bounding box with 1-px padding to speed up skeletonization
        h_img, w_img = component_mask.shape
        ymin = max(0, bbox.ymin - 1)
        ymax = min(h_img, bbox.ymax + 1)
        xmin = max(0, bbox.xmin - 1)
        xmax = min(w_img, bbox.xmax + 1)

        crop = component_mask[ymin:ymax, xmin:xmax]
        skel_crop = compute_morphological_skeleton(crop)
        skeleton_length = int(np.sum(skel_crop > 0))

        # Ensure skeleton length is at least 1 pixel for non-empty components
        if skeleton_length == 0 and pixel_area > 0:
            skeleton_length = max(1, int(round(max(bbox.width, bbox.height))))

        estimated_avg_width = float(pixel_area) / float(max(1, skeleton_length))

    return ShapeFeatures(
        perimeter_pixels=perimeter,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        solidity=solidity,
        extent=extent,
        orientation_deg=orientation_deg,
        is_linear_defect=should_extract_skeleton,
        skeleton_length_pixels=skeleton_length,
        estimated_avg_width_pixels=estimated_avg_width,
    )


def analyze_connected_components(
    binary_mask: np.ndarray,
    probability_map: np.ndarray,
    class_id: int,
    class_name: str,
    image_shape: Tuple[int, int],
    roi_mask: Optional[np.ndarray] = None,
    min_area_pixels: int = 10,
    is_linear_class: bool = False,
    return_retained_mask: bool = False,
) -> Union[
    Tuple[List[DefectComponentMeasurement], ClassMeasurementSummary],
    Tuple[List[DefectComponentMeasurement], ClassMeasurementSummary, np.ndarray],
]:
    """
    Performs 8-connectivity connected component analysis on a binary class mask.
    Filters out noise regions smaller than min_area_pixels.
    Calculates detailed component measurements and class-level summary.

    Args:
        binary_mask: 2D uint8 array (0 or 1).
        probability_map: 2D float32 array in [0.0, 1.0].
        class_id: Canonical integer ID of the class (0-18).
        class_name: Canonical class name string.
        image_shape: (H, W) of the full image.
        roi_mask: Optional 2D boolean/uint8 ROI mask.
        min_area_pixels: Minimum pixel area to retain as a valid defect component.
        is_linear_class: Whether this class is inherently linear (Crack, ACrack, etc.).
        return_retained_mask: If True, also returns the 2D boolean mask of retained pixels.

    Returns:
        Tuple of (List[DefectComponentMeasurement], ClassMeasurementSummary)
        or (List[DefectComponentMeasurement], ClassMeasurementSummary, np.ndarray) if return_retained_mask=True.
    """
    total_image_pixels = image_shape[0] * image_shape[1]
    roi_pixels = int(np.sum(roi_mask > 0)) if roi_mask is not None else None

    # Handle all-zero mask
    total_positive_pixels = int(np.sum(binary_mask > 0))
    if total_positive_pixels == 0:
        empty_summary = ClassMeasurementSummary(
            class_id=class_id,
            class_name=class_name,
            detected=False,
            component_count=0,
            total_pixel_area=0,
            image_relative_area_pct=0.0,
            confidence=None,
            roi_relative_area_pct=0.0 if roi_mask is not None else None,
            total_pixels_inside_roi=0 if roi_mask is not None else None,
            components=[],
        )
        if return_retained_mask:
            return [], empty_summary, np.zeros(image_shape, dtype=bool)
        return [], empty_summary

    # Ensure mask is uint8 {0, 255} for OpenCV
    mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255

    # Run 8-way connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    component_measurements: List[DefectComponentMeasurement] = []
    filtered_total_pixel_area = 0
    filtered_roi_pixel_area = 0

    # Labels 1 .. num_labels-1 (0 is background)
    comp_counter = 1
    for label_idx in range(1, num_labels):
        comp_area = int(stats[label_idx, cv2.CC_STAT_AREA])

        # Filter tiny noise components
        if comp_area < min_area_pixels:
            continue

        xmin = int(stats[label_idx, cv2.CC_STAT_LEFT])
        ymin = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        xmax = xmin + width
        ymax = ymin + height

        bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        cx = float(centroids[label_idx, 0])
        cy = float(centroids[label_idx, 1])

        # Isolate this component mask
        comp_mask = (labels == label_idx).astype(np.uint8) * 255

        # Image-relative percentage
        img_rel_pct = (comp_area / float(total_image_pixels)) * 100.0

        # ROI calculations
        roi_rel_pct = None
        comp_roi_pixels = None
        if roi_mask is not None:
            comp_roi_pixels = int(np.sum((labels == label_idx) & (roi_mask > 0)))
            if roi_pixels is not None and roi_pixels > 0:
                roi_rel_pct = (comp_roi_pixels / float(roi_pixels)) * 100.0
            else:
                roi_rel_pct = 0.0
            filtered_roi_pixel_area += comp_roi_pixels

        # Confidence stats for this component
        conf = compute_confidence_stats(probability_map, labels == label_idx)

        # Shape descriptors
        shape_feats = extract_shape_features(
            comp_mask, comp_area, bbox, is_linear_defect=is_linear_class
        )

        measurement = DefectComponentMeasurement(
            component_id=comp_counter,
            class_id=class_id,
            class_name=class_name,
            pixel_area=comp_area,
            image_relative_area_pct=img_rel_pct,
            roi_relative_area_pct=roi_rel_pct,
            pixels_inside_roi=comp_roi_pixels,
            centroid_xy=(cx, cy),
            bbox=bbox,
            confidence=conf,
            shape_features=shape_feats,
        )

        component_measurements.append(measurement)
        filtered_total_pixel_area += comp_area
        comp_counter += 1

    # Class-level summary across retained components
    is_detected = len(component_measurements) > 0
    total_img_pct = (filtered_total_pixel_area / float(total_image_pixels)) * 100.0

    class_roi_rel_pct = None
    if roi_mask is not None and roi_pixels is not None and roi_pixels > 0:
        class_roi_rel_pct = (filtered_roi_pixel_area / float(roi_pixels)) * 100.0
    elif roi_mask is not None:
        class_roi_rel_pct = 0.0

    # Overall class confidence over all retained components
    if is_detected:
        retained_mask = np.isin(
            labels,
            [
                label_idx
                for label_idx in range(1, num_labels)
                if stats[label_idx, cv2.CC_STAT_AREA] >= min_area_pixels
            ],
        )
        class_conf = compute_confidence_stats(probability_map, retained_mask)
    else:
        retained_mask = np.zeros(image_shape, dtype=bool)
        class_conf = None

    class_summary = ClassMeasurementSummary(
        class_id=class_id,
        class_name=class_name,
        detected=is_detected,
        component_count=len(component_measurements),
        total_pixel_area=filtered_total_pixel_area,
        image_relative_area_pct=total_img_pct,
        roi_relative_area_pct=class_roi_rel_pct,
        total_pixels_inside_roi=filtered_roi_pixel_area if roi_mask is not None else None,
        confidence=class_conf,
        components=component_measurements,
    )

    if return_retained_mask:
        return component_measurements, class_summary, retained_mask
    return component_measurements, class_summary
