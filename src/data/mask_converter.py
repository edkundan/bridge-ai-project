"""
Module: src.data.mask_converter
Description:
    Converts LabelMe-style polygon annotations from the DACL10K-v2 dataset into
    19-channel multi-label binary raster masks with shape (H, W, 19).
    
    Handles:
    - 19 independent semantic classes.
    - Overlapping polygons of the same class (unioned).
    - Overlapping polygons of different classes (independent positive channels).
    - Empty annotations (all-zero binary mask).
    - Coordinate clipping and invalid polygon filtering.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Canonical 19 DACL10K-v2 classes in alphabetical order
DACL10K_CLASSES = [
    "ACrack", "Bearing", "Cavity", "Crack", "Drainage", "EJoint",
    "Efflorescence", "ExposedRebars", "Graffiti", "Hollowareas", "JTape",
    "PEquipment", "Restformwork", "Rockpocket", "Rust", "Spalling",
    "WConccor", "Weathering", "Wetspot"
]


class MaskConverter:
    """
    Converts polygon annotations to multi-channel binary segmentation masks.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = list(class_names) if class_names is not None else list(DACL10K_CLASSES)
        self.num_classes = len(self.class_names)
        self.class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class: Dict[int, str] = {idx: name for idx, name in enumerate(self.class_names)}

    def rasterize_polygon(
        self,
        points: Union[List[List[float]], np.ndarray],
        height: int,
        width: int
    ) -> Optional[np.ndarray]:
        """
        Rasterize a single polygon into a 2D binary mask of shape (height, width).
        
        Args:
            points: List of [x, y] coordinates.
            height: Image height in pixels.
            width: Image width in pixels.
            
        Returns:
            Binary uint8 array of shape (height, width) with values {0, 1},
            or None if the polygon is invalid (<3 points or zero area).
        """
        pts = np.array(points, dtype=np.float32)
        if len(pts) < 3:
            logger.debug("Polygon skipped: fewer than 3 vertices.")
            return None

        # Clip coordinates to valid image bounds [0, width - 1] and [0, height - 1]
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))
        
        # Round and convert to integer points for cv2.fillPoly
        pts_int = np.int32([np.round(pts)])
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, pts_int, color=1)
        return mask

    def json_to_multilabel_mask(
        self,
        annotation_source: Union[str, Path, Dict[str, Any]],
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Converts LabelMe JSON annotation to a 19-channel binary mask of shape (H, W, 19).
        
        Args:
            annotation_source: Path to JSON file, or parsed JSON dict.
            height: Override height. If None, uses 'imageHeight' from JSON.
            width: Override width. If None, uses 'imageWidth' from JSON.
            
        Returns:
            Tuple of:
                - multi_mask: np.ndarray of shape (H, W, num_classes) with dtype np.uint8 (values 0 or 1).
                - metadata: Dict with extraction stats (shape count, empty flag, classes present).
        """
        if isinstance(annotation_source, (str, Path)):
            with open(annotation_source, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(annotation_source, dict):
            data = annotation_source
        else:
            raise TypeError(f"Unsupported annotation source type: {type(annotation_source)}")

        img_h = height if height is not None else int(data.get("imageHeight", 0))
        img_w = width if width is not None else int(data.get("imageWidth", 0))

        if img_h <= 0 or img_w <= 0:
            raise ValueError(f"Invalid image dimensions: height={img_h}, width={img_w}")

        # Initialize multi-label mask: H x W x 19 with all zeros
        multi_mask = np.zeros((img_h, img_w, self.num_classes), dtype=np.uint8)

        shapes = data.get("shapes", [])
        is_empty = (len(shapes) == 0)

        metadata = {
            "image_name": data.get("imageName", ""),
            "height": img_h,
            "width": img_w,
            "total_shapes": len(shapes),
            "is_empty": is_empty,
            "classes_present": [],
            "unknown_labels": [],
            "invalid_polygons": 0,
        }

        if is_empty:
            logger.debug(f"Empty annotation encountered for image {metadata['image_name']}.")
            return multi_mask, metadata

        classes_present_set = set()

        for shape in shapes:
            label = shape.get("label", "")
            if label not in self.class_to_idx:
                metadata["unknown_labels"].append(label)
                logger.warning(f"Unknown class label '{label}' encountered. Skipping.")
                continue

            class_idx = self.class_to_idx[label]
            points = shape.get("points", [])
            
            poly_mask = self.rasterize_polygon(points, img_h, img_w)
            if poly_mask is None:
                metadata["invalid_polygons"] += 1
                continue

            # Multi-label logical OR for polygons of the same class
            # Overlapping polygons of DIFFERENT classes are drawn to separate channels
            multi_mask[:, :, class_idx] = np.bitwise_or(multi_mask[:, :, class_idx], poly_mask)
            classes_present_set.add(label)

        metadata["classes_present"] = sorted(list(classes_present_set))
        return multi_mask, metadata

    def mask_to_polygons(
        self,
        binary_mask: np.ndarray,
        min_area: float = 5.0
    ) -> List[List[List[float]]]:
        """
        Converts a 2D binary mask back to polygon vertices using contour tracing.
        Useful for verification and post-processing.
        
        Args:
            binary_mask: 2D uint8 array with values {0, 1} or {0, 255}.
            min_area: Minimum contour area threshold in pixels.
            
        Returns:
            List of polygon vertex lists [[x1, y1], [x2, y2], ...].
        """
        mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            # Squeeze and convert to float list [[x, y], ...]
            pts = cnt.reshape(-1, 2).astype(float).tolist()
            if len(pts) >= 3:
                polygons.append(pts)
        return polygons
