"""
Module: src.measurement.schema
Description:
    Standard dataclasses, types, and JSON serialization schemas for
    Stage 2: Defect Measurement and Severity Assessment Foundation.

    Ensures consistent, machine-readable representations of:
    - Bounding boxes (xyxy and xywh)
    - Pixel areas and relative area percentages (image-relative and ROI-relative)
    - Connected component counts and individual component features
    - Pixel-based shape and elongation descriptors (perimeter, circularity, solidity, skeleton length)
    - Statistical confidence distributions
    - Per-class and whole-image summaries
    - Literature-grounded severity assessment results with explicit calibration disclaimers
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


CALIBRATION_DISCLAIMER = (
    "OPTICAL MEASUREMENT NOTICE: All lengths, widths, perimeters, and areas "
    "are reported strictly in pixel-domain units (px, px^2) or image/ROI-relative "
    "dimensionless percentages (%). Physical metric measurements (mm, cm, m) "
    "cannot be claimed without metric camera calibration, sensor distance measurements, "
    "or physical reference scale markers in the image plane."
)

SEVERITY_VALIDATION_DISCLAIMER = (
    "ENGINEERING SEVERITY DISCLAIMER: This severity assessment is an uncalibrated "
    "research prototype grounded in literature principles (AASHTO MBEI, FHWA BIRM, "
    "DIN 1076, FIB Model Code). It provides visual evidence categorization only and "
    "does NOT constitute a certified structural safety rating, remaining service life "
    "prediction, or replacement for certified structural bridge engineer inspection."
)


@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box coordinates.
    Provides both [ymin, xmin, ymax, xmax] and [xmin, ymin, width, height].
    """
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def xyxy(self) -> List[int]:
        """[xmin, ymin, xmax, ymax] format (standard for OpenCV/Matplotlib)."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def yxyx(self) -> List[int]:
        """[ymin, xmin, ymax, xmax] format (standard for TensorFlow/PyTorch object detection)."""
        return [self.ymin, self.xmin, self.ymax, self.xmax]

    @property
    def xywh(self) -> List[int]:
        """[xmin, ymin, width, height] format (standard for COCO)."""
        return [self.xmin, self.ymin, self.width, self.height]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "width": self.width,
            "height": self.height,
            "xyxy": self.xyxy,
            "xywh": self.xywh,
            "bbox_area_pixels": self.area,
        }


@dataclass
class ConfidenceStats:
    """
    Confidence statistics across positive defect pixels for a given component or class.
    """
    mean: float
    min: float
    max: float
    median: float
    std: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": round(float(self.mean), 4),
            "min": round(float(self.min), 4),
            "max": round(float(self.max), 4),
            "median": round(float(self.median), 4),
            "std": round(float(self.std), 4),
        }


@dataclass
class ShapeFeatures:
    """
    Pixel-domain shape and morphology descriptors.
    All dimensions are in pixel units.
    """
    perimeter_pixels: float
    circularity: float  # 4 * pi * area / (perimeter^2), 1.0 for perfect circle
    aspect_ratio: float  # major_axis / minor_axis >= 1.0
    solidity: float  # area / convex_hull_area in [0.0, 1.0]
    extent: float  # area / bounding_box_area in [0.0, 1.0]
    orientation_deg: float  # principal axis angle in degrees [-90, 90]
    is_linear_defect: bool = False
    skeleton_length_pixels: Optional[int] = None
    estimated_avg_width_pixels: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perimeter_pixels": round(float(self.perimeter_pixels), 2),
            "circularity": round(float(self.circularity), 4),
            "aspect_ratio": round(float(self.aspect_ratio), 4),
            "solidity": round(float(self.solidity), 4),
            "extent": round(float(self.extent), 4),
            "orientation_deg": round(float(self.orientation_deg), 2),
            "is_linear_defect": self.is_linear_defect,
            "skeleton_length_pixels": self.skeleton_length_pixels,
            "estimated_avg_width_pixels": (
                round(float(self.estimated_avg_width_pixels), 2)
                if self.estimated_avg_width_pixels is not None
                else None
            ),
        }


@dataclass
class DefectComponentMeasurement:
    """
    Detailed measurement of a single connected component of a defect class.
    """
    component_id: int
    class_id: int
    class_name: str
    pixel_area: int
    image_relative_area_pct: float
    centroid_xy: Tuple[float, float]
    bbox: BoundingBox
    confidence: ConfidenceStats
    shape_features: ShapeFeatures
    roi_relative_area_pct: Optional[float] = None
    pixels_inside_roi: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "pixel_area": self.pixel_area,
            "image_relative_area_pct": round(float(self.image_relative_area_pct), 5),
            "roi_relative_area_pct": (
                round(float(self.roi_relative_area_pct), 5)
                if self.roi_relative_area_pct is not None
                else None
            ),
            "pixels_inside_roi": self.pixels_inside_roi,
            "centroid_xy": [round(float(self.centroid_xy[0]), 2), round(float(self.centroid_xy[1]), 2)],
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence.to_dict(),
            "shape_features": self.shape_features.to_dict(),
        }


@dataclass
class ClassMeasurementSummary:
    """
    Summary of defect measurements for a specific class across all its components.
    """
    class_id: int
    class_name: str
    detected: bool
    component_count: int
    total_pixel_area: int
    image_relative_area_pct: float
    confidence: Optional[ConfidenceStats] = None
    roi_relative_area_pct: Optional[float] = None
    total_pixels_inside_roi: Optional[int] = None
    components: List[DefectComponentMeasurement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "detected": self.detected,
            "component_count": self.component_count,
            "total_pixel_area": self.total_pixel_area,
            "image_relative_area_pct": round(float(self.image_relative_area_pct), 5),
            "roi_relative_area_pct": (
                round(float(self.roi_relative_area_pct), 5)
                if self.roi_relative_area_pct is not None
                else None
            ),
            "total_pixels_inside_roi": self.total_pixels_inside_roi,
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "components": [c.to_dict() for c in self.components],
        }


@dataclass
class ImageMeasurementResult:
    """
    Complete measurement result for a single image, encompassing all 19 DACL10K classes.
    """
    image_id: str
    image_height: int
    image_width: int
    threshold: Union[float, Dict[str, float]]
    min_component_area_pixels: int
    roi_present: bool
    roi_pixel_area: Optional[int]
    roi_relative_area_pct: Optional[float]
    total_detected_classes: int
    total_components_count: int
    total_defect_pixel_area: int
    total_defect_image_area_pct: float
    total_defect_roi_area_pct: Optional[float]
    class_measurements: Dict[str, ClassMeasurementSummary]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    disclaimer: str = CALIBRATION_DISCLAIMER

    @property
    def total_image_pixels(self) -> int:
        return self.image_height * self.image_width

    def get_detected_classes(self) -> List[str]:
        """Returns list of class names with >= 1 detected component."""
        return [
            name for name, summary in self.class_measurements.items()
            if summary.detected
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "image_id": self.image_id,
                "image_dimensions": {
                    "height": self.image_height,
                    "width": self.image_width,
                    "total_pixels": self.total_image_pixels,
                },
                "timestamp_utc": self.timestamp,
                "threshold": self.threshold,
                "min_component_area_pixels": self.min_component_area_pixels,
                "roi": {
                    "present": self.roi_present,
                    "roi_pixel_area": self.roi_pixel_area,
                    "roi_relative_area_pct": (
                        round(float(self.roi_relative_area_pct), 5)
                        if self.roi_relative_area_pct is not None
                        else None
                    ),
                },
                "disclaimer": self.disclaimer,
            },
            "summary": {
                "total_detected_classes": self.total_detected_classes,
                "detected_class_names": self.get_detected_classes(),
                "total_components_count": self.total_components_count,
                "total_defect_pixel_area": self.total_defect_pixel_area,
                "total_defect_image_area_pct": round(float(self.total_defect_image_area_pct), 5),
                "total_defect_roi_area_pct": (
                    round(float(self.total_defect_roi_area_pct), 5)
                    if self.total_defect_roi_area_pct is not None
                    else None
                ),
            },
            "class_measurements": {
                name: summary.to_dict()
                for name, summary in self.class_measurements.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
