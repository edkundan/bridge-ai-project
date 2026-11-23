"""
Unit tests for Stage 2: Defect Measurement and Severity Assessment Foundation.
Verifies:
- Configurable thresholding and multi-channel probability handling
- Overlapping class support without channel interference
- Bounding box, area, centroid, and confidence stats correctness
- Connected component segmentation and noise filtering
- Image-relative and ROI-relative percentage calculations
- Shape features: circularity, aspect ratio, solidity, skeleton length
- JSON schema validity and serialization
- Visualizer overlay rendering
- Literature-grounded severity assessment and disclaimer contracts
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.mask_converter import DACL10K_CLASSES
from src.measurement import (
    CALIBRATION_DISCLAIMER,
    SEVERITY_VALIDATION_DISCLAIMER,
    DefectMeasurementPipeline,
    DefectVisualizer,
    LiteratureGroundedSeverityEvaluator,
    StructuralCriticalityTier,
)
from src.measurement.geometry import (
    compute_confidence_stats,
    compute_morphological_skeleton,
    extract_shape_features,
)
from src.measurement.schema import BoundingBox


class TestDefectMeasurement(unittest.TestCase):
    """Test suite for defect measurement and severity foundation."""

    def setUp(self):
        self.h = 256
        self.w = 256
        self.total_pixels = self.h * self.w
        self.num_classes = len(DACL10K_CLASSES)
        self.pipeline = DefectMeasurementPipeline(
            threshold=0.5,
            min_component_area_pixels=10,
        )

    def test_canonical_classes_order_preserved(self):
        """Verify pipeline preserves all 19 DACL10K canonical classes in exact order."""
        self.assertEqual(self.pipeline.num_classes, 19)
        self.assertEqual(self.pipeline.class_names, list(DACL10K_CLASSES))
        self.assertEqual(self.pipeline.class_names[0], "ACrack")
        self.assertEqual(self.pipeline.class_names[3], "Crack")
        self.assertEqual(self.pipeline.class_names[7], "ExposedRebars")
        self.assertEqual(self.pipeline.class_names[15], "Spalling")
        self.assertEqual(self.pipeline.class_names[16], "WConccor")

    def test_empty_mask_handling(self):
        """Verify that all-zero or below-threshold probabilities return empty result without crashing."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        result = self.pipeline.measure_probabilities(probs, image_id="empty_test")

        self.assertEqual(result.total_detected_classes, 0)
        self.assertEqual(result.total_components_count, 0)
        self.assertEqual(result.total_defect_pixel_area, 0)
        self.assertEqual(result.total_defect_image_area_pct, 0.0)
        self.assertEqual(len(result.get_detected_classes()), 0)
        self.assertIn("OPTICAL MEASUREMENT NOTICE", result.disclaimer)

        for name, summary in result.class_measurements.items():
            self.assertFalse(summary.detected)
            self.assertEqual(summary.component_count, 0)
            self.assertEqual(summary.total_pixel_area, 0)
            self.assertEqual(len(summary.components), 0)

    def test_single_component_measurement_correctness(self):
        """Verify exact bounding box, area, area %, and confidence statistics for a known defect."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        # Create a rectangular defect for Spalling (idx 15): 20 rows x 30 cols = 600 pixels
        ymin, ymax = 40, 60
        xmin, xmax = 50, 80
        spall_idx = DACL10K_CLASSES.index("Spalling")
        probs[spall_idx, ymin:ymax, xmin:xmax] = 0.85

        result = self.pipeline.measure_probabilities(probs, image_id="single_spall")

        self.assertEqual(result.total_detected_classes, 1)
        self.assertEqual(result.get_detected_classes(), ["Spalling"])
        self.assertEqual(result.total_components_count, 1)
        self.assertEqual(result.total_defect_pixel_area, 600)

        expected_pct = (600 / float(self.total_pixels)) * 100.0
        self.assertAlmostEqual(result.total_defect_image_area_pct, expected_pct, places=4)

        spall_summary = result.class_measurements["Spalling"]
        self.assertTrue(spall_summary.detected)
        self.assertEqual(spall_summary.component_count, 1)
        self.assertEqual(spall_summary.total_pixel_area, 600)

        comp = spall_summary.components[0]
        self.assertEqual(comp.component_id, 1)
        self.assertEqual(comp.class_name, "Spalling")
        self.assertEqual(comp.pixel_area, 600)
        self.assertEqual(comp.bbox.xmin, xmin)
        self.assertEqual(comp.bbox.ymin, ymin)
        self.assertEqual(comp.bbox.xmax, xmax)
        self.assertEqual(comp.bbox.ymax, ymax)
        self.assertEqual(comp.bbox.width, 30)
        self.assertEqual(comp.bbox.height, 20)
        self.assertEqual(comp.bbox.xyxy, [50, 40, 80, 60])
        self.assertEqual(comp.bbox.xywh, [50, 40, 30, 20])
        self.assertEqual(comp.bbox.yxyx, [40, 50, 60, 80])

        # Centroid should be center of rect: (64.5, 49.5)
        self.assertAlmostEqual(comp.centroid_xy[0], 64.5, places=0)
        self.assertAlmostEqual(comp.centroid_xy[1], 49.5, places=0)

        # Confidence stats: all pixels are 0.85
        self.assertAlmostEqual(comp.confidence.mean, 0.85, places=2)
        self.assertAlmostEqual(comp.confidence.min, 0.85, places=2)
        self.assertAlmostEqual(comp.confidence.max, 0.85, places=2)
        self.assertAlmostEqual(comp.confidence.median, 0.85, places=2)

    def test_multiple_connected_components_same_class(self):
        """Verify two disjoint defect regions of the same class produce two distinct components."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        spall_idx = DACL10K_CLASSES.index("Spalling")
        # Region 1: 10 x 15 = 150 px
        probs[spall_idx, 20:30, 20:35] = 0.80
        # Region 2: 25 x 20 = 500 px (disjoint)
        probs[spall_idx, 100:125, 150:170] = 0.90

        result = self.pipeline.measure_probabilities(probs, image_id="two_spalls")

        self.assertEqual(result.total_detected_classes, 1)
        self.assertEqual(result.total_components_count, 2)
        self.assertEqual(result.total_defect_pixel_area, 650)

        spall_summary = result.class_measurements["Spalling"]
        self.assertEqual(spall_summary.component_count, 2)
        areas = sorted([c.pixel_area for c in spall_summary.components])
        self.assertEqual(areas, [150, 500])

    def test_overlapping_classes_independence(self):
        """Verify multi-label overlapping classes at the exact same spatial location are both measured."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        spall_idx = DACL10K_CLASSES.index("Spalling")
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")

        # Large spalling patch (50 x 50)
        probs[spall_idx, 100:150, 100:150] = 0.90
        # Exposed rebars running inside the spalling region (10 x 40)
        probs[rebar_idx, 120:130, 105:145] = 0.95

        result = self.pipeline.measure_probabilities(probs, image_id="overlapping_defects")

        self.assertEqual(result.total_detected_classes, 2)
        self.assertIn("Spalling", result.get_detected_classes())
        self.assertIn("ExposedRebars", result.get_detected_classes())

        self.assertEqual(result.class_measurements["Spalling"].total_pixel_area, 2500)
        self.assertEqual(result.class_measurements["ExposedRebars"].total_pixel_area, 400)

    def test_noise_filtering_min_component_area(self):
        """Verify that components smaller than min_component_area_pixels are filtered out."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        crack_idx = DACL10K_CLASSES.index("Crack")

        # Isolated 2x2 noise blob = 4 px (should be filtered when min_area=10)
        probs[crack_idx, 10:12, 10:12] = 0.99
        # Valid crack = 25 px (should be retained)
        probs[crack_idx, 50:75, 50:51] = 0.85

        result = self.pipeline.measure_probabilities(probs, image_id="noise_test")

        crack_summary = result.class_measurements["Crack"]
        self.assertEqual(crack_summary.component_count, 1)
        self.assertEqual(crack_summary.total_pixel_area, 25)

    def test_roi_mask_calculations(self):
        """Verify image-relative vs ROI-relative percentage calculations with a structural element mask."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        spall_idx = DACL10K_CLASSES.index("Spalling")
        # Defect area: 20 x 50 = 1000 px
        probs[spall_idx, 50:70, 50:100] = 0.90

        # Structural ROI (e.g. concrete pier column): 100 x 100 = 10,000 px
        roi_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        roi_mask[30:130, 30:130] = 1  # 10,000 px
        # The defect [50:70, 50:100] lies entirely inside the ROI

        result = self.pipeline.measure_probabilities(
            probs, image_id="roi_test", roi_mask=roi_mask
        )

        self.assertTrue(result.roi_present)
        self.assertEqual(result.roi_pixel_area, 10000)
        expected_roi_rel_pct = (10000 / float(self.total_pixels)) * 100.0
        self.assertAlmostEqual(result.roi_relative_area_pct, expected_roi_rel_pct, places=4)

        # Defect relative to image: 1000 / 65536 = ~1.5258%
        expected_img_pct = (1000 / float(self.total_pixels)) * 100.0
        self.assertAlmostEqual(result.total_defect_image_area_pct, expected_img_pct, places=4)

        # Defect relative to ROI: 1000 / 10000 = 10.0%
        self.assertAlmostEqual(result.total_defect_roi_area_pct, 10.0, places=4)

        spall_comp = result.class_measurements["Spalling"].components[0]
        self.assertAlmostEqual(spall_comp.roi_relative_area_pct, 10.0, places=4)
        self.assertEqual(spall_comp.pixels_inside_roi, 1000)

    def test_crack_skeleton_and_shape_features(self):
        """Verify skeletonization and shape descriptors for a synthetic crack."""
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        # Vertical crack with thickness 3 and length 80
        mask[40:120, 100:103] = 255
        pixel_area = int(np.sum(mask > 0))  # 80 * 3 = 240 px

        bbox = BoundingBox(xmin=100, ymin=40, xmax=103, ymax=120)
        shape_feats = extract_shape_features(mask, pixel_area, bbox, is_linear_defect=True)

        self.assertTrue(shape_feats.is_linear_defect)
        self.assertIsNotNone(shape_feats.skeleton_length_pixels)
        self.assertIsNotNone(shape_feats.estimated_avg_width_pixels)
        # Skeleton length should be close to 80
        self.assertGreater(shape_feats.skeleton_length_pixels, 60)
        self.assertLess(shape_feats.skeleton_length_pixels, 100)
        # Estimated width should be around 3.0 (240 / ~80)
        self.assertAlmostEqual(shape_feats.estimated_avg_width_pixels, 3.0, delta=0.8)
        # Aspect ratio should be high (~80/3 = 26)
        self.assertGreater(shape_feats.aspect_ratio, 10.0)

    def test_json_schema_serialization_and_validity(self):
        """Verify serialization produces strict valid JSON with all required metadata."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")
        probs[rebar_idx, 30:60, 30:40] = 0.92

        result = self.pipeline.measure_probabilities(probs, image_id="json_schema_check")
        json_str = result.to_json(indent=2)
        parsed = json.loads(json_str)

        # Validate top-level keys
        self.assertIn("metadata", parsed)
        self.assertIn("summary", parsed)
        self.assertIn("class_measurements", parsed)

        # Validate metadata
        meta = parsed["metadata"]
        self.assertEqual(meta["image_id"], "json_schema_check")
        self.assertEqual(meta["image_dimensions"]["height"], 256)
        self.assertEqual(meta["image_dimensions"]["width"], 256)
        self.assertIn("disclaimer", meta)
        self.assertIn("OPTICAL MEASUREMENT NOTICE", meta["disclaimer"])

        # Validate summary
        summary = parsed["summary"]
        self.assertEqual(summary["total_detected_classes"], 1)
        self.assertEqual(summary["detected_class_names"], ["ExposedRebars"])

        # Validate class measurements
        rebar_data = parsed["class_measurements"]["ExposedRebars"]
        self.assertTrue(rebar_data["detected"])
        self.assertEqual(len(rebar_data["components"]), 1)
        comp = rebar_data["components"][0]
        self.assertIn("bbox", comp)
        self.assertIn("xyxy", comp["bbox"])
        self.assertIn("xywh", comp["bbox"])
        self.assertIn("confidence", comp)
        self.assertIn("shape_features", comp)

    def test_literature_grounded_severity_evaluator(self):
        """Verify severity evaluator categorizes critical tiers correctly and flags disclosures."""
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        # Add a Tier 1 defect: ExposedRebars
        rebar_idx = DACL10K_CLASSES.index("ExposedRebars")
        probs[rebar_idx, 50:70, 50:70] = 0.95

        # Add a Tier 2 defect: Crack
        crack_idx = DACL10K_CLASSES.index("Crack")
        probs[crack_idx, 100:150, 100:102] = 0.88

        result = self.pipeline.measure_probabilities(probs, image_id="severity_eval_test")
        evaluator = LiteratureGroundedSeverityEvaluator()
        severity = evaluator.evaluate(result)

        self.assertEqual(severity.highest_criticality_tier, StructuralCriticalityTier.TIER_1_CRITICAL.value)
        self.assertIn(StructuralCriticalityTier.TIER_1_CRITICAL.value, severity.detected_tiers)
        self.assertIn(StructuralCriticalityTier.TIER_2_PROGRESSIVE.value, severity.detected_tiers)

        # Check required validation parameters
        self.assertFalse(severity.is_calibrated)
        self.assertTrue(severity.requires_expert_validation)
        self.assertGreater(len(severity.missing_engineering_parameters), 3)
        self.assertIn("AASHTO", " ".join(severity.literature_grounding))
        self.assertIn("ENGINEERING SEVERITY DISCLAIMER", severity.disclaimer)

        # Test dictionary export
        sev_dict = severity.to_dict()
        self.assertEqual(sev_dict["provisional_criticality_tier"], "TIER_1_CRITICAL")
        self.assertIn("validation_status", sev_dict)

    def test_visualizer_overlay_generation(self):
        """Verify visualizer renders overlay without errors and saves to file."""
        img_rgb = np.ones((self.h, self.w, 3), dtype=np.uint8) * 150
        probs = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        crack_idx = DACL10K_CLASSES.index("Crack")
        probs[crack_idx, 30:100, 50:52] = 0.90

        result = self.pipeline.measure_probabilities(probs, image_id="vis_test")
        visualizer = DefectVisualizer()

        overlay = visualizer.render_overlay(
            image_rgb=img_rgb,
            measurement=result,
            probability_maps=probs,
            show_banner=True,
        )

        # Canvas height increases by banner height
        self.assertGreater(overlay.shape[0], self.h)
        self.assertEqual(overlay.shape[1], self.w)
        self.assertEqual(overlay.shape[2], 3)
        self.assertEqual(overlay.dtype, np.uint8)

        # Test save to disk
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_overlay.png"
            visualizer.save_overlay(out_path, img_rgb, result, probability_maps=probs)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
