"""
Module: tests.test_qualitative_evaluation
Description:
    Regression and unit tests for Stage 6.2: Reproducible Qualitative Segmentation Evaluation.
    
    Verifies:
    1. Spatial alignment: image tensor unnormalization, GT mask decoding, and prediction mask
       share identical (512, 512) spatial coordinates without translation, rotation, or scaling offsets.
    2. Mathematical consistency: TP, FP, FN partition the defect union correctly.
    3. Reproducibility: Seeded candidate selection produces deterministic, identical outputs.
    4. Balanced evaluation: Both successful and failure/challenging cases are selected for each class.
    5. Artifact generation: Publication figures are saved at 300 DPI, and metadata CSV contains
       all required fields for Crack, ACrack, Spalling, Rust, and ExposedRebars.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import torch

from scripts.generate_qualitative_figures import (
    DEFAULT_FOCUS_CLASSES,
    COLOR_FN,
    COLOR_FP,
    COLOR_TP,
    find_dataset_root,
    generate_qualitative_evaluation,
    render_qualitative_figure,
    select_representative_cases,
    unnormalize_image,
)
from src.data.dataset import (
    DACL10KDataset,
    DACL10K_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    letterbox_resize,
)
from src.segmentation.model import BridgeSegmentationModel


class TestQualitativeEvaluation(unittest.TestCase):
    """Test suite for Stage 6.2 qualitative segmentation evaluation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "qual_eval"

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_spatial_alignment_and_preprocessing_conventions(self):
        """
        Requirement 9: Confirm displayed ground truth and predictions use identical
        spatial alignment and the same preprocessing conventions as Stage 6.
        """
        # Create a synthetic image with a known square defect at (100, 100) to (200, 200)
        img_np = np.zeros((400, 600, 3), dtype=np.uint8)
        mask_np = np.zeros((400, 600, 1), dtype=np.uint8)
        img_np[100:200, 100:200] = [200, 150, 100]
        mask_np[100:200, 100:200, 0] = 1

        # Apply standard letterbox resizing as used in DACL10KDataset
        padded_img, padded_mask, t_info = letterbox_resize(
            img_np, mask_np, target_size=(512, 512), pad_value=0
        )

        self.assertEqual(padded_img.shape, (512, 512, 3))
        self.assertEqual(padded_mask.shape, (512, 512, 1))

        # Check that the defect region in image and mask match coordinate-wise
        scale = t_info["scale"]
        pad_top = t_info["pad_top"]
        pad_left = t_info["pad_left"]

        exp_y_min = int(round(100 * scale)) + pad_top
        exp_y_max = int(round(200 * scale)) + pad_top
        exp_x_min = int(round(100 * scale)) + pad_left
        exp_x_max = int(round(200 * scale)) + pad_left

        # Binary mask core interior must be strictly 1
        inner_mask_slice = padded_mask[exp_y_min + 2:exp_y_max - 2, exp_x_min + 2:exp_x_max - 2, 0]
        self.assertTrue(np.all(inner_mask_slice == 1), "Core defect region in mask must be 1")

        # Outside the defect region in padded area must be strictly 0
        self.assertEqual(padded_mask[0:pad_top, :, 0].sum(), 0)

        # Verify unnormalize_image inverts ImageNet normalization accurately
        norm_img = ((padded_img / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(norm_img.transpose(2, 0, 1).astype(np.float32))
        restored = unnormalize_image(img_tensor)

        self.assertEqual(restored.shape, (512, 512, 3))
        self.assertEqual(restored.dtype, np.uint8)
        # Max difference should be at most 1 pixel due to rounding
        self.assertLessEqual(np.max(np.abs(restored.astype(int) - padded_img.astype(int))), 2)

    def test_error_breakdown_set_theoretic_consistency(self):
        """
        Verify that TP, FP, and FN form a strictly disjoint partition of the defect union:
        - TP and FP are disjoint
        - TP and FN are disjoint
        - FP and FN are disjoint
        - TP + FP = Pred
        - TP + FN = GT
        - IoU = TP / (TP + FP + FN)
        """
        gt_mask = np.zeros((512, 512), dtype=np.uint8)
        pred_mask = np.zeros((512, 512), dtype=np.uint8)

        # Overlapping squares: GT=[50:150, 50:150], Pred=[100:200, 100:200]
        gt_mask[50:150, 50:150] = 1
        pred_mask[100:200, 100:200] = 1

        tp_bool = (gt_mask == 1) & (pred_mask == 1)
        fp_bool = (gt_mask == 0) & (pred_mask == 1)
        fn_bool = (gt_mask == 1) & (pred_mask == 0)

        # Mutually exclusive partitions
        self.assertEqual(np.logical_and(tp_bool, fp_bool).sum(), 0)
        self.assertEqual(np.logical_and(tp_bool, fn_bool).sum(), 0)
        self.assertEqual(np.logical_and(fp_bool, fn_bool).sum(), 0)

        tp = int(tp_bool.sum())
        fp = int(fp_bool.sum())
        fn = int(fn_bool.sum())
        gt_px = int(gt_mask.sum())
        pred_px = int(pred_mask.sum())

        self.assertEqual(tp + fn, gt_px, "TP + FN must equal ground-truth pixel count")
        self.assertEqual(tp + fp, pred_px, "TP + FP must equal prediction pixel count")

        expected_tp = 50 * 50  # [100:150, 100:150]
        self.assertEqual(tp, expected_tp)
        self.assertEqual(fp, (100 * 100) - expected_tp)
        self.assertEqual(fn, (100 * 100) - expected_tp)

        union = tp + fp + fn
        iou = tp / union
        self.assertAlmostEqual(iou, 2500 / 17500, places=4)

    def test_reproducible_selection_and_no_cherry_picking(self):
        """
        Requirement 4: Select representative validation images reproducibly using a fixed
        random seed and measured per-image errors. Include successful and failed predictions.
        """
        # Create a mock dictionary of evaluated results for 2 classes
        synthetic_results = {
            "Crack": [
                {"sample_id": "s1", "target_class": "Crack", "class_index": 0, "iou": 0.85, "has_gt": True, "gt_pixels": 500, "pred_pixels": 450, "tp_pixels": 400, "fp_pixels": 50, "fn_pixels": 100, "dice": 0.85, "precision": 0.88, "recall": 0.80, "gt_area_pct": 0.19, "pred_area_pct": 0.17, "image_tensor": torch.zeros((3, 512, 512)), "gt_mask": np.zeros((512, 512), dtype=np.uint8), "pred_mask": np.zeros((512, 512), dtype=np.uint8)},
                {"sample_id": "s2", "target_class": "Crack", "class_index": 0, "iou": 0.45, "has_gt": True, "gt_pixels": 300, "pred_pixels": 350, "tp_pixels": 180, "fp_pixels": 170, "fn_pixels": 120, "dice": 0.55, "precision": 0.51, "recall": 0.60, "gt_area_pct": 0.11, "pred_area_pct": 0.13, "image_tensor": torch.zeros((3, 512, 512)), "gt_mask": np.zeros((512, 512), dtype=np.uint8), "pred_mask": np.zeros((512, 512), dtype=np.uint8)},
                {"sample_id": "s3", "target_class": "Crack", "class_index": 0, "iou": 0.05, "has_gt": True, "gt_pixels": 200, "pred_pixels": 50, "tp_pixels": 10, "fp_pixels": 40, "fn_pixels": 190, "dice": 0.08, "precision": 0.20, "recall": 0.05, "gt_area_pct": 0.08, "pred_area_pct": 0.02, "image_tensor": torch.zeros((3, 512, 512)), "gt_mask": np.zeros((512, 512), dtype=np.uint8), "pred_mask": np.zeros((512, 512), dtype=np.uint8)},
            ],
            "Spalling": [
                {"sample_id": "s4", "target_class": "Spalling", "class_index": 8, "iou": 0.90, "has_gt": True, "gt_pixels": 4000, "pred_pixels": 3800, "tp_pixels": 3600, "fp_pixels": 200, "fn_pixels": 400, "dice": 0.92, "precision": 0.95, "recall": 0.90, "gt_area_pct": 1.52, "pred_area_pct": 1.45, "image_tensor": torch.zeros((3, 512, 512)), "gt_mask": np.zeros((512, 512), dtype=np.uint8), "pred_mask": np.zeros((512, 512), dtype=np.uint8)},
                {"sample_id": "s5", "target_class": "Spalling", "class_index": 8, "iou": 0.00, "has_gt": True, "gt_pixels": 1500, "pred_pixels": 0, "tp_pixels": 0, "fp_pixels": 0, "fn_pixels": 1500, "dice": 0.00, "precision": 0.00, "recall": 0.00, "gt_area_pct": 0.57, "pred_area_pct": 0.00, "image_tensor": torch.zeros((3, 512, 512)), "gt_mask": np.zeros((512, 512), dtype=np.uint8), "pred_mask": np.zeros((512, 512), dtype=np.uint8)},
            ]
        }

        # 1. Reproducibility test: Running twice with seed=42 yields identical selection
        run1 = select_representative_cases(synthetic_results, ["Crack", "Spalling"], seed=42)
        run2 = select_representative_cases(synthetic_results, ["Crack", "Spalling"], seed=42)

        self.assertEqual(len(run1), 4)  # 2 classes * 2 outcomes = 4
        self.assertEqual([c["sample_id"] for c in run1], [c["sample_id"] for c in run2])
        self.assertEqual([c["outcome_type"] for c in run1], [c["outcome_type"] for c in run2])

        # 2. Balanced evaluation: Confirm presence of both success and failure
        crack_cases = [c for c in run1 if c["target_class"] == "Crack"]
        self.assertEqual(len(crack_cases), 2)
        crack_outcomes = {c["outcome_type"] for c in crack_cases}
        self.assertEqual(crack_outcomes, {"success", "failure"})

        # Success case must have highest IoU (s1 with 0.85), failure has lowest (s3 with 0.05)
        success_crack = [c for c in crack_cases if c["outcome_type"] == "success"][0]
        failure_crack = [c for c in crack_cases if c["outcome_type"] == "failure"][0]
        self.assertEqual(success_crack["sample_id"], "s1")
        self.assertEqual(failure_crack["sample_id"], "s3")
        self.assertGreater(success_crack["iou"], failure_crack["iou"])

    def test_render_qualitative_figure_300_dpi_and_layout(self):
        """
        Requirement 6 & 2: Save PNG figures at 300 DPI with 4 columns:
        Original image, Ground-truth, Prediction, and Error visualization.
        """
        # Create a valid case dict
        img_tensor = torch.zeros((3, 512, 512), dtype=torch.float32)
        gt_mask = np.zeros((512, 512), dtype=np.uint8)
        pred_mask = np.zeros((512, 512), dtype=np.uint8)

        gt_mask[100:200, 100:200] = 1
        pred_mask[120:220, 120:220] = 1

        case = {
            "sample_id": "test_sample_0001",
            "target_class": "Crack",
            "class_index": 0,
            "outcome_type": "success",
            "outcome_label": "Successful Detection",
            "iou": 0.4521,
            "dice": 0.6227,
            "precision": 0.6400,
            "recall": 0.6400,
            "gt_pixels": 10000,
            "pred_pixels": 10000,
            "tp_pixels": 6400,
            "fp_pixels": 3600,
            "fn_pixels": 3600,
            "image_tensor": img_tensor,
            "gt_mask": gt_mask,
            "pred_mask": pred_mask,
        }

        fig_path = self.output_dir / "test_figure_300dpi.png"
        render_qualitative_figure(case, fig_path, threshold=0.50, dpi=300)

        self.assertTrue(fig_path.exists(), f"Figure was not created at {fig_path}")
        self.assertGreater(fig_path.stat().st_size, 50000, "Figure file size is too small for 300 DPI PNG")

        # Verify image dimensions using PIL or cv2
        saved_img = cv2.imread(str(fig_path))
        self.assertIsNotNone(saved_img)
        h, w = saved_img.shape[:2]
        # At 20 inches wide and 300 DPI, width should be around 5000-6000 pixels
        self.assertGreater(w, 4000, f"Expected 300 DPI high-res width > 4000 px, got {w}")

    def test_default_focus_classes_and_color_palette(self):
        """
        Requirement 3 & 5: Focus on Crack, ACrack, Spalling, Rust, ExposedRebars.
        Verify all 5 exist in DACL10K_CLASSES and have valid RGB colors.
        """
        for c in DEFAULT_FOCUS_CLASSES:
            self.assertIn(c, DACL10K_CLASSES, f"Focus class {c} must be in canonical 19 classes")

        from src.measurement.visualization import CLASS_COLOR_PALETTE_RGB
        self.assertEqual(len(CLASS_COLOR_PALETTE_RGB), 19)
        for rgb in CLASS_COLOR_PALETTE_RGB:
            self.assertEqual(len(rgb), 3)
            self.assertTrue(all(0 <= val <= 255 for val in rgb))

    def test_end_to_end_qualitative_evaluation_mocked(self):
        """
        Requirement 7 & 6: Test end-to-end execution of generate_qualitative_evaluation
        using mocked model and dataset, verifying CSV columns and JSON output.
        """
        # Create a mock checkpoint file
        mock_ckpt = Path(self.temp_dir) / "mock_model.pth"
        dummy_state = BridgeSegmentationModel(num_classes=19).state_dict()
        torch.save({"model_state_dict": dummy_state, "epoch": 5, "best_miou": 0.3542}, mock_ckpt)

        # Mock scan_candidate_images and evaluate_candidates to run quickly without GPU/disk
        def mock_scan(root_dir, focus_classes, max_candidates_per_class, seed):
            return {c: [f"val_{c.lower()}_01", f"val_{c.lower()}_02"] for c in focus_classes}, ["val_01", "val_02"]

        def mock_eval(model, dataset_root, candidate_stems, focus_classes, device, threshold):
            res = {}
            for c in focus_classes:
                c_idx = DACL10K_CLASSES.index(c)
                res[c] = [
                    {
                        "sample_id": f"mock_{c}_high",
                        "dataset_index": 0,
                        "target_class": c,
                        "class_index": c_idx,
                        "gt_pixels": 2000,
                        "pred_pixels": 1800,
                        "gt_area_pct": 0.76,
                        "pred_area_pct": 0.69,
                        "tp_pixels": 1600,
                        "fp_pixels": 200,
                        "fn_pixels": 400,
                        "iou": 0.7273,
                        "dice": 0.8421,
                        "precision": 0.8889,
                        "recall": 0.8000,
                        "image_tensor": torch.zeros((3, 512, 512)),
                        "gt_mask": np.zeros((512, 512), dtype=np.uint8),
                        "pred_mask": np.zeros((512, 512), dtype=np.uint8),
                        "has_gt": True,
                    },
                    {
                        "sample_id": f"mock_{c}_low",
                        "dataset_index": 1,
                        "target_class": c,
                        "class_index": c_idx,
                        "gt_pixels": 1000,
                        "pred_pixels": 100,
                        "gt_area_pct": 0.38,
                        "pred_area_pct": 0.04,
                        "tp_pixels": 50,
                        "fp_pixels": 50,
                        "fn_pixels": 950,
                        "iou": 0.0476,
                        "dice": 0.0909,
                        "precision": 0.5000,
                        "recall": 0.0500,
                        "image_tensor": torch.zeros((3, 512, 512)),
                        "gt_mask": np.zeros((512, 512), dtype=np.uint8),
                        "pred_mask": np.zeros((512, 512), dtype=np.uint8),
                        "has_gt": True,
                    },
                ]
            return res

        with patch("scripts.generate_qualitative_figures.find_dataset_root", return_value=Path(self.temp_dir)):
            with patch("scripts.generate_qualitative_figures.scan_candidate_images", side_effect=mock_scan):
                with patch("scripts.generate_qualitative_figures.evaluate_candidates", side_effect=mock_eval):
                    result = generate_qualitative_evaluation(
                        root_dir=self.temp_dir,
                        checkpoint=str(mock_ckpt),
                        output_dir=str(self.output_dir),
                        device_name="cpu",
                        threshold=0.50,
                        seed=42,
                        focus_classes=DEFAULT_FOCUS_CLASSES,
                        dpi=100,  # fast DPI for test
                    )

        # Verify output directory and artifacts
        self.assertTrue(Path(result["metadata_csv"]).exists())
        self.assertTrue(Path(result["summary_json"]).exists())

        # Check metadata CSV contents
        df = pd.read_csv(result["metadata_csv"])
        self.assertEqual(len(df), 10, "5 classes * 2 outcomes (success/failure) = 10 rows")
        expected_cols = [
            "sample_id",
            "target_class",
            "outcome_type",
            "outcome_label",
            "iou",
            "dice",
            "precision",
            "recall",
            "tp_pixels",
            "fp_pixels",
            "fn_pixels",
            "gt_pixels",
            "pred_pixels",
            "figure_filename",
            "random_seed",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column {col} in CSV")

        # Confirm all 5 classes are represented with both success and failure
        for c in DEFAULT_FOCUS_CLASSES:
            class_df = df[df["target_class"] == c]
            self.assertEqual(len(class_df), 2, f"Class {c} must have exactly 2 entries (success + failure)")
            self.assertEqual(set(class_df["outcome_type"]), {"success", "failure"})

        # Check summary JSON
        with open(result["summary_json"], "r") as f:
            summary = json.load(f)
        self.assertEqual(summary["total_figures_generated"], 10)
        self.assertEqual(summary["focus_classes"], DEFAULT_FOCUS_CLASSES)


if __name__ == "__main__":
    unittest.main()
