"""
Test Suite: tests.test_pipeline
Description:
    Unit and integration tests for DACL10K-v2 multi-label bridge defect
    segmentation pipeline.
    
    Covers:
    1. Class-name mapping (19 classes)
    2. JSON annotation parsing
    3. Polygon-to-mask conversion
    4. Overlapping labels (same class and across different classes)
    5. Empty annotations handling (all-zero mask)
    6. Output tensor shapes (B, 19, H, W)
    7. Multi-label loss calculations (BCE, Dice, Combined, Focal)
    8. One training step parameter updates
    9. Validation metric calculation & zero-division handling
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.mask_converter import DACL10K_CLASSES, MaskConverter
from src.data.dataset import DACL10KDataset, letterbox_resize
from src.segmentation.model import BridgeSegmentationModel
from src.segmentation.losses import (
    MultiLabelSoftDiceLoss,
    CombinedBCEDiceLoss,
    MultiLabelFocalLoss,
    create_loss_function,
)
from src.segmentation.evaluate import SegmentationEvaluator


class TestSegmentationPipeline(unittest.TestCase):
    def setUp(self):
        self.classes = list(DACL10K_CLASSES)
        self.converter = MaskConverter(class_names=self.classes)
        self.device = torch.device("cpu")

    def test_class_name_mapping(self):
        """Verify 19 canonical classes are uniquely mapped."""
        self.assertEqual(len(self.classes), 19)
        self.assertEqual(len(self.converter.class_to_idx), 19)
        self.assertEqual(len(self.converter.idx_to_class), 19)

        # Check round-trip mapping
        for idx, name in enumerate(self.classes):
            self.assertEqual(self.converter.class_to_idx[name], idx)
            self.assertEqual(self.converter.idx_to_class[idx], name)

    def test_polygon_to_mask_conversion(self):
        """Test rasterization of a square polygon onto a 2D mask."""
        h, w = 100, 100
        # 20x20 square from (10, 10) to (30, 30)
        points = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
        mask = self.converter.rasterize_polygon(points, height=h, width=w)

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (h, w))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))

        # Point inside square must be 1, outside must be 0
        self.assertEqual(mask[20, 20], 1)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[50, 50], 0)

        # Test invalid polygon (<3 vertices)
        invalid = self.converter.rasterize_polygon([[10.0, 10.0], [20.0, 20.0]], h, w)
        self.assertIsNone(invalid)

    def test_overlapping_labels_same_class(self):
        """Two overlapping polygons of the same class must be unioned without overflow."""
        h, w = 100, 100
        # Two overlapping squares of class 'Crack'
        anno_data = {
            "imageHeight": h,
            "imageWidth": w,
            "imageName": "test_overlap_same.jpg",
            "shapes": [
                {
                    "label": "Crack",
                    "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]],
                    "shape_type": "polygon",
                },
                {
                    "label": "Crack",
                    "points": [[20.0, 20.0], [50.0, 20.0], [50.0, 50.0], [20.0, 50.0]],
                    "shape_type": "polygon",
                },
            ],
        }

        multi_mask, meta = self.converter.json_to_multilabel_mask(anno_data)
        crack_idx = self.converter.class_to_idx["Crack"]
        crack_channel = multi_mask[:, :, crack_idx]

        self.assertEqual(multi_mask.shape, (h, w, 19))
        self.assertTrue(set(np.unique(crack_channel)).issubset({0, 1}))
        # In the overlap zone (30, 30), value must be strictly 1 (not 2)
        self.assertEqual(crack_channel[30, 30], 1)
        # In the union zone (45, 45), value must be 1
        self.assertEqual(crack_channel[45, 45], 1)
        # Outside must be 0
        self.assertEqual(crack_channel[80, 80], 0)

    def test_overlapping_labels_different_classes(self):
        """Polygons of different classes overlapping spatially must both be 1 in their channels."""
        h, w = 100, 100
        anno_data = {
            "imageHeight": h,
            "imageWidth": w,
            "imageName": "test_overlap_diff.jpg",
            "shapes": [
                {
                    "label": "Rust",
                    "points": [[20.0, 20.0], [60.0, 20.0], [60.0, 60.0], [20.0, 60.0]],
                    "shape_type": "polygon",
                },
                {
                    "label": "Spalling",
                    "points": [[40.0, 40.0], [80.0, 40.0], [80.0, 80.0], [40.0, 80.0]],
                    "shape_type": "polygon",
                },
            ],
        }

        multi_mask, meta = self.converter.json_to_multilabel_mask(anno_data)
        rust_idx = self.converter.class_to_idx["Rust"]
        spalling_idx = self.converter.class_to_idx["Spalling"]

        # Pixel (50, 50) is inside both Rust and Spalling polygons
        self.assertEqual(multi_mask[50, 50, rust_idx], 1)
        self.assertEqual(multi_mask[50, 50, spalling_idx], 1)

        # Pixel (30, 30) is inside Rust only
        self.assertEqual(multi_mask[30, 30, rust_idx], 1)
        self.assertEqual(multi_mask[30, 30, spalling_idx], 0)

        # Pixel (70, 70) is inside Spalling only
        self.assertEqual(multi_mask[70, 70, rust_idx], 0)
        self.assertEqual(multi_mask[70, 70, spalling_idx], 1)

        # All other 17 channels must be 0 everywhere
        for c in range(19):
            if c not in [rust_idx, spalling_idx]:
                self.assertEqual(np.sum(multi_mask[:, :, c]), 0)

    def test_empty_annotations(self):
        """Images with 0 shapes must produce all-zero masks without crashing."""
        h, w = 120, 160
        empty_anno = {
            "imageHeight": h,
            "imageWidth": w,
            "imageName": "empty_sample.jpg",
            "shapes": [],
        }

        multi_mask, meta = self.converter.json_to_multilabel_mask(empty_anno)
        self.assertEqual(multi_mask.shape, (h, w, 19))
        self.assertEqual(np.sum(multi_mask), 0)
        self.assertTrue(meta["is_empty"])
        self.assertEqual(meta["classes_present"], [])

    def test_output_tensor_shapes(self):
        """Forward pass must produce tensor shape (B, 19, H, W)."""
        model = BridgeSegmentationModel(
            architecture="DeepLabV3Plus",
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            num_classes=19,
        ).to(self.device)
        model.eval()

        b, h, w = 2, 256, 256
        dummy_input = torch.randn(b, 3, h, w, device=self.device)

        with torch.no_grad():
            logits = model(dummy_input)
            probs = model.predict_proba(dummy_input)
            masks = model.predict_mask(dummy_input, threshold=0.5)

        self.assertEqual(logits.shape, (b, 19, h, w))
        self.assertEqual(probs.shape, (b, 19, h, w))
        self.assertEqual(masks.shape, (b, 19, h, w))

        # Probabilities must be strictly bounded in [0.0, 1.0]
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))

        # Binary masks must contain only values 0 and 1
        unique_vals = set(torch.unique(masks).cpu().numpy())
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_loss_calculations(self):
        """Test multi-label loss functions for correctness and non-negativity."""
        b, c, h, w = 2, 19, 64, 64
        logits = torch.randn(b, c, h, w, requires_grad=True)
        targets = (torch.rand(b, c, h, w) > 0.8).float()

        # BCE Loss
        bce_loss = nn.BCEWithLogitsLoss()(logits, targets)
        self.assertTrue(torch.isfinite(bce_loss))
        self.assertGreater(bce_loss.item(), 0.0)

        # Soft Dice Loss
        dice_loss = MultiLabelSoftDiceLoss()(logits, targets)
        self.assertTrue(torch.isfinite(dice_loss))
        self.assertGreaterEqual(dice_loss.item(), 0.0)
        self.assertLessEqual(dice_loss.item(), 1.0)

        # Combined BCE + Dice Loss
        combined_loss = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)(logits, targets)
        self.assertTrue(torch.isfinite(combined_loss))
        self.assertGreater(combined_loss.item(), 0.0)

        # Focal Loss
        focal_loss = MultiLabelFocalLoss()(logits, targets)
        self.assertTrue(torch.isfinite(focal_loss))
        self.assertGreater(focal_loss.item(), 0.0)

    def test_one_training_step(self):
        """Verify backprop gradients flow and model weights actually update."""
        model = BridgeSegmentationModel(
            architecture="DeepLabV3Plus",
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            num_classes=19,
        ).to(self.device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = CombinedBCEDiceLoss()

        # Pick one parameter to trace weight changes
        param_to_track = list(model.parameters())[0]
        initial_weight = param_to_track.clone().detach()

        # Synthetic batch
        x = torch.randn(2, 3, 128, 128, device=self.device)
        y = (torch.rand(2, 19, 128, 128, device=self.device) > 0.9).float()

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        updated_weight = param_to_track.clone().detach()
        weight_diff = torch.norm(updated_weight - initial_weight).item()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(weight_diff, 0.0, "Model parameters must update after optimizer step!")

    def test_metric_calculation_and_zero_division(self):
        """Verify IoU, Dice, Precision, Recall, and handling of absent classes."""
        evaluator = SegmentationEvaluator(class_names=self.classes, threshold=0.5)

        # Create known ground truth and prediction
        b, c, h, w = 1, 19, 10, 10
        preds = torch.zeros(b, c, h, w)
        targets = torch.zeros(b, c, h, w)

        # Class 0: Perfect match (10 pixels)
        preds[0, 0, :5, :2] = 1.0
        targets[0, 0, :5, :2] = 1.0

        # Class 1: Partial overlap (TP=5, FP=5, FN=5)
        # GT: 10 pixels (row 0..9, col 0)
        targets[0, 1, :10, 0] = 1.0
        # Pred: 10 pixels (row 5..9 col 0, row 0..4 col 1)
        preds[0, 1, 5:10, 0] = 1.0
        preds[0, 1, :5, 1] = 1.0

        # Class 2..18: Completely absent from GT and Pred (all zeros)

        # Pass is_logits=False since preds are exact binary probability indicators {0.0, 1.0}
        evaluator.update(preds, targets, is_logits=False)
        metrics = evaluator.compute()

        per_class = {item["class_name"]: item for item in metrics["per_class"]}

        # Class 0: IoU=1.0, Dice=1.0, Precision=1.0, Recall=1.0
        c0 = per_class[self.classes[0]]
        self.assertAlmostEqual(c0["iou"], 1.0, places=3)
        self.assertAlmostEqual(c0["dice"], 1.0, places=3)
        self.assertAlmostEqual(c0["precision"], 1.0, places=3)
        self.assertAlmostEqual(c0["recall"], 1.0, places=3)

        # Class 1: TP=5, FP=5, FN=5 -> IoU = 5 / (5+5+5) = 0.3333
        c1 = per_class[self.classes[1]]
        self.assertAlmostEqual(c1["iou"], 5.0 / 15.0, places=3)
        self.assertAlmostEqual(c1["dice"], (2 * 5.0) / (2 * 5.0 + 5.0 + 5.0), places=3)

        # Absent classes: Should have is_present_in_gt = False
        c2 = per_class[self.classes[2]]
        self.assertFalse(c2["is_present_in_gt"])
        self.assertEqual(c2["iou"], 1.0)  # Empty match

        # Macro summary
        macro = metrics["macro"]
        self.assertEqual(macro["classes_present_in_split"], 2)
        # Present mIoU: mean of class 0 (1.0) and class 1 (0.3333) -> ~0.6667
        self.assertAlmostEqual(macro["mIoU_present_classes"], (1.0 + 1.0 / 3.0) / 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
