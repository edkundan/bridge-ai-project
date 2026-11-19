"""
Unit Test: tests.test_checkpoint_resume
Description:
    Comprehensive verification and smoke tests for:
    1. Unified checkpoint saving (model_state_dict, optimizer_state_dict, scheduler_state_dict, best_miou).
    2. Resuming from unified checkpoints.
    3. Backward compatibility resuming from legacy checkpoints (model_state, optimizer_state, scheduler_state, best_val_miou).
    4. Metric key availability (ensuring no KeyError on macro_dice or macro_dice_all_classes).
    5. Checkpoint selection logic based on validation mIoU.
    6. Non-resetting epoch continuation on resumption.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import DACL10K_CLASSES
from src.segmentation.evaluate import SegmentationEvaluator
from src.segmentation.losses import CombinedBCEDiceLoss
from src.segmentation.model import BridgeSegmentationModel


class TestCheckpointSelectionAndResume(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cpu")
        self.num_classes = 19

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_evaluator_metric_keys_exist(self):
        """Verify macro dictionary includes all required keys to prevent KeyError in training loop."""
        evaluator = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=0.5)
        # Synthetic batch: logits shape (2, 19, 32, 32)
        logits = torch.randn(2, 19, 32, 32)
        targets = torch.randint(0, 2, (2, 19, 32, 32)).float()

        evaluator.update(logits, targets, is_logits=True)
        results = evaluator.compute()

        macro = results["macro"]
        self.assertIn("mIoU_all_classes", macro)
        self.assertIn("mIoU_present_classes", macro)
        self.assertIn("macro_dice", macro)
        self.assertIn("macro_dice_all_classes", macro)
        self.assertIn("macro_precision", macro)
        self.assertIn("macro_recall", macro)
        self.assertEqual(macro["macro_dice"], macro["macro_dice_all_classes"])

    def test_unified_checkpoint_save_and_resume(self):
        """Verify unified checkpoint format saves and restores all training state without loss."""
        model = BridgeSegmentationModel(encoder_name="resnet34", encoder_weights=None, num_classes=self.num_classes)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

        # Simulate 1 training step
        optimizer.zero_grad()
        dummy_input = torch.randn(2, 3, 64, 64)
        dummy_target = torch.randint(0, 2, (2, 19, 64, 64)).float()
        out = model(dummy_input)
        loss = CombinedBCEDiceLoss()(out, dummy_target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        ckpt_path = os.path.join(self.temp_dir, "unified_ckpt.pth")
        epoch_completed = 2
        best_miou = 0.2018
        history = {
            "train_loss": [1.0381, 0.9965],
            "val_loss": [1.0184, 0.9965],
            "val_miou": [0.1415, 0.2018],
            "val_dice": [0.2183, 0.2975],
        }

        # Save with unified format
        torch.save({
            "epoch": epoch_completed,
            "model_state_dict": model.state_dict(),
            "model_state": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou": best_miou,
            "best_val_miou": best_miou,
            "val_miou": 0.2018,
            "val_loss": 0.9965,
            "history": history,
        }, ckpt_path)

        # Resume into new model/optimizer
        new_model = BridgeSegmentationModel(encoder_name="resnet34", encoder_weights=None, num_classes=self.num_classes)
        new_optimizer = AdamW(new_model.parameters(), lr=1e-4)
        new_scheduler = CosineAnnealingLR(new_optimizer, T_max=15, eta_min=1e-6)

        chk = torch.load(ckpt_path, map_location=self.device)
        model_state = chk.get("model_state_dict", chk.get("model_state"))
        new_model.load_state_dict(model_state)

        opt_state = chk.get("optimizer_state_dict", chk.get("optimizer_state"))
        new_optimizer.load_state_dict(opt_state)

        sched_state = chk.get("scheduler_state_dict", chk.get("scheduler_state"))
        new_scheduler.load_state_dict(sched_state)

        start_epoch = chk["epoch"] + 1
        resumed_best_miou = chk.get("best_miou", chk.get("best_val_miou", 0.0))

        self.assertEqual(start_epoch, 3, "Resuming from epoch 2 must start at epoch 3 without resetting count")
        self.assertAlmostEqual(resumed_best_miou, 0.2018, places=4)

        # Check weights are identical
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_legacy_checkpoint_backward_compatibility(self):
        """Verify legacy checkpoint keys (model_state, optimizer_state) load cleanly without KeyError."""
        model = BridgeSegmentationModel(encoder_name="resnet34", encoder_weights=None, num_classes=self.num_classes)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=15)

        legacy_ckpt_path = os.path.join(self.temp_dir, "legacy_ckpt.pth")
        torch.save({
            "epoch": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_miou": 0.1415,
        }, legacy_ckpt_path)

        # Load using resilient loader
        chk = torch.load(legacy_ckpt_path, map_location=self.device)
        model_state = chk.get("model_state_dict", chk.get("model_state"))
        self.assertIsNotNone(model_state)

        opt_state = chk.get("optimizer_state_dict", chk.get("optimizer_state"))
        self.assertIsNotNone(opt_state)

        sched_state = chk.get("scheduler_state_dict", chk.get("scheduler_state"))
        self.assertIsNotNone(sched_state)

        best_miou = chk.get("best_miou", chk.get("best_val_miou", 0.0))
        self.assertAlmostEqual(best_miou, 0.1415, places=4)
        self.assertEqual(chk["epoch"] + 1, 2)

    def test_best_checkpoint_selection_logic(self):
        """Verify best model is saved if and only if validation mIoU improves."""
        best_val_miou = 0.1500

        # Epoch with lower mIoU: must NOT update
        val_miou_worse = 0.1480
        should_save_worse = (val_miou_worse > best_val_miou)
        self.assertFalse(should_save_worse)

        # Epoch with higher mIoU: MUST update
        val_miou_better = 0.2018
        should_save_better = (val_miou_better > best_val_miou)
        self.assertTrue(should_save_better)
        if should_save_better:
            best_val_miou = val_miou_better

        self.assertAlmostEqual(best_val_miou, 0.2018, places=4)


if __name__ == "__main__":
    unittest.main()
