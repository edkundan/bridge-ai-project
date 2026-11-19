"""
Unit tests for Experiment 02: Square-Root Class-Weighted BCE-Dice Loss
and transfer configuration.

Validates:
1. Canonical class ordering, length 19, and normalization (mean ~ 1.0, sum ~ 19.0).
2. ClassWeightedBCEDiceLoss forward pass and backward gradient finiteness.
3. Gradient scaling ratio for rare classes (w_Rockpocket / w_Weathering ~ 6.9x).
4. Pretrained checkpoint weight loading into model without altering fresh optimizer/scheduler states.
5. Strict output directory isolation between baseline and exp_02.
6. Verification that inner pos_weight is not applied (no double-weighting).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Ensure project root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.dataset import DACL10K_CLASSES
from src.segmentation.losses import ClassWeightedBCEDiceLoss, CombinedBCEDiceLoss, create_loss_function
from src.segmentation.model import BridgeSegmentationModel


class TestExp02WeightedLoss(unittest.TestCase):
    """Test suite for Experiment 02 loss and training setup."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = repo_root / "configs" / "exp_02_sqrt_weighted_bcedice.yaml"
        cls.baseline_config_path = repo_root / "configs" / "colab_training.yaml"
        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

    def test_class_weights_structure_and_normalization(self):
        """Test that candidate weights have 19 classes, mean ~ 1.0, sum ~ 19.0, and match DACL10K classes."""
        weights = self.config["loss"]["class_weights"]
        self.assertEqual(len(weights), 19, f"Expected 19 class weights, got {len(weights)}")

        # Check classes list in config matches canonical classes
        config_classes = self.config["dataset"]["classes"]
        self.assertEqual(config_classes, list(DACL10K_CLASSES))

        # Check mean and sum normalization
        w_tensor = torch.tensor(weights, dtype=torch.float32)
        mean_val = w_tensor.mean().item()
        sum_val = w_tensor.sum().item()
        self.assertAlmostEqual(mean_val, 1.0, places=3, msg=f"Mean weight {mean_val} != 1.0")
        self.assertAlmostEqual(sum_val, 19.0, places=2, msg=f"Sum of weights {sum_val} != 19.0")

        # Verify key expected values from training-split analysis
        # 13: Rockpocket = 1.7576, 16: WConccor = 1.3025, 17: Weathering = 0.2538
        self.assertAlmostEqual(weights[13], 1.7576, places=3, msg="Rockpocket weight mismatch")
        self.assertAlmostEqual(weights[16], 1.3025, places=3, msg="WConccor weight mismatch")
        self.assertAlmostEqual(weights[17], 0.2538, places=3, msg="Weathering weight mismatch")
        self.assertAlmostEqual(weights[3], 1.5679, places=3, msg="Crack weight mismatch")

    def test_class_weighted_loss_forward_backward(self):
        """Test ClassWeightedBCEDiceLoss returns finite scalar loss and gradients."""
        weights = self.config["loss"]["class_weights"]
        criterion = ClassWeightedBCEDiceLoss(class_weights=weights, bce_weight=1.0, dice_weight=1.0)

        # Batch of 2 samples, 19 channels, 32x32 resolution
        logits = torch.randn(2, 19, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (2, 19, 32, 32)).float()

        loss = criterion(logits, targets)
        self.assertEqual(loss.ndim, 0, f"Expected 0-D scalar loss, got shape {loss.shape}")
        self.assertTrue(torch.isfinite(loss), f"Loss is non-finite: {loss.item()}")
        self.assertGreater(loss.item(), 0.0, "Loss should be strictly positive")

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(torch.isfinite(logits.grad)), "Gradients contain NaN or Inf values")
        self.assertFalse(torch.all(logits.grad == 0), "Gradients should not be all zero")

    def test_rare_class_gradient_scaling_ratio(self):
        """Verify that Rockpocket (idx 13) receives ~6.9x gradient scaling compared to Weathering (idx 17)."""
        weights = self.config["loss"]["class_weights"]
        w_rockpocket = weights[13]  # 1.7576
        w_weathering = weights[17]  # 0.2538
        expected_ratio = w_rockpocket / w_weathering
        self.assertAlmostEqual(expected_ratio, 6.925, places=2)

        criterion = ClassWeightedBCEDiceLoss(class_weights=weights, bce_weight=1.0, dice_weight=1.0)

        # Create two identical prediction/target errors on channel 13 vs channel 17
        logits_rp = torch.zeros(1, 19, 16, 16, requires_grad=True)
        targets_rp = torch.zeros(1, 19, 16, 16)
        targets_rp[0, 13] = 1.0  # complete miss on Rockpocket

        loss_rp = criterion(logits_rp, targets_rp)
        loss_rp.backward()
        grad_norm_rp = logits_rp.grad[0, 13].norm().item()

        logits_weath = torch.zeros(1, 19, 16, 16, requires_grad=True)
        targets_weath = torch.zeros(1, 19, 16, 16)
        targets_weath[0, 17] = 1.0  # identical complete miss on Weathering

        loss_weath = criterion(logits_weath, targets_weath)
        loss_weath.backward()
        grad_norm_weath = logits_weath.grad[0, 17].norm().item()

        actual_grad_ratio = grad_norm_rp / max(1e-8, grad_norm_weath)
        self.assertAlmostEqual(
            actual_grad_ratio,
            expected_ratio,
            places=2,
            msg=f"Gradient scaling ratio {actual_grad_ratio:.3f} did not match weight ratio {expected_ratio:.3f}",
        )

    def test_factory_registration(self):
        """Test that create_loss_function handles 'class_weighted_bce_dice' correctly."""
        weights = self.config["loss"]["class_weights"]
        loss_fn = create_loss_function(
            loss_type="class_weighted_bce_dice",
            class_weights=weights,
            bce_weight=1.0,
            dice_weight=1.0,
        )
        self.assertIsInstance(loss_fn, ClassWeightedBCEDiceLoss)

    def test_pretrained_baseline_loading_without_optimizer_state(self):
        """Test loading model weights from checkpoint while keeping fresh optimizer/scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "baseline_best_model.pth")

            # Create a model and save a dummy baseline checkpoint
            base_model = BridgeSegmentationModel(encoder_name="resnet34", num_classes=19)
            base_opt = torch.optim.AdamW(base_model.parameters(), lr=1e-4)

            # Alter a parameter to verify transfer
            with torch.no_grad():
                for param in base_model.parameters():
                    param.fill_(0.42)
                    break

            torch.save(
                {
                    "epoch": 15,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": base_opt.state_dict(),
                    "best_miou": 0.3384,
                },
                chk_path,
            )

            # Initialize a new target model with standard random weights
            target_model = BridgeSegmentationModel(encoder_name="resnet34", num_classes=19)
            target_opt = torch.optim.AdamW(target_model.parameters(), lr=5e-5)
            target_sched = torch.optim.lr_scheduler.CosineAnnealingLR(target_opt, T_max=5, eta_min=1e-6)

            # Load strictly weights into target_model
            chk = torch.load(chk_path, map_location="cpu")
            target_model.load_state_dict(chk["model_state_dict"])

            # Verify target model now has the baseline weights
            for param in target_model.parameters():
                self.assertTrue(torch.allclose(param, torch.tensor(0.42)))
                break

            # Verify optimizer learning rate and state remain fresh
            self.assertEqual(target_opt.param_groups[0]["lr"], 5e-5)
            self.assertEqual(len(target_opt.state), 0, "Optimizer state should be empty (fresh)")
            self.assertEqual(target_sched.last_epoch, 0, "Scheduler last_epoch should be 0")

    def test_independent_output_directory_isolation(self):
        """Verify exp_02 output directory is distinct and isolated from baseline."""
        baseline_dir = self.baseline_config_path.exists()
        if baseline_dir:
            with open(self.baseline_config_path, "r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f)
            base_out = base_cfg["experiment"]["output_dir"]
        else:
            base_out = "/content/drive/MyDrive/bridge_ai_experiments/segmentation_baseline"

        exp02_out = self.config["experiment"]["output_dir"]
        self.assertNotEqual(
            base_out,
            exp02_out,
            "Exp 02 output directory must not match baseline output directory",
        )
        self.assertIn("exp_02_sqrt_weighted_bcedice", exp02_out)

    def test_no_inner_pos_weight(self):
        """Confirm pos_weight is not applied in Exp 02 to prevent double-weighting positive pixels."""
        use_pos_weight = self.config["loss"].get("use_pos_weight", False)
        self.assertFalse(use_pos_weight, "use_pos_weight must be false for isolated exp_02")

    def test_all_ones_weights_reproduce_unweighted_loss(self):
        """Verify that all-one class weights reproduce the original unweighted loss within numerical tolerance."""
        torch.manual_seed(42)
        logits1 = torch.randn(4, 19, 32, 32, requires_grad=True)
        logits2 = logits1.clone().detach().requires_grad_(True)
        targets = torch.randint(0, 2, (4, 19, 32, 32)).float()

        unweighted_fn = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
        ones_weights = [1.0] * 19
        weighted_ones_fn = ClassWeightedBCEDiceLoss(class_weights=ones_weights, bce_weight=1.0, dice_weight=1.0)

        loss_unweighted = unweighted_fn(logits1, targets)
        loss_weighted_ones = weighted_ones_fn(logits2, targets)

        # Forward loss equivalence
        loss_diff = torch.abs(loss_unweighted - loss_weighted_ones).item()
        self.assertLess(
            loss_diff,
            1e-6,
            f"Forward loss discrepancy between unweighted and all-ones weighted: {loss_diff:.8e}",
        )

        # Backward gradient equivalence
        loss_unweighted.backward()
        loss_weighted_ones.backward()
        max_grad_diff = torch.max(torch.abs(logits1.grad - logits2.grad)).item()
        self.assertLess(
            max_grad_diff,
            1e-6,
            f"Backward gradient discrepancy between unweighted and all-ones weighted: {max_grad_diff:.8e}",
        )

    def test_total_shapes_count_and_wconccor_proportion(self):
        """Verify the 54,158 total shape count across 19 classes and WConccor proportion ~0.665%."""
        freq_csv = repo_root / "reports" / "train_class_frequencies.csv"
        if freq_csv.exists():
            import pandas as pd
            df = pd.read_csv(freq_csv)
            self.assertEqual(len(df), 19, "Expected 19 classes in frequencies report")
            total_shapes = int(df["total_shapes"].sum())
            self.assertEqual(total_shapes, 54158, f"Total shapes {total_shapes} != 54158")

            wconccor_row = df[df["class_name"] == "WConccor"]
            self.assertEqual(len(wconccor_row), 1)
            wconccor_shapes = int(wconccor_row["total_shapes"].values[0])
            self.assertEqual(wconccor_shapes, 360)
            wconccor_pct = (wconccor_shapes / total_shapes) * 100
            self.assertAlmostEqual(wconccor_pct, 0.6647, places=3)
            self.assertAlmostEqual(wconccor_pct, 0.665, places=2)


if __name__ == "__main__":
    unittest.main()
