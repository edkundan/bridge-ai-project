"""
Unit tests for Experiment 03: Rare-Class-Aware Sampling Ablation.

Validates:
1. Annotation-to-label matching against canonical DACL10K_CLASSES.
2. Sampling weight calculation, formula coefficients, and clipping to max_weight 4.0.
3. Alignment on filtered/subset datasets (len(weights) == len(dataset)).
4. Deterministic sampling across epochs and reproducible resume behavior.
5. Disabled-sampler compatibility (exact baseline behavior when sampling is disabled).
6. Training vs. validation loader isolation (validation never sampled or augmented).
7. Correct unweighted loss selection (CombinedBCEDiceLoss, no class weighting).
8. Checkpoint and output directory isolation from baseline and Experiment 02.
9. RecordingWeightedRandomSampler index recording and actual exposure metrics tracking.
"""

import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Ensure project root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.dataset import (
    DACL10KDataset,
    DACL10K_CLASSES,
    RecordingWeightedRandomSampler,
    compute_rare_class_sampling_weights,
    create_dataloaders,
)
from src.segmentation.losses import CombinedBCEDiceLoss, create_loss_function
from src.segmentation.model import BridgeSegmentationModel


class TestExp03Sampling(unittest.TestCase):
    """Test suite for Experiment 03 sampling pipeline, configuration, and reproducibility."""

    @classmethod
    def setUpClass(cls):
        cls.exp03_config_path = repo_root / "configs" / "exp_03_rare_class_sampling.yaml"
        cls.exp02_config_path = repo_root / "configs" / "exp_02_sqrt_weighted_bcedice.yaml"
        cls.baseline_config_path = repo_root / "configs" / "colab_training.yaml"

        with open(cls.exp03_config_path, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

    def test_sampling_weight_formula_and_clipping(self):
        """Verify the exact formula: weight = min(4.0, 1.0 + 2.0*has_WC + 1.5*has_RP + 0.5*has_Crack)."""
        cfg = self.config.get("sampling", {})
        base_w = cfg.get("base_weight", 1.0)
        wc_b = cfg.get("wconccor_boost", 2.0)
        rp_b = cfg.get("rockpocket_boost", 1.5)
        cr_b = cfg.get("crack_boost", 0.5)
        max_w = cfg.get("max_weight", 4.0)

        self.assertEqual(base_w, 1.0)
        self.assertEqual(wc_b, 2.0)
        self.assertEqual(rp_b, 1.5)
        self.assertEqual(cr_b, 0.5)
        self.assertEqual(max_w, 4.0)

        # Test all 8 binary combinations
        cases = [
            (0, 0, 0, 1.0),                  # Base image
            (0, 0, 1, 1.5),                  # Crack only
            (0, 1, 0, 2.5),                  # Rockpocket only
            (1, 0, 0, 3.0),                  # WConccor only
            (0, 1, 1, 3.0),                  # Rockpocket + Crack
            (1, 0, 1, 3.5),                  # WConccor + Crack
            (1, 1, 0, 4.0),                  # WConccor + Rockpocket (clamped from 4.5)
            (1, 1, 1, 4.0),                  # All three (clamped from 5.0)
        ]

        for has_wc, has_rp, has_cr, expected in cases:
            calculated = min(max_w, base_w + wc_b * has_wc + rp_b * has_rp + cr_b * has_cr)
            self.assertEqual(
                calculated,
                expected,
                f"Weight mismatch for (WC={has_wc}, RP={has_rp}, CR={has_cr}): {calculated} != {expected}",
            )

    def test_annotation_to_label_matching(self):
        """Verify that class presence is parsed accurately from LabelMe JSON shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_dir = tmp_path / "images" / "train"
            ann_dir = tmp_path / "annotations" / "train"
            img_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)

            # Create test samples with synthetic shapes
            test_cases = [
                ("sample_01", ["WConccor"]),                          # WC only -> 3.0
                ("sample_02", ["Rockpocket"]),                        # RP only -> 2.5
                ("sample_03", ["Crack"]),                             # CR only -> 1.5
                ("sample_04", ["Rust", "Spalling", "Weathering"]),     # Non-target -> 1.0
                ("sample_05", ["WConccor", "Rockpocket"]),            # WC+RP -> 4.0
                ("sample_06", ["WConccor", "Crack"]),                 # WC+CR -> 3.5
                ("sample_07", []),                                    # Empty -> 1.0
            ]

            import cv2
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

            for sid, labels in test_cases:
                cv2.imwrite(str(img_dir / f"{sid}.jpg"), dummy_img)
                shapes = [{"label": l, "points": [[10, 10], [50, 10], [50, 50], [10, 50]]} for l in labels]
                data = {"imageHeight": 100, "imageWidth": 100, "shapes": shapes}
                with open(ann_dir / f"{sid}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f)

            ds = DACL10KDataset(root_dir=tmp_path, split="train", target_size=(128, 128))
            weights, presence, summary = compute_rare_class_sampling_weights(ds, self.config.get("sampling", {}))

            self.assertEqual(len(weights), 7)
            expected_weights = [3.0, 2.5, 1.5, 1.0, 4.0, 3.5, 1.0]
            self.assertEqual(weights, expected_weights)

            # Check presence matrix
            wc_idx = DACL10K_CLASSES.index("WConccor")
            rp_idx = DACL10K_CLASSES.index("Rockpocket")
            cr_idx = DACL10K_CLASSES.index("Crack")

            self.assertEqual(presence[0, wc_idx], 1)
            self.assertEqual(presence[0, rp_idx], 0)
            self.assertEqual(presence[1, rp_idx], 1)
            self.assertEqual(presence[2, cr_idx], 1)
            self.assertEqual(presence[3, wc_idx], 0)

    def test_sampling_weights_length_and_alignment(self):
        """Test that weight vector length and ordering strictly align with dataset indices even when subsetted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_dir = tmp_path / "images" / "train"
            ann_dir = tmp_path / "annotations" / "train"
            img_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)

            import cv2
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)

            for i in range(10):
                sid = f"img_{i:03d}"
                cv2.imwrite(str(img_dir / f"{sid}.jpg"), dummy_img)
                label = "WConccor" if i % 2 == 0 else "Rust"
                data = {"imageHeight": 64, "imageWidth": 64, "shapes": [{"label": label, "points": [[5, 5], [10, 5], [10, 10]]}]}
                with open(ann_dir / f"{sid}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f)

            # Test full dataset
            ds_full = DACL10KDataset(root_dir=tmp_path, split="train", target_size=(64, 64))
            weights_full, _, _ = compute_rare_class_sampling_weights(ds_full)
            self.assertEqual(len(weights_full), 10)
            self.assertEqual(weights_full[0], 3.0)
            self.assertEqual(weights_full[1], 1.0)

            # Test subset (max_samples=4)
            ds_sub = DACL10KDataset(root_dir=tmp_path, split="train", target_size=(64, 64), max_samples=4)
            weights_sub, _, _ = compute_rare_class_sampling_weights(ds_sub)
            self.assertEqual(len(weights_sub), 4)
            self.assertEqual(weights_sub, weights_full[:4])

    def test_deterministic_sampling_and_resume(self):
        """Verify deterministic sampling with seed 42 and exact reproducible behavior upon resume."""
        weights = [1.0, 1.5, 2.5, 3.0, 4.0, 1.0, 1.0, 3.5]
        num_samples = len(weights)
        seed = 42

        # Run 1: Sequential epochs 1, 2, 3
        gen1 = torch.Generator()
        sampler1 = RecordingWeightedRandomSampler(weights, num_samples=num_samples, replacement=True, generator=gen1)

        gen1.manual_seed(seed + 1)
        draw_epoch1 = list(sampler1)

        gen1.manual_seed(seed + 2)
        draw_epoch2 = list(sampler1)

        gen1.manual_seed(seed + 3)
        draw_epoch3 = list(sampler1)

        # Run 2: Resume directly at epoch 2
        gen_resume = torch.Generator()
        sampler_resume = RecordingWeightedRandomSampler(weights, num_samples=num_samples, replacement=True, generator=gen_resume)

        # Resuming at epoch 2
        gen_resume.manual_seed(seed + 2)
        draw_resume_epoch2 = list(sampler_resume)

        # Continuing to epoch 3
        gen_resume.manual_seed(seed + 3)
        draw_resume_epoch3 = list(sampler_resume)

        self.assertEqual(draw_epoch2, draw_resume_epoch2, "Resumed epoch 2 draw did not match original epoch 2")
        self.assertEqual(draw_epoch3, draw_resume_epoch3, "Resumed epoch 3 draw did not match original epoch 3")

    def test_disabled_sampler_compatibility(self):
        """When sampling.enabled is False, DataLoader must use standard shuffle without WeightedRandomSampler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_dir = tmp_path / "images" / "train"
            ann_dir = tmp_path / "annotations" / "train"
            val_img_dir = tmp_path / "images" / "validation"
            val_ann_dir = tmp_path / "annotations" / "validation"
            for d in [img_dir, ann_dir, val_img_dir, val_ann_dir]:
                d.mkdir(parents=True)

            import cv2
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "s1.jpg"), dummy_img)
            cv2.imwrite(str(val_img_dir / "v1.jpg"), dummy_img)
            with open(ann_dir / "s1.json", "w") as f:
                json.dump({"imageHeight": 64, "imageWidth": 64, "shapes": []}, f)
            with open(val_ann_dir / "v1.json", "w") as f:
                json.dump({"imageHeight": 64, "imageWidth": 64, "shapes": []}, f)

            disabled_cfg = {"enabled": False}
            train_loader, val_loader = create_dataloaders(
                root_dir=tmp_path,
                batch_size=1,
                sampling_cfg=disabled_cfg,
            )

            # Sampler on train_loader should be standard RandomSampler, NOT RecordingWeightedRandomSampler
            self.assertNotIsInstance(train_loader.sampler, RecordingWeightedRandomSampler)
            self.assertFalse(hasattr(train_loader.sampler, "expected_summary"))

    def test_training_validation_isolation(self):
        """Validation loader must NEVER use rare-class sampling, replacement, or augmentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_dir = tmp_path / "images" / "train"
            ann_dir = tmp_path / "annotations" / "train"
            val_img_dir = tmp_path / "images" / "validation"
            val_ann_dir = tmp_path / "annotations" / "validation"
            for d in [img_dir, ann_dir, val_img_dir, val_ann_dir]:
                d.mkdir(parents=True)

            import cv2
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / "s1.jpg"), dummy_img)
            cv2.imwrite(str(val_img_dir / "v1.jpg"), dummy_img)
            with open(ann_dir / "s1.json", "w") as f:
                json.dump({"imageHeight": 64, "imageWidth": 64, "shapes": []}, f)
            with open(val_ann_dir / "v1.json", "w") as f:
                json.dump({"imageHeight": 64, "imageWidth": 64, "shapes": []}, f)

            sampling_cfg = {"enabled": True}
            train_loader, val_loader = create_dataloaders(
                root_dir=tmp_path,
                batch_size=1,
                sampling_cfg=sampling_cfg,
            )

            # Train loader has WeightedRandomSampler
            self.assertIsInstance(train_loader.sampler, RecordingWeightedRandomSampler)
            self.assertTrue(train_loader.dataset.augment)

            # Validation loader has SequentialSampler (no weighted sampling, no augmentation)
            self.assertNotIsInstance(val_loader.sampler, RecordingWeightedRandomSampler)
            self.assertFalse(val_loader.dataset.augment)

    def test_correct_loss_selection_unweighted(self):
        """Verify Experiment 03 strictly isolates sampling by using the unweighted CombinedBCEDiceLoss."""
        loss_type = self.config["loss"]["type"]
        self.assertEqual(loss_type, "combined_bce_dice")
        self.assertFalse(self.config["loss"].get("use_pos_weight", False))
        self.assertNotIn("class_weights", self.config["loss"])

        loss_fn = create_loss_function(
            loss_type=loss_type,
            bce_weight=self.config["loss"].get("bce_weight", 1.0),
            dice_weight=self.config["loss"].get("dice_weight", 1.0),
            pos_weight=None,
        )
        self.assertIsInstance(loss_fn, CombinedBCEDiceLoss)

    def test_checkpoint_and_output_path_isolation(self):
        """Verify Exp 03 output directory is isolated from baseline and Experiment 02."""
        exp03_dir = self.config["experiment"]["output_dir"]
        self.assertIn("exp_03_rare_class_sampling", exp03_dir)

        if self.baseline_config_path.exists():
            with open(self.baseline_config_path, "r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f)
            self.assertNotEqual(exp03_dir, base_cfg["experiment"]["output_dir"])

        if self.exp02_config_path.exists():
            with open(self.exp02_config_path, "r", encoding="utf-8") as f:
                exp02_cfg = yaml.safe_load(f)
            self.assertNotEqual(exp03_dir, exp02_cfg["experiment"]["output_dir"])

    def test_recording_sampler_exposure_tracking(self):
        """Verify RecordingWeightedRandomSampler records drawn indices and tracks class exposure."""
        weights = [1.0, 3.0, 1.0, 4.0]
        gen = torch.Generator().manual_seed(42)
        sampler = RecordingWeightedRandomSampler(weights, num_samples=1000, replacement=True, generator=gen)

        indices = list(sampler)
        self.assertEqual(len(sampler.last_epoch_indices), 1000)
        self.assertEqual(indices, sampler.last_epoch_indices)

        counts = Counter(indices)
        # Weights 3.0 and 4.0 (indices 1 and 3) total 7/9 (~77.8%) vs indices 0 and 2 total 2/9 (~22.2%)
        high_weight_count = counts[1] + counts[3]
        low_weight_count = counts[0] + counts[2]
        self.assertGreater(high_weight_count, low_weight_count * 2)


if __name__ == "__main__":
    unittest.main()
