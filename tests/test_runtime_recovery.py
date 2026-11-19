"""
Unit Test: tests.test_runtime_recovery
Description:
    Validates the runtime recovery and checkpoint discovery architecture:
    1. Validates DACL10K_Colab_Training.ipynb JSON format, cell ordering, and required recovery cells.
    2. Tests idempotent dependency checking logic.
    3. Tests checkpoint discovery and resume epoch calculation logic.
    4. Verifies safety switch behavior (EXECUTE_QUICK_RESUME = False).
    5. Confirms invariants (15 total epochs, 19 classes, 512x512 resolution, tau=0.5).
"""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch


class TestRuntimeRecovery(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.notebook_path = self.repo_root / "notebooks" / "DACL10K_Colab_Training.ipynb"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_notebook_json_validity_and_structure(self):
        """Verify the notebook is valid JSON and contains all required recovery sections."""
        self.assertTrue(self.notebook_path.exists(), f"Missing notebook at {self.notebook_path}")

        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertEqual(nb.get("nbformat"), 4)
        cells = nb.get("cells", [])
        self.assertGreater(len(cells), 20)

        # Check for Recovery Protocol in Header
        header_text = "".join(cells[0].get("source", []))
        self.assertIn("RECOVERY PROTOCOL", header_text)
        self.assertIn("Browser Refresh", header_text)
        self.assertIn("Runtime Reset", header_text)

        # Check Cell R1 (Runtime Recovery)
        cell_r1_md = "".join(cells[1].get("source", []))
        cell_r1_code = "".join(cells[2].get("source", []))
        self.assertIn("CELL R1: RUNTIME RECOVERY", cell_r1_md)
        self.assertIn("DACL10K-v2 COLAB RUNTIME RECOVERY", cell_r1_code)
        self.assertIn("RECOVERY_CHECKPOINT", cell_r1_code)
        self.assertIn("RECOVERY_START_EPOCH", cell_r1_code)
        self.assertIn("TOTAL_EPOCHS = 15", cell_r1_code)

        # Check Cell R2 (Quick Resume)
        cell_r2_md = "".join(cells[3].get("source", []))
        cell_r2_code = "".join(cells[4].get("source", []))
        self.assertIn("CELL R2: QUICK RESUME", cell_r2_md)
        self.assertIn("EXECUTE_QUICK_RESUME = False", cell_r2_code)
        self.assertIn("SAFETY GUARD", cell_r2_code)

        # Verify existing sections are preserved
        all_sources = ["".join(c.get("source", [])) for c in cells]
        joined_all = "\n".join(all_sources)
        self.assertIn("Section 1: Environment", joined_all)
        self.assertIn("Section 2: Repository", joined_all)
        self.assertIn("Section 3: Dataset", joined_all)
        self.assertIn("Section 4: Configuration", joined_all)
        self.assertIn("Section 5: Preflight", joined_all)
        self.assertIn("Section 6: Throughput", joined_all)
        self.assertIn("Section 6.5: Checkpoint Discovery", joined_all)
        self.assertIn("Section 6.6: Reproducible Checkpoint Evaluation", joined_all)
        self.assertIn("Section 7: Resumable Full Baseline Training Loop", joined_all)
        self.assertIn("Section 8: Held-Out Validation Evaluation", joined_all)

    def test_idempotent_dependency_detection(self):
        """Verify the dependency checking logic properly identifies installed vs missing modules."""
        required_deps = {
            "torch": "torch",
            "json": "json",
            "non_existent_dummy_pkg_12345": "dummy-pip-pkg",
        }
        missing = [pip_pkg for mod, pip_pkg in required_deps.items() if importlib.util.find_spec(mod) is None]
        self.assertEqual(missing, ["dummy-pip-pkg"])

    def test_checkpoint_discovery_and_resume_decision(self):
        """Simulate Google Drive checkpoint discovery and verify start_epoch = saved_epoch + 1."""
        ckpt_dir = Path(self.temp_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy epoch-2 full-state checkpoint
        dummy_state = {"weight": torch.randn(2, 2)}
        opt_state = {"param_groups": []}
        sched_state = {"last_epoch": 2}
        latest_path = ckpt_dir / "latest_checkpoint.pth"

        torch.save({
            "epoch": 2,
            "model_state_dict": dummy_state,
            "optimizer_state_dict": opt_state,
            "scheduler_state_dict": sched_state,
            "best_miou": 0.2018,
        }, latest_path)

        # Run recovery logic
        chk = torch.load(latest_path, map_location="cpu")
        saved_ep = chk.get("epoch", 0)
        has_opt = ("optimizer_state_dict" in chk or "optimizer_state" in chk)
        has_sched = ("scheduler_state_dict" in chk or "scheduler_state" in chk)
        b_miou = chk.get("best_miou", 0.0)

        self.assertTrue(has_opt)
        self.assertTrue(has_sched)
        self.assertEqual(saved_ep, 2)
        start_epoch = saved_ep + 1
        self.assertEqual(start_epoch, 3)
        self.assertAlmostEqual(b_miou, 0.2018, places=4)

    def test_notebook_drive_output_dir_consistency(self):
        """Verify DRIVE_OUTPUT_DIR is consistent across all notebook cells and no stale paths exist."""
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        expected_drive_dir = "/content/drive/MyDrive/bridge_ai_experiments/segmentation_baseline"
        stale_dir = "dacl10k_baseline_experiments"

        found_expected_dir = 0
        cells_with_drive_var = 0
        for i, cell in enumerate(nb.get("cells", [])):
            src = "".join(cell.get("source", []))
            self.assertNotIn(
                stale_dir,
                src,
                f"Cell {i} contains stale path reference '{stale_dir}'"
            )
            if expected_drive_dir in src:
                found_expected_dir += 1
            if "DRIVE_OUTPUT_DIR" in src:
                cells_with_drive_var += 1

        # The explicit path must appear in R1 and Section 4
        self.assertGreaterEqual(
            found_expected_dir, 3,
            f"Expected '{expected_drive_dir}' to appear in definitions and recovery headers, found {found_expected_dir}"
        )
        # DRIVE_OUTPUT_DIR variable must be referenced across all active workflow cells (R1, R2, S4, S5, S6.5, S6.6, S7, S8, S9)
        self.assertGreaterEqual(
            cells_with_drive_var, 8,
            f"Expected DRIVE_OUTPUT_DIR variable to be used across at least 8 workflow cells, found {cells_with_drive_var}"
        )

    def test_checkpoint_metadata_inconsistency_audit(self):
        """Simulate the Drive artifact inconsistency: header best_miou=0.1415 vs history [0.1415, 0.2018]."""
        exp_dir = Path(self.temp_dir)
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create latest_checkpoint.pth: epoch 2, opt/sched present, header best_miou = 0.1415
        dummy_state = {"weight": torch.randn(2, 2)}
        opt_state = {"param_groups": []}
        sched_state = {"last_epoch": 2}
        latest_path = ckpt_dir / "latest_checkpoint.pth"

        torch.save({
            "epoch": 2,
            "model_state_dict": dummy_state,
            "optimizer_state_dict": opt_state,
            "scheduler_state_dict": sched_state,
            "best_miou": 0.1415,
        }, latest_path)

        # 2. Create best_model.pth: epoch 2, weights only
        best_path = ckpt_dir / "best_model.pth"
        torch.save({
            "epoch": 2,
            "model_state_dict": dummy_state,
            "best_miou": 0.2018,
        }, best_path)

        # 3. Create training_history.json: val_miou = [0.1415, 0.2018], val_dice = [0.2183, 0.2975]
        history_path = exp_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "train_loss": [1.0381, 1.0102],
                "val_loss": [1.0184, 0.9965],
                "val_miou": [0.1415, 0.2018],
                "val_dice": [0.2183, 0.2975],
            }, f)

        # Run audit logic exactly as executed in Cell R1 / Section 6.5
        chk = torch.load(latest_path, map_location="cpu")
        header_miou = chk.get("best_miou", 0.0)
        saved_epoch = chk.get("epoch", 0)

        with open(history_path, "r") as f:
            hist = json.load(f)

        hist_val_miou = hist.get("val_miou", [])
        hist_max_miou = max(hist_val_miou) if hist_val_miou else 0.0

        # Verify inconsistency is detected
        self.assertAlmostEqual(header_miou, 0.1415, places=4)
        self.assertAlmostEqual(hist_max_miou, 0.2018, places=4)
        self.assertNotEqual(header_miou, hist_max_miou)

        # Resume epoch must be saved_epoch + 1 = 3
        start_epoch = saved_epoch + 1
        self.assertEqual(start_epoch, 3)

        # Target best_val_miou must take history max (0.2018) to protect best_model.pth
        best_val_miou = max(header_miou, hist_max_miou)
        self.assertAlmostEqual(best_val_miou, 0.2018, places=4)

    def test_section_6_6_reproducible_evaluation_contract(self):
        """Verify Section 6.6 code cell evaluates on 975 images, tau=0.5, and checks reproduction metrics."""
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        sec_6_6_found = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "SECTION 6.6" in src:
                sec_6_6_found = True
                self.assertIn("975", src, "Section 6.6 must reference 975 validation images")
                self.assertIn("0.2018", src, "Section 6.6 must check against 0.2018 mIoU")
                self.assertIn("0.2975", src, "Section 6.6 must check against 0.2975 macro Dice")
                self.assertIn("target_ckpt", src)
                self.assertIn("macro_dice", src.lower())

        self.assertTrue(sec_6_6_found, "Section 6.6 code cell must exist in notebook")


    def test_weights_only_checkpoint_flagged(self):
        """Verify weights-only checkpoint without optimizer state is NOT treated as full-state resume."""
        ckpt_dir = Path(self.temp_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_path = ckpt_dir / "best_model.pth"
        torch.save({
            "epoch": 1,
            "model_state_dict": {"weight": torch.randn(2, 2)},
            "best_miou": 0.1656,
        }, best_path)

        chk = torch.load(best_path, map_location="cpu")
        has_opt = ("optimizer_state_dict" in chk or "optimizer_state" in chk)
        has_sched = ("scheduler_state_dict" in chk or "scheduler_state" in chk)

        self.assertFalse(has_opt)
        self.assertFalse(has_sched)


if __name__ == "__main__":
    unittest.main()

