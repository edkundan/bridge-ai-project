"""
Script: src.segmentation.overfit_test
Description:
    Small-subset overfit test (Stage 1A diagnostic) for DACL10K-v2 segmentation.
    
    Diagnostic Goals:
    - Verifies network capacity to fit a small 16-image subset without stochastic augmentation.
    - Evaluates whether training loss decreases systematically and training-subset mIoU/Dice increase.
    - Tracks per-class IoU for represented classes across 20 epochs.
    - Generates 4-panel side-by-side qualitative comparisons (Original, GT, Predicted Mask, Overlay).
    - Verifies checkpoint saving and reloading for inference.
    
    IMPORTANT NOTE:
    - Metrics produced in this script are training-subset diagnostics only.
    - They do NOT represent validation or generalization capability.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DACL10KDataset, DACL10K_CLASSES, custom_collate_fn
from src.segmentation.evaluate import SegmentationEvaluator, save_qualitative_comparison
from src.segmentation.losses import create_loss_function
from src.segmentation.model import BridgeSegmentationModel
from src.segmentation.predict import BridgeDefectPredictor
from src.segmentation.train import select_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("overfit_test")


def plot_overfit_curves(history: Dict[str, List[float]], save_path: Path):
    """Plots training loss and diagnostic training-subset mIoU curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], color="tab:blue", marker="o", lw=2, label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Combined Loss", fontsize=11)
    ax1.set_title("Training Loss Trajectory (16-Sample Overfit)", fontsize=12, weight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # mIoU curve
    ax2.plot(epochs, history["subset_miou"], color="tab:green", marker="s", lw=2, label="Subset mIoU (All Classes)")
    ax2.plot(epochs, history["subset_dice"], color="tab:purple", marker="^", lw=2, linestyle="--", label="Subset Macro Dice")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Training-Subset Metric Progression", fontsize=12, weight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.suptitle("Stage 1A: 16-Sample Overfit Test Diagnostic Curves", fontsize=14, weight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def run_overfit_test(config_path: str = "configs/overfit_test.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_dir = Path(config.get("experiment", {}).get("output_dir", "experiments/overfit_test"))
    exp_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = exp_dir / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = exp_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    log_file = exp_dir / "overfit_test.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("STARTING STAGE 1A: SMALL-SUBSET OVERFIT TEST")
    logger.info(f"Config: {config_path} | Output Dir: {exp_dir}")
    logger.info("=" * 70)

    # Reproducibility
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)

    # Device
    device = select_device(config.get("training", {}).get("device", "auto"))
    logger.info(f"Execution Device: {device}")

    # Dataset & DataLoader
    root_dir = config["dataset"]["root_dir"]
    sample_filenames = config["dataset"]["sample_filenames"]
    image_size = tuple(config["training"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    threshold = float(config["training"].get("threshold", 0.5))

    logger.info(f"Target 16 samples to overfit: {len(sample_filenames)} files specified.")

    train_ds = DACL10KDataset(
        root_dir=root_dir,
        split="train",
        target_size=image_size,
        augment=False,  # Strictly NO augmentation for deterministic overfit test
        sample_filenames=sample_filenames,
        return_metadata=True,
    )
    assert len(train_ds) == 16, f"Expected exactly 16 samples, loaded {len(train_ds)}"

    # Audit classes present in the 16-sample set
    all_classes_in_subset = set()
    sample_records = []
    for idx in range(len(train_ds)):
        _, _, meta = train_ds[idx]
        clss = meta.get("classes_present", [])
        all_classes_in_subset.update(clss)
        sample_records.append({
            "image_id": meta["image_id"],
            "classes": clss,
            "shape_count": meta.get("total_shapes", 0),
        })

    logger.info(f"16-sample subset spans {len(all_classes_in_subset)} / 19 classes.")
    logger.info(f"Classes represented: {sorted(list(all_classes_in_subset))}")

    # Training and eval loaders for the same 16 images
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn
    )
    eval_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn
    )

    # Model
    model = BridgeSegmentationModel(
        architecture=config.get("model", {}).get("architecture", "DeepLabV3Plus"),
        encoder_name=config.get("model", {}).get("encoder_name", "resnet34"),
        encoder_weights=config.get("model", {}).get("encoder_weights", "imagenet"),
        in_channels=3,
        num_classes=19,
    ).to(device)

    # Loss
    loss_cfg = config.get("loss", {})
    criterion = create_loss_function(
        loss_type=loss_cfg.get("type", "combined_bce_dice"),
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
    )

    # Optimizer
    lr = float(config.get("training", {}).get("learning_rate", 3e-4))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    total_epochs = int(config.get("training", {}).get("epochs", 20))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    history: Dict[str, List[Any]] = {
        "train_loss": [],
        "subset_miou": [],
        "subset_dice": [],
        "subset_precision": [],
        "subset_recall": [],
        "epoch_duration": [],
    }

    per_class_history_e1 = None
    per_class_history_e20 = None
    best_miou = 0.0

    logger.info(f"Beginning 20-epoch overfit loop (batch_size={batch_size}, lr={lr})...")

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss.item()}")

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        scheduler.step()

        # Evaluate on the SAME 16 training samples
        model.eval()
        evaluator = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=threshold)
        with torch.no_grad():
            for images, masks, metas in eval_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                evaluator.update(logits, masks, sample_ids=[m["image_id"] for m in metas], is_logits=True)

        metrics = evaluator.compute()
        epoch_dur = time.time() - t0

        miou = metrics["macro"]["mIoU_all_classes"]
        dice = metrics["macro"]["macro_dice"]
        prec = metrics["macro"]["macro_precision"]
        rec = metrics["macro"]["macro_recall"]

        history["train_loss"].append(round(epoch_loss, 4))
        history["subset_miou"].append(round(miou, 4))
        history["subset_dice"].append(round(dice, 4))
        history["subset_precision"].append(round(prec, 4))
        history["subset_recall"].append(round(rec, 4))
        history["epoch_duration"].append(round(epoch_dur, 2))

        logger.info(
            f"Epoch {epoch:02d}/{total_epochs:02d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Subset mIoU: {miou:.4f} | "
            f"Subset Dice: {dice:.4f} | "
            f"Precision: {prec:.4f} | "
            f"Recall: {rec:.4f} | "
            f"Duration: {epoch_dur:.1f}s"
        )

        if epoch == 1:
            per_class_history_e1 = metrics["per_class"]
            pd.DataFrame(metrics["per_class"]).to_csv(exp_dir / "per_class_metrics_epoch_01.csv", index=False)

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "config": config,
            }, chk_dir / "best_overfit_model.pth")

        plot_overfit_curves(history, exp_dir / "overfit_learning_curves.png")

    # Final epoch metrics
    per_class_history_e20 = metrics["per_class"]
    pd.DataFrame(metrics["per_class"]).to_csv(exp_dir / "per_class_metrics_epoch_20.csv", index=False)

    # Save final model
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "final_metrics": metrics,
        "config": config,
    }, chk_dir / "final_overfit_model.pth")

    with open(exp_dir / "overfit_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    # Generate 4-panel visual comparisons for ALL 16 samples
    logger.info("Generating side-by-side 4-panel visualizations for all 16 overfit samples...")
    model.eval()
    with torch.no_grad():
        for batch_i, (images, masks, metas) in enumerate(eval_loader):
            images_dev = images.to(device)
            logits = model(images_dev)
            probs = torch.sigmoid(logits)
            preds_binary = (probs >= threshold).float()

            for s_idx in range(images.size(0)):
                sid = metas[s_idx]["image_id"]
                save_path = vis_dir / f"{sid}_overfit_comparison.png"
                save_qualitative_comparison(
                    image_tensor=images[s_idx],
                    gt_mask=masks[s_idx],
                    pred_mask=preds_binary[s_idx].cpu(),
                    class_names=DACL10K_CLASSES,
                    save_path=save_path,
                    title=f"Stage 1A Overfit Check: {sid} (Threshold: {threshold})",
                )

    # Verify reload capability
    logger.info("Verifying saved checkpoint reloadability for inference...")
    predictor = BridgeDefectPredictor(
        checkpoint_path=str(chk_dir / "best_overfit_model.pth"),
        device="auto",
        threshold=threshold,
    )
    first_img_path = train_ds.image_paths[0]
    pred_res = predictor.predict_image(first_img_path)
    logger.info(
        f"Inference reload test successful! Detected classes on {Path(first_img_path).name}: "
        f"{pred_res['detected_classes']}"
    )

    logger.info("=" * 70)
    logger.info("STAGE 1A OVERFIT TEST COMPLETED SUCCESSFULLY!")
    logger.info(f"Loss start -> end: {history['train_loss'][0]} -> {history['train_loss'][-1]}")
    logger.info(f"mIoU start -> end: {history['subset_miou'][0]} -> {history['subset_miou'][-1]}")
    logger.info(f"Dice start -> end: {history['subset_dice'][0]} -> {history['subset_dice'][-1]}")
    logger.info("=" * 70)

    return {
        "history": history,
        "per_class_e1": per_class_history_e1,
        "per_class_e20": per_class_history_e20,
        "best_miou": best_miou,
        "sample_records": sample_records,
    }


if __name__ == "__main__":
    run_overfit_test()
