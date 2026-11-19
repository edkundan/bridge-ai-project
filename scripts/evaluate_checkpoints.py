"""
Script: scripts/evaluate_checkpoints.py
Description:
    Reproducible, deterministic evaluation script for DACL10K-v2 checkpoints.
    Processes exactly all 975 validation images (or configurable sample limit).
    Calculates:
        - Per-class TP (Intersection), Union, IoU, Dice, Precision, Recall
        - Arithmetic mean of 19 per-class IoUs
        - Macro Dice (arithmetic mean of 19 per-class Dice scores)
        - Evaluator reported aggregate metrics
        - Mathematical consistency verification
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.dataset import DACL10KDataset, DACL10K_CLASSES, custom_collate_fn
from src.segmentation.evaluate import SegmentationEvaluator
from src.segmentation.model import BridgeSegmentationModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate_checkpoint")


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_root: str,
    output_dir: str,
    threshold: float = 0.5,
    batch_size: int = 8,
    num_workers: int = 2,
    device_name: str = "auto",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device selection
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    logger.info(f"Using device: {device}")
    abs_ckpt_path = os.path.abspath(checkpoint_path)
    logger.info(f"Checkpoint file: {abs_ckpt_path}")

    if not os.path.exists(abs_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {abs_ckpt_path}")

    chk = torch.load(abs_ckpt_path, map_location=device)
    model_state = chk.get("model_state_dict", chk.get("model_state", chk))
    epoch = chk.get("epoch", "unknown")
    best_miou = chk.get("best_miou", chk.get("best_val_miou", chk.get("val_miou", "unknown")))
    has_opt = ("optimizer_state_dict" in chk or "optimizer_state" in chk)
    has_sched = ("scheduler_state_dict" in chk or "scheduler_state" in chk)

    logger.info(f"Checkpoint Saved Epoch: {epoch}")
    logger.info(f"Checkpoint Recorded mIoU: {best_miou}")
    logger.info(f"Has Optimizer State: {has_opt}")
    logger.info(f"Has Scheduler State: {has_sched}")

    # 2. Build model
    model = BridgeSegmentationModel(
        architecture="DeepLabV3Plus",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=19,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    # 3. Build validation dataset & loader
    val_dataset = DACL10KDataset(
        root_dir=dataset_root,
        split="validation",
        target_size=(512, 512),
        augment=False,
        max_samples=max_samples,
        return_metadata=True,
    )
    total_images = len(val_dataset)
    num_batches = int(np.ceil(total_images / batch_size))
    total_pixels_per_class = total_images * 512 * 512

    logger.info("=" * 70)
    logger.info(f"EVALUATION PASS: {total_images} validation images ({num_batches} batches, batch_size={batch_size})")
    logger.info(f"Total Pixels Per Channel: {total_pixels_per_class:,} (512x512)")
    logger.info(f"Threshold: {threshold}")
    logger.info("=" * 70)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    evaluator = SegmentationEvaluator(
        class_names=list(DACL10K_CLASSES),
        threshold=threshold,
        eps=1e-7,
    )

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating Epoch {epoch}"):
            if len(batch) == 3:
                images, masks, metas = batch
                sample_ids = [m["image_id"] for m in metas]
            else:
                images, masks = batch
                sample_ids = None

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            evaluator.update(logits, masks, sample_ids=sample_ids, is_logits=True)

    elapsed_time = time.time() - start_time
    logger.info(f"Evaluation complete in {elapsed_time:.2f} seconds ({elapsed_time/total_images:.3f} s/image).")

    # 4. Compute metrics
    metrics = evaluator.compute()
    per_class = metrics["per_class"]
    macro = metrics["macro"]

    # Calculate per-class table and arithmetic averages
    rows = []
    ious = []
    dices = []
    precisions = []
    recalls = []

    for c in per_class:
        tp = c["tp"]
        fp = c["fp"]
        fn = c["fn"]
        union = tp + fp + fn
        gt_px = c["gt_pixels"]
        pred_px = c["pred_pixels"]

        iou = c["iou"]
        dice = c["dice"]
        prec = c["precision"]
        rec = c["recall"]

        ious.append(iou)
        dices.append(dice)
        precisions.append(prec)
        recalls.append(rec)

        rows.append({
            "class_index": c["class_index"],
            "class_name": c["class_name"],
            "intersection_tp": tp,
            "union": union,
            "gt_pixels": gt_px,
            "pred_pixels": pred_px,
            "iou": iou,
            "dice": dice,
            "precision": prec,
            "recall": rec,
        })

    arith_mean_iou = float(np.mean(ious))
    arith_mean_dice = float(np.mean(dices))
    arith_mean_prec = float(np.mean(precisions))
    arith_mean_rec = float(np.mean(recalls))

    evaluator_miou = macro["mIoU_all_classes"]
    evaluator_dice = macro["macro_dice_all_classes"]

    miou_diff = abs(evaluator_miou - arith_mean_iou)
    dice_diff = abs(evaluator_dice - arith_mean_dice)

    report_data = {
        "metadata": {
            "checkpoint_path": abs_ckpt_path,
            "checkpoint_saved_epoch": epoch,
            "checkpoint_recorded_miou": best_miou,
            "has_optimizer_state": has_opt,
            "has_scheduler_state": has_sched,
            "total_images_evaluated": total_images,
            "total_pixels_per_class": total_pixels_per_class,
            "threshold": threshold,
            "classes_evaluated": len(DACL10K_CLASSES),
            "device": str(device),
            "evaluation_duration_seconds": round(elapsed_time, 2),
            "metric_definitions": {
                "IoU": "TP / (TP + FP + FN + eps) (or 1.0 if union == 0)",
                "Dice": "2*TP / (2*TP + FP + FN + eps) (or 1.0 if denom == 0)",
                "Precision": "TP / (TP + FP + eps)",
                "Recall": "TP / (TP + FN + eps)",
                "mIoU_all_classes": "Unweighted arithmetic mean of 19 per-class IoU values",
                "macro_dice_all_classes": "Unweighted arithmetic mean of 19 per-class Dice values",
            },
        },
        "aggregate_verification": {
            "arithmetic_mean_19_class_iou": round(arith_mean_iou, 6),
            "evaluator_reported_miou_all_classes": round(evaluator_miou, 6),
            "miou_difference": round(miou_diff, 8),
            "miou_consistent": bool(miou_diff < 1e-4),
            "arithmetic_mean_19_class_dice": round(arith_mean_dice, 6),
            "evaluator_reported_macro_dice": round(evaluator_dice, 6),
            "dice_difference": round(dice_diff, 8),
            "dice_consistent": bool(dice_diff < 1e-4),
            "macro_precision": round(arith_mean_prec, 6),
            "macro_recall": round(arith_mean_rec, 6),
        },
        "per_class": rows,
    }

    # Save JSON and CSV
    json_path = out_dir / "eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    logger.info(f"Saved evaluation JSON: {json_path}")

    csv_path = out_dir / "per_class_table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Saved per-class CSV: {csv_path}")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Checkpoint Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--root-dir", type=str, default="/Users/kundankumar/Downloads/dacl10k_v2_devphase")
    parser.add_argument("--output-dir", type=str, default="reports/checkpoint_eval")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_root=args.root_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        max_samples=args.max_samples,
    )
