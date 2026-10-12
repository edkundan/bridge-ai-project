"""
Module: src.segmentation.evaluate
Description:
    Comprehensive evaluation pipeline for multi-label semantic bridge defect segmentation.
    
    Evaluates:
    - Per-class Intersection over Union (IoU)
    - Mean IoU (mIoU) across all 19 classes
    - Present-class mIoU (macro average strictly over classes with GT occurrences)
    - Per-class Dice (F1) scores
    - Per-class Precision and Recall
    - Pixel-level ground-truth coverage and prediction coverage
    
    Artifacts Saved:
    - CSV of per-class metrics
    - JSON summary of aggregate metrics
    - Qualitative visualizations comparing Ground Truth vs Predictions
    - Failure cases (lowest IoU samples)
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DACL10KDataset, DACL10K_CLASSES, custom_collate_fn
from src.segmentation.model import BridgeSegmentationModel

logger = logging.getLogger(__name__)

# Fixed visually distinguishable colormap for qualitative overlays
np.random.seed(42)
OVERLAY_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))[:19]


class SegmentationEvaluator:
    """
    Computes multi-label semantic segmentation metrics per class and macro aggregates.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        eps: float = 1e-7,
    ):
        """
        Args:
            class_names: List of 19 class names.
            threshold: Probability threshold for binarizing sigmoid predictions.
            eps: Numerical stability constant for avoiding division by zero.
        """
        self.class_names = class_names if class_names is not None else list(DACL10K_CLASSES)
        self.num_classes = len(self.class_names)
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        """Reset accumulated confusion matrix counts."""
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)
        self.tn = np.zeros(self.num_classes, dtype=np.int64)
        self.sample_ious: List[Dict[str, Any]] = []

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        sample_ids: Optional[List[str]] = None,
        is_logits: bool = True,
    ):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            preds: Model output logits or probabilities (B, C, H, W).
            targets: Binary ground truth tensor (B, C, H, W) in {0, 1}.
            sample_ids: Optional list of sample ID strings for per-sample tracking.
            is_logits: If True, applies sigmoid to convert raw logits to probabilities in [0, 1].
        """
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}")

        # Ensure probability conversion via sigmoid
        if is_logits:
            probs = torch.sigmoid(preds)
        else:
            probs = torch.clamp(preds, 0.0, 1.0)


        # Binarize with threshold
        pred_binary = (probs >= self.threshold).cpu().numpy().astype(np.uint8)
        gt_binary = (targets >= 0.5).cpu().numpy().astype(np.uint8)

        b, c = pred_binary.shape[:2]

        for i in range(b):
            sample_tp = np.zeros(c, dtype=np.int64)
            sample_fp = np.zeros(c, dtype=np.int64)
            sample_fn = np.zeros(c, dtype=np.int64)

            for cls_idx in range(c):
                p = pred_binary[i, cls_idx]
                g = gt_binary[i, cls_idx]

                cur_tp = int(np.sum((p == 1) & (g == 1)))
                cur_fp = int(np.sum((p == 1) & (g == 0)))
                cur_fn = int(np.sum((p == 0) & (g == 1)))
                cur_tn = int(np.sum((p == 0) & (g == 0)))

                self.tp[cls_idx] += cur_tp
                self.fp[cls_idx] += cur_fp
                self.fn[cls_idx] += cur_fn
                self.tn[cls_idx] += cur_tn

                sample_tp[cls_idx] = cur_tp
                sample_fp[cls_idx] = cur_fp
                sample_fn[cls_idx] = cur_fn

            # Per-sample mean IoU tracking for failure case analysis
            sample_denom = sample_tp + sample_fp + sample_fn
            present_mask = sample_denom > 0
            if np.any(present_mask):
                sample_miou = float(np.mean(sample_tp[present_mask] / (sample_denom[present_mask] + self.eps)))
            else:
                sample_miou = 1.0  # Perfect match on empty sample

            sid = sample_ids[i] if (sample_ids is not None and i < len(sample_ids)) else f"sample_{len(self.sample_ious)}"
            self.sample_ious.append({
                "sample_id": sid,
                "sample_index": len(self.sample_ious),
                "miou": sample_miou,
            })

    def compute(self) -> Dict[str, Any]:
        """
        Compute final per-class and macro-aggregated metrics.
        
        Returns:
            Dictionary containing:
                - per_class: List of dicts with metrics per class
                - macro: Aggregate summary dict (mIoU, macro_dice, macro_precision, macro_recall)
        """
        per_class_results = []
        ious = []
        dices = []
        precisions = []
        recalls = []

        present_ious = []
        present_dices = []

        for c, cls_name in enumerate(self.class_names):
            tp_c = self.tp[c]
            fp_c = self.fp[c]
            fn_c = self.fn[c]
            tn_c = self.tn[c]

            gt_pixels = tp_c + fn_c
            pred_pixels = tp_c + fp_c

            # IoU = TP / (TP + FP + FN + eps)
            iou_denom = tp_c + fp_c + fn_c
            if iou_denom == 0:
                # Class absent from both GT and prediction: IoU is 1.0
                iou = 1.0
            else:
                iou = float(tp_c / (iou_denom + self.eps))

            # Dice = 2 * TP / (2 * TP + FP + FN + eps)
            dice_denom = 2 * tp_c + fp_c + fn_c
            if dice_denom == 0:
                dice = 1.0
            else:
                dice = float((2 * tp_c) / (dice_denom + self.eps))

            # Precision = TP / (TP + FP + eps)
            precision = float(tp_c / (pred_pixels + self.eps)) if pred_pixels > 0 else 0.0

            # Recall = TP / (TP + FN + eps)
            recall = float(tp_c / (gt_pixels + self.eps)) if gt_pixels > 0 else 0.0

            cls_record = {
                "class_index": c,
                "class_name": cls_name,
                "iou": round(iou, 4),
                "dice": round(dice, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "gt_pixels": int(gt_pixels),
                "pred_pixels": int(pred_pixels),
                "tp": int(tp_c),
                "fp": int(fp_c),
                "fn": int(fn_c),
                "is_present_in_gt": bool(gt_pixels > 0),
            }
            per_class_results.append(cls_record)

            ious.append(iou)
            dices.append(dice)
            precisions.append(precision)
            recalls.append(recall)

            if gt_pixels > 0:
                present_ious.append(iou)
                present_dices.append(dice)

        macro_summary = {
            "mIoU_all_classes": round(float(np.mean(ious)), 4),
            "mIoU_present_classes": round(float(np.mean(present_ious)) if present_ious else 0.0, 4),
            "macro_dice": round(float(np.mean(dices)), 4),
            "macro_dice_present": round(float(np.mean(present_dices)) if present_dices else 0.0, 4),
            "macro_precision": round(float(np.mean(precisions)), 4),
            "macro_recall": round(float(np.mean(recalls)), 4),
            "threshold": self.threshold,
            "classes_evaluated": self.num_classes,
            "classes_present_in_split": len(present_ious),
        }

        return {
            "per_class": per_class_results,
            "macro": macro_summary,
            "sample_ious": self.sample_ious,
        }


def save_qualitative_comparison(
    image_tensor: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "Segmentation Qualitative Comparison",
):
    """
    Saves a 4-panel figure:
    (1) Original Image, (2) Ground-Truth Masks, (3) Predicted Masks, (4) Predicted Overlays.
    
    Args:
        image_tensor: Normalized (3, H, W) tensor.
        gt_mask: Binary (19, H, W) tensor.
        pred_mask: Binary (19, H, W) tensor.
        class_names: List of class names.
        save_path: Destination image path.
        title: Plot title.
    """
    # Unnormalize image tensor: (3, H, W) -> (H, W, 3)
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.clip((img * std + mean) * 255.0, 0, 255).astype(np.uint8)

    gt_np = gt_mask.cpu().numpy().astype(np.uint8)
    pred_np = pred_mask.cpu().numpy().astype(np.uint8)

    h, w = img.shape[:2]
    gt_mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pred_mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pred_overlay = img.copy()

    gt_classes = []
    pred_classes = []

    for c_idx, c_name in enumerate(class_names):
        color = (OVERLAY_COLORS[c_idx][:3] * 255).astype(np.uint8)

        # Ground truth channel
        if np.any(gt_np[c_idx] > 0):
            gt_classes.append(c_name)
            mask_c = gt_np[c_idx] > 0
            gt_mask_canvas[mask_c] = color

        # Prediction channel
        if np.any(pred_np[c_idx] > 0):
            pred_classes.append(c_name)
            mask_c = pred_np[c_idx] > 0
            pred_mask_canvas[mask_c] = color
            colored_mask = np.zeros_like(img)
            colored_mask[mask_c] = color
            pred_overlay = cv2.addWeighted(pred_overlay, 0.7, colored_mask, 0.3, 0)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=11, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_mask_canvas)
    axes[1].set_title(f"Ground-Truth Masks\n({len(gt_classes)} classes: {', '.join(gt_classes[:2]) if gt_classes else 'None'})", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(pred_mask_canvas)
    axes[2].set_title(f"Predicted Masks\n({len(pred_classes)} classes: {', '.join(pred_classes[:2]) if pred_classes else 'None'})", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(pred_overlay)
    axes[3].set_title(f"Predicted Overlays\n(Threshold = 0.5)", fontsize=10)
    axes[3].axis("off")

    plt.suptitle(title, fontsize=13, weight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)



def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Union[str, Path],
    threshold: float = 0.5,
    save_visualizations: int = 10,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates segmentation model over dataloader and persists metrics.
    
    Args:
        model: PyTorch segmentation model.
        dataloader: Validation DataLoader.
        device: torch.device ('mps', 'cuda', 'cpu').
        output_dir: Directory to save CSV, JSON, and PNG artifacts.
        threshold: Decision threshold for sigmoid predictions.
        save_visualizations: Number of sample comparisons to save.
        class_names: List of class names.
        
    Returns:
        Evaluation summary metrics dict.
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "qualitative_results"
    vis_dir.mkdir(parents=True, exist_ok=True)

    classes = class_names if class_names is not None else list(DACL10K_CLASSES)
    evaluator = SegmentationEvaluator(class_names=classes, threshold=threshold)

    vis_saved = 0
    batch_idx = 0

    logger.info(f"Starting evaluation across {len(dataloader)} batches (threshold={threshold}).")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 3:
                images, masks, metas = batch
                sample_ids = [m["image_id"] for m in metas]
            else:
                images, masks = batch
                sample_ids = [f"batch_{batch_idx}_sample_{i}" for i in range(len(images))]

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            evaluator.update(logits, masks, sample_ids=sample_ids)

            # Save qualitative visual comparisons
            if vis_saved < save_visualizations:
                probs = torch.sigmoid(logits)
                preds_binary = (probs >= threshold).float()
                for b_i in range(images.size(0)):
                    if vis_saved >= save_visualizations:
                        break
                    sid = sample_ids[b_i]
                    save_p = vis_dir / f"val_{vis_saved:03d}_{sid}_comparison.png"
                    save_qualitative_comparison(
                        images[b_i], masks[b_i], preds_binary[b_i],
                        classes, save_p, title=f"Sample: {sid} (Threshold: {threshold})"
                    )
                    vis_saved += 1

            batch_idx += 1

    results = evaluator.compute()

    # Save per-class metrics to CSV
    df = pd.DataFrame(results["per_class"])
    csv_path = output_dir / "per_class_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Per-class metrics saved to: {csv_path}")

    # Save aggregate metrics to JSON
    json_path = output_dir / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation summary saved to: {json_path}")

    # Identify and save failure cases (lowest IoU samples)
    sample_df = pd.DataFrame(results["sample_ious"]).sort_values("miou", ascending=True)
    failures_csv = output_dir / "sample_miou_ranking.csv"
    sample_df.to_csv(failures_csv, index=False)

    logger.info(
        f"Evaluation Complete. Macro mIoU (all): {results['macro']['mIoU_all_classes']:.4f} | "
        f"Macro mIoU (present): {results['macro']['mIoU_present_classes']:.4f}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bridge Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="experiments/evaluation_results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device selection
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load model
    model = BridgeSegmentationModel(
        architecture=config.get("model", {}).get("architecture", "DeepLabV3Plus"),
        encoder_name=config.get("model", {}).get("encoder_name", "resnet34"),
        encoder_weights=None,
        num_classes=19,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint from: {args.checkpoint}")

    # Build validation dataloader
    dataset_cfg = config.get("dataset", {})
    root_dir = dataset_cfg.get("root_dir")
    target_size = tuple(config.get("training", {}).get("image_size", [512, 512]))
    batch_size = config.get("training", {}).get("batch_size", 4)

    val_dataset = DACL10KDataset(
        root_dir=root_dir,
        split="validation",
        target_size=target_size,
        max_samples=args.max_samples,
        return_metadata=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
    )

    evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        output_dir=args.output_dir,
        threshold=args.threshold,
        save_visualizations=10,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
