"""
Script: scripts/generate_qualitative_figures.py
Description:
    Stage 6.2: Reproducible Qualitative Segmentation Evaluation.
    
    Generates publication-quality 4-panel figures at 300 DPI comparing:
    Column 1: Original bridge inspection image (unnormalized RGB)
    Column 2: Ground-truth segmentation mask (target defect class in matching color)
    Column 3: Experiment 02 prediction mask (DeepLabV3+ ResNet34 at threshold tau=0.50)
    Column 4: Error analysis breakdown (True Positives, False Positives, False Negatives)

    Target focus classes:
    - Crack
    - ACrack (Alligator Crack)
    - Spalling
    - Rust
    - ExposedRebars

    Reproducible selection:
    Uses a bounded candidate subset to keep CPU execution practical.
    Selects representative successful and challenging/failed cases per class
    based on measured per-image IoU and pixel errors without cherry-picking.
    
    Artifacts Saved:
    - 300 DPI PNG figures in --output-dir
    - qualitative_evaluation_metadata.csv with complete sample IDs and metrics
    - qualitative_evaluation_summary.json
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import (
    DACL10KDataset,
    DACL10K_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    custom_collate_fn,
)
from src.measurement.visualization import CLASS_COLOR_PALETTE_RGB
from src.segmentation.model import BridgeSegmentationModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("qualitative_eval")

DEFAULT_FOCUS_CLASSES = ["Crack", "ACrack", "Spalling", "Rust", "ExposedRebars"]

# Color palette for error visualization (RGB uint8)
COLOR_TP = (40, 220, 40)    # Lime Green
COLOR_FP = (220, 40, 40)    # Crimson Red
COLOR_FN = (0, 210, 255)    # Vivid Cyan


def find_dataset_root(candidate_path: Optional[str] = None) -> Path:
    """Discovers the DACL10K-v2 dataset root directory across common locations."""
    if candidate_path and Path(candidate_path).exists():
        return Path(candidate_path).resolve()

    search_paths = [
        Path("/Users/kundankumar/Downloads/dacl10k_v2_devphase"),
        REPO_ROOT / "data" / "raw" / "dacl10k_v2",
        REPO_ROOT / "data" / "dacl10k_v2",
        Path.home() / "Downloads" / "dacl10k_v2_devphase",
    ]
    for p in search_paths:
        if p.exists() and (p / "images" / "validation").exists():
            return p.resolve()

    if candidate_path:
        raise FileNotFoundError(f"Specified dataset root directory does not exist: {candidate_path}")
    raise FileNotFoundError(
        "Could not automatically locate DACL10K-v2 dataset root. "
        "Please specify --root-dir with the path to the dataset directory containing images/ and annotations/."
    )


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Loads the DeepLabV3+ ResNet34 model from checkpoint."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_absolute():
        ckpt_file = (REPO_ROOT / checkpoint_path).resolve()

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    logger.info(f"Loading checkpoint from: {ckpt_file}")
    checkpoint = torch.load(str(ckpt_file), map_location=device)

    model_state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
    epoch = checkpoint.get("epoch", "unknown")
    best_miou = checkpoint.get("best_miou", checkpoint.get("best_val_miou", "unknown"))
    logger.info(f"Loaded checkpoint from epoch {epoch} (Recorded mIoU: {best_miou})")

    model = BridgeSegmentationModel(
        architecture="DeepLabV3Plus",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=19,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


def scan_candidate_images(
    root_dir: Path,
    focus_classes: List[str],
    max_candidates_per_class: int = 15,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Scans validation annotations to find images containing the focus classes.
    Returns:
        class_candidates: Dict mapping class_name -> list of image stem IDs containing GT.
        candidate_pool: Bounded list of unique image stems to evaluate with model.
    """
    anno_dir = root_dir / "annotations" / "validation"
    if not anno_dir.exists():
        raise FileNotFoundError(f"Validation annotation directory not found at: {anno_dir}")

    json_files = sorted(list(anno_dir.glob("*.json")))
    logger.info(f"Scanning {len(json_files)} validation annotations for focus classes: {focus_classes}")

    class_all_stems: Dict[str, List[str]] = {c: [] for c in focus_classes}
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            labels = {s.get("label") for s in data.get("shapes", []) if isinstance(s, dict)}
            for c in focus_classes:
                if c in labels:
                    class_all_stems[c].append(jf.stem)
        except Exception as e:
            logger.warning(f"Error reading annotation file {jf}: {e}")

    rng = random.Random(seed)
    class_candidates: Dict[str, List[str]] = {}
    candidate_pool_set = set()

    for c in focus_classes:
        stems = sorted(class_all_stems[c])
        if not stems:
            logger.warning(f"No validation annotations found for focus class: {c}")
            class_candidates[c] = []
            continue

        # Select a bounded reproducible pool for each class
        if len(stems) <= max_candidates_per_class:
            selected = stems
        else:
            # Deterministic seeded selection to get varied distribution
            selected = rng.sample(stems, max_candidates_per_class)
            selected.sort()

        class_candidates[c] = selected
        candidate_pool_set.update(selected)
        logger.info(f"Class '{c}': found {len(stems)} GT occurrences, selected {len(selected)} candidate samples.")

    candidate_pool = sorted(list(candidate_pool_set))
    logger.info(f"Total bounded candidate pool to evaluate: {len(candidate_pool)} images")
    return class_candidates, candidate_pool


def evaluate_candidates(
    model: torch.nn.Module,
    dataset_root: Path,
    candidate_stems: List[str],
    focus_classes: List[str],
    device: torch.device,
    threshold: float = 0.50,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Runs model inference over candidate images and computes per-image metrics for focus classes.
    """
    sample_filenames = [f"{stem}.jpg" for stem in candidate_stems]
    val_dataset = DACL10KDataset(
        root_dir=dataset_root,
        split="validation",
        sample_filenames=sample_filenames,
        augment=False,
        return_metadata=True,
    )

    results_by_class: Dict[str, List[Dict[str, Any]]] = {c: [] for c in focus_classes}

    logger.info(f"Evaluating {len(val_dataset)} candidate images on device: {device}")
    with torch.no_grad():
        for i in range(len(val_dataset)):
            img_tensor, mask_tensor, meta = val_dataset[i]
            sample_id = meta["image_id"]

            logits = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits)[0]
            pred_masks = (probs >= threshold).cpu().numpy().astype(np.uint8)
            gt_masks = mask_tensor.numpy().astype(np.uint8)

            for c in focus_classes:
                c_idx = DACL10K_CLASSES.index(c)
                gt = gt_masks[c_idx]
                pred = pred_masks[c_idx]

                gt_px = int(gt.sum())
                pred_px = int(pred.sum())

                # Skip if neither GT nor prediction exists for this class
                if gt_px == 0 and pred_px == 0:
                    continue

                tp = int(np.logical_and(gt == 1, pred == 1).sum())
                fp = int(np.logical_and(gt == 0, pred == 1).sum())
                fn = int(np.logical_and(gt == 1, pred == 0).sum())

                denom = tp + fp + fn
                iou = float(tp / denom) if denom > 0 else 0.0
                dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

                total_pixels = 512 * 512
                record = {
                    "sample_id": sample_id,
                    "dataset_index": i,
                    "target_class": c,
                    "class_index": c_idx,
                    "gt_pixels": gt_px,
                    "pred_pixels": pred_px,
                    "gt_area_pct": round(float(gt_px / total_pixels * 100.0), 4),
                    "pred_area_pct": round(float(pred_px / total_pixels * 100.0), 4),
                    "tp_pixels": tp,
                    "fp_pixels": fp,
                    "fn_pixels": fn,
                    "iou": round(iou, 4),
                    "dice": round(dice, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "image_tensor": img_tensor,
                    "gt_mask": gt,
                    "pred_mask": pred,
                    "has_gt": bool(gt_px > 0),
                }
                results_by_class[c].append(record)

    return results_by_class


def select_representative_cases(
    class_results: Dict[str, List[Dict[str, Any]]],
    focus_classes: List[str],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Selects representative successful and challenging/failed cases for each focus class.
    Criteria:
    - Successful: Ground Truth present (gt_pixels > 0), highest or top-tier IoU.
    - Challenging/Failed: Ground Truth present (gt_pixels > 0), lowest or low-tier IoU (missed / boundary / FP error).
    """
    rng = random.Random(seed)
    selected_cases: List[Dict[str, Any]] = []

    for c in focus_classes:
        candidates = class_results.get(c, [])
        # Prefer candidates that have ground truth present for meaningful defect evaluation
        gt_candidates = [item for item in candidates if item["has_gt"]]
        if not gt_candidates:
            gt_candidates = candidates

        if not gt_candidates:
            logger.warning(f"No evaluation results available for class '{c}'")
            continue

        # Sort by IoU descending
        sorted_by_iou = sorted(gt_candidates, key=lambda x: x["iou"], reverse=True)

        # 1. Success case: Highest IoU
        success_case = dict(sorted_by_iou[0])
        success_case["outcome_type"] = "success"
        success_case["outcome_label"] = "Successful Prediction"
        selected_cases.append(success_case)

        # 2. Failure / Challenging case: Lowest IoU
        # If there are multiple, pick from bottom with gt_pixels > 0
        failure_case = dict(sorted_by_iou[-1])
        # If best and worst happen to be the exact same sample (e.g. only 1 sample), still label as challenging
        if failure_case["sample_id"] == success_case["sample_id"] and len(sorted_by_iou) > 1:
            failure_case = dict(sorted_by_iou[1])

        failure_case["outcome_type"] = "failure"
        failure_case["outcome_label"] = "Challenging / Failed Prediction"
        selected_cases.append(failure_case)

        logger.info(
            f"Class '{c}' selected: "
            f"Success={success_case['sample_id']} (IoU: {success_case['iou']:.4f}), "
            f"Failure={failure_case['sample_id']} (IoU: {failure_case['iou']:.4f})"
        )

    return selected_cases


def unnormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Converts normalized PyTorch image tensor (3, H, W) back to uint8 RGB (H, W, 3)."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def render_qualitative_figure(
    case: Dict[str, Any],
    output_path: Path,
    threshold: float = 0.50,
    dpi: int = 300,
) -> None:
    """
    Renders a publication-quality 4-panel figure at 300 DPI:
    Panel 1: Original bridge inspection image
    Panel 2: Ground-truth segmentation mask
    Panel 3: Experiment 02 prediction mask
    Panel 4: Error visualization (True Positives, False Positives, False Negatives)
    """
    img = unnormalize_image(case["image_tensor"])
    gt_mask = case["gt_mask"]
    pred_mask = case["pred_mask"]
    class_name = case["target_class"]
    c_idx = case["class_index"]
    sample_id = case["sample_id"]
    outcome_label = case["outcome_label"]
    iou = case["iou"]
    dice = case["dice"]
    prec = case["precision"]
    rec = case["recall"]
    gt_px = case["gt_pixels"]
    pred_px = case["pred_pixels"]
    tp = case["tp_pixels"]
    fp = case["fp_pixels"]
    fn = case["fn_pixels"]

    total_px = 512 * 512
    gt_pct = (gt_px / total_px) * 100.0
    pred_pct = (pred_px / total_px) * 100.0
    class_color_rgb = CLASS_COLOR_PALETTE_RGB[c_idx]

    # Panel 2: Ground-Truth Overlay
    gt_overlay = img.copy()
    if np.any(gt_mask == 1):
        gt_bool = gt_mask == 1
        gt_overlay[gt_bool] = (0.45 * img[gt_bool] + 0.55 * np.array(class_color_rgb)).astype(np.uint8)
        cnts, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_overlay, cnts, -1, class_color_rgb, 2)

    # Panel 3: Prediction Overlay
    pred_overlay = img.copy()
    if np.any(pred_mask == 1):
        pred_bool = pred_mask == 1
        pred_overlay[pred_bool] = (0.45 * img[pred_bool] + 0.55 * np.array(class_color_rgb)).astype(np.uint8)
        cnts, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pred_overlay, cnts, -1, class_color_rgb, 2)

    # Panel 4: Error Overlay (TP, FP, FN)
    err_overlay = img.copy()
    tp_bool = (gt_mask == 1) & (pred_mask == 1)
    fp_bool = (gt_mask == 0) & (pred_mask == 1)
    fn_bool = (gt_mask == 1) & (pred_mask == 0)

    if np.any(tp_bool):
        err_overlay[tp_bool] = (0.40 * img[tp_bool] + 0.60 * np.array(COLOR_TP)).astype(np.uint8)
        cnts, _ = cv2.findContours(tp_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(err_overlay, cnts, -1, COLOR_TP, 2)

    if np.any(fp_bool):
        err_overlay[fp_bool] = (0.40 * img[fp_bool] + 0.60 * np.array(COLOR_FP)).astype(np.uint8)
        cnts, _ = cv2.findContours(fp_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(err_overlay, cnts, -1, COLOR_FP, 2)

    if np.any(fn_bool):
        err_overlay[fn_bool] = (0.40 * img[fn_bool] + 0.60 * np.array(COLOR_FN)).astype(np.uint8)
        cnts, _ = cv2.findContours(fn_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(err_overlay, cnts, -1, COLOR_FN, 2)

    # Setup publication figure layout (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=dpi)

    # (a) Original image
    axes[0].imshow(img)
    axes[0].set_title(f"(a) Original Bridge Image\nID: {sample_id}", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # (b) Ground-Truth
    axes[1].imshow(gt_overlay)
    axes[1].set_title(
        f"(b) Ground-Truth Annotation\nClass: {class_name} ({gt_px:,} px, {gt_pct:.2f}%)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    axes[1].axis("off")
    gt_patch = mpatches.Patch(color=np.array(class_color_rgb) / 255.0, label=f"GT: {class_name}")
    axes[1].legend(handles=[gt_patch], loc="lower right", framealpha=0.88, fontsize=9)

    # (c) Prediction
    axes[2].imshow(pred_overlay)
    axes[2].set_title(
        f"(c) Experiment 02 Prediction (\u03c4={threshold:.2f})\nClass: {class_name} ({pred_px:,} px, {pred_pct:.2f}%)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    axes[2].axis("off")
    pred_patch = mpatches.Patch(color=np.array(class_color_rgb) / 255.0, label=f"Pred: {class_name}")
    axes[2].legend(handles=[pred_patch], loc="lower right", framealpha=0.88, fontsize=9)

    # (d) Error Analysis
    axes[3].imshow(err_overlay)
    axes[3].set_title(
        f"(d) Error Analysis Breakdown\nIoU: {iou:.4f} | Dice (F1): {dice:.4f}",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    axes[3].axis("off")

    tp_patch = mpatches.Patch(color=np.array(COLOR_TP) / 255.0, label=f"True Positive (TP): {tp:,} px")
    fp_patch = mpatches.Patch(color=np.array(COLOR_FP) / 255.0, label=f"False Positive (FP): {fp:,} px")
    fn_patch = mpatches.Patch(color=np.array(COLOR_FN) / 255.0, label=f"False Negative (FN): {fn:,} px")
    axes[3].legend(handles=[tp_patch, fp_patch, fn_patch], loc="lower right", framealpha=0.88, fontsize=8.5)

    # Main Figure Title
    fig.suptitle(
        f"Stage 6.2 Qualitative Evaluation: {class_name} | {outcome_label.upper()} | Sample: {sample_id}\n"
        f"IoU = {iou:.4f} | Dice = {dice:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f}",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved publication figure: {output_path} ({dpi} DPI)")


def generate_qualitative_evaluation(
    root_dir: Optional[str] = None,
    checkpoint: str = "experiments/exp_02_sqrt_weighted_bcedice/checkpoints/best_model.pth",
    output_dir: str = "reports/qualitative_evaluation_stage_6_2",
    device_name: str = "auto",
    threshold: float = 0.50,
    seed: int = 42,
    focus_classes: Optional[List[str]] = None,
    max_candidates_per_class: int = 15,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Main orchestration routine for Stage 6.2 qualitative evaluation.
    """
    classes = focus_classes if focus_classes is not None else DEFAULT_FOCUS_CLASSES
    dataset_root = find_dataset_root(root_dir)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    logger.info("=" * 75)
    logger.info("STAGE 6.2: REPRODUCIBLE QUALITATIVE SEGMENTATION EVALUATION")
    logger.info(f"Dataset Root:    {dataset_root}")
    logger.info(f"Checkpoint:      {checkpoint}")
    logger.info(f"Output Dir:      {out_dir}")
    logger.info(f"Device:          {device}")
    logger.info(f"Threshold:       {threshold}")
    logger.info(f"Random Seed:     {seed}")
    logger.info(f"Focus Classes:   {classes}")
    logger.info(f"DPI:             {dpi}")
    logger.info("=" * 75)

    # 1. Load trained model
    model = load_model(checkpoint, device)

    # 2. Discover candidates reproducibly
    class_candidates, candidate_pool = scan_candidate_images(
        root_dir=dataset_root,
        focus_classes=classes,
        max_candidates_per_class=max_candidates_per_class,
        seed=seed,
    )

    # 3. Run inference on bounded candidate pool
    class_eval_results = evaluate_candidates(
        model=model,
        dataset_root=dataset_root,
        candidate_stems=candidate_pool,
        focus_classes=classes,
        device=device,
        threshold=threshold,
    )

    # 4. Select representative successful and failure cases
    selected_cases = select_representative_cases(
        class_results=class_eval_results,
        focus_classes=classes,
        seed=seed,
    )

    # 5. Render publication-quality figures & assemble metadata
    metadata_rows: List[Dict[str, Any]] = []

    for case in selected_cases:
        c_name = case["target_class"]
        outcome = case["outcome_type"]
        sample_id = case["sample_id"]
        fname = f"qualitative_{c_name.lower()}_{outcome}_{sample_id}.png"
        fig_path = out_dir / fname

        render_qualitative_figure(
            case=case,
            output_path=fig_path,
            threshold=threshold,
            dpi=dpi,
        )

        metadata_rows.append({
            "sample_id": sample_id,
            "target_class": c_name,
            "outcome_type": outcome,
            "outcome_label": case["outcome_label"],
            "iou": case["iou"],
            "dice": case["dice"],
            "precision": case["precision"],
            "recall": case["recall"],
            "tp_pixels": case["tp_pixels"],
            "fp_pixels": case["fp_pixels"],
            "fn_pixels": case["fn_pixels"],
            "gt_pixels": case["gt_pixels"],
            "pred_pixels": case["pred_pixels"],
            "gt_area_pct": case["gt_area_pct"],
            "pred_area_pct": case["pred_area_pct"],
            "threshold": threshold,
            "checkpoint": str(checkpoint),
            "figure_filename": fname,
            "figure_path": str(fig_path),
            "random_seed": seed,
            "dpi": dpi,
        })

    # 6. Save metadata CSV and JSON
    meta_df = pd.DataFrame(metadata_rows)
    csv_path = out_dir / "qualitative_evaluation_metadata.csv"
    meta_df.to_csv(csv_path, index=False)
    logger.info(f"Saved evaluation metadata CSV: {csv_path} ({len(meta_df)} rows)")

    summary_path = out_dir / "qualitative_evaluation_summary.json"
    summary_data = {
        "title": "Stage 6.2 Qualitative Segmentation Evaluation Summary",
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "threshold": threshold,
        "seed": seed,
        "dpi": dpi,
        "total_figures_generated": len(metadata_rows),
        "focus_classes": classes,
        "records": [
            {
                k: v for k, v in row.items()
                if k not in ["figure_path"]
            }
            for row in metadata_rows
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved evaluation summary JSON: {summary_path}")

    return {
        "output_dir": str(out_dir),
        "metadata_csv": str(csv_path),
        "summary_json": str(summary_path),
        "figures": [r["figure_path"] for r in metadata_rows],
        "metadata": metadata_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 6.2: Reproducible Qualitative Segmentation Evaluation for Bridge Defect Inspection."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Path to DACL10K-v2 dataset root (default: auto-detected from common locations).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/exp_02_sqrt_weighted_bcedice/checkpoints/best_model.pth",
        help="Path to trained DeepLabV3+ model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/qualitative_evaluation_stage_6_2",
        help="Directory to save publication figures and metadata CSV.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device for inference (default: auto).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Binarization decision threshold for sigmoid predictions (default: 0.50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for reproducible candidate selection (default: 42).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=15,
        dest="max_candidates_per_class",
        help="Maximum candidate images evaluated per focus class (default: 15).",
    )
    parser.add_argument(
        "--focus-classes",
        type=str,
        default="Crack,ACrack,Spalling,Rust,ExposedRebars",
        help="Comma-separated list of defect classes to evaluate.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI for saved figures (default: 300).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    focus_classes = [c.strip() for c in args.focus_classes.split(",") if c.strip()]
    generate_qualitative_evaluation(
        root_dir=args.root_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        threshold=args.threshold,
        seed=args.seed,
        focus_classes=focus_classes,
        max_candidates_per_class=args.max_candidates_per_class,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
