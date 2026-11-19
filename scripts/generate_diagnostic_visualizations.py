"""
Script: scripts/generate_diagnostic_visualizations.py
Description:
    Generates class-specific 4-panel diagnostic visualizations for weak and focal classes:
    (1) Original Image (512x512)
    (2) Ground-Truth Mask for target class only (unoccluded, from DACL10KDataset)
    (3) Prediction Mask for target class only (threshold 0.5)
    (4) Color-coded Diagnostic Overlay:
        - True Positive (TP): Bright Green [0, 255, 0]
        - False Positive (FP): Bright Red [255, 0, 0]
        - False Negative (FN): Bright Cyan [0, 255, 255]

    Includes both correctly detected and missed examples for:
    - WConccor (Missed: Sample 0024 & Sample 0055)
    - Rockpocket (Detected: Sample 0017; Missed: Sample 0016)
    - Crack (Detected: Sample 0016; Missed: Sample 0011)
    - ACrack (Detected: Sample 0026; Over-predicted: Sample 0038)
    - Cavity & ExposedRebars (Additional reference)

    Saves high-resolution PNGs to reports/visualizations/
"""

import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.data.dataset import DACL10KDataset, DACL10K_CLASSES

OVERLAY_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))[:19]
CLASS_COLORS = {DACL10K_CLASSES[i]: (OVERLAY_COLORS[i][:3] * 255).astype(int) for i in range(19)}


def create_class_diagnostic_panel(
    source_file_path: Path,
    target_class: str,
    output_path: Path,
    category: str,
    sample_id: str,
    val_dataset: DACL10KDataset
):
    img_raw = cv2.imread(str(source_file_path))
    if img_raw is None:
        raise FileNotFoundError(f"Cannot read: {source_file_path}")
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # Matplotlib subplots bounding boxes from saved 2777x707 figure
    orig_img = img_rgb[73:694, 13:635]
    pred_canvas = img_rgb[73:694, 1433:2055]
    h, w = orig_img.shape[:2]

    # Find sample index in val_dataset from sample_id
    sample_num_match = re.search(r"validation_(\d+)", sample_id)
    if not sample_num_match:
        raise ValueError(f"Could not parse validation index from {sample_id}")
    sample_num = int(sample_num_match.group(1))

    # Retrieve unoccluded ground-truth mask directly from dataset
    c_idx = DACL10K_CLASSES.index(target_class)
    _, mask_t = val_dataset[sample_num]
    gt_512 = mask_t[c_idx].numpy().astype(np.uint8)

    # Resize GT binary mask to match canvas resolution (w, h)
    gt_bin = cv2.resize(gt_512, (w, h), interpolation=cv2.INTER_NEAREST)

    # Extract target class Pred mask from pred_canvas using unique class color
    class_color = CLASS_COLORS[target_class]
    pred_dist = np.linalg.norm(pred_canvas.astype(float) - class_color, axis=2)
    pred_bin = (pred_dist < 20).astype(np.uint8)

    # Error analysis categories
    tp_mask = np.logical_and(gt_bin == 1, pred_bin == 1).astype(np.uint8)
    fp_mask = np.logical_and(gt_bin == 0, pred_bin == 1).astype(np.uint8)
    fn_mask = np.logical_and(gt_bin == 1, pred_bin == 0).astype(np.uint8)

    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    gt_total = int(gt_bin.sum())
    pred_total = int(pred_bin.sum())

    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Panel 2: Ground-truth mask for target class only (binary mask on black background)
    gt_only_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    gt_only_canvas[gt_bin == 1] = [255, 255, 255]

    # Panel 3: Prediction mask for target class only at threshold 0.5
    pred_only_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pred_only_canvas[pred_bin == 1] = [255, 255, 255]

    # Panel 4: Diagnostic TP / FP / FN Overlay on original image
    overlay = (orig_img.astype(float) * 0.65).astype(np.uint8)

    color_tp = np.array([0, 255, 0], dtype=np.uint8)       # Bright Green
    color_fp = np.array([255, 40, 40], dtype=np.uint8)     # Bright Red
    color_fn = np.array([0, 230, 255], dtype=np.uint8)     # Bright Cyan

    alpha = 0.65
    if tp > 0:
        overlay[tp_mask == 1] = (alpha * color_tp + (1 - alpha) * overlay[tp_mask == 1]).astype(np.uint8)
    if fp > 0:
        overlay[fp_mask == 1] = (alpha * color_fp + (1 - alpha) * overlay[fp_mask == 1]).astype(np.uint8)
    if fn > 0:
        overlay[fn_mask == 1] = (alpha * color_fn + (1 - alpha) * overlay[fn_mask == 1]).astype(np.uint8)

    contours_tp, _ = cv2.findContours(tp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_tp, -1, (0, 255, 0), 2)

    contours_fp, _ = cv2.findContours(fp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_fp, -1, (255, 0, 0), 2)

    contours_fn, _ = cv2.findContours(fn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_fn, -1, (0, 255, 255), 2)

    # Create 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.2), dpi=150)
    fig.patch.set_facecolor("#1e1e1e")

    fig_title = (
        f"Class: {target_class} | Sample: {sample_id} | Status: {category}\n"
        f"GT Pixels: {gt_total:,} | Pred Pixels: {pred_total:,} | TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | "
        f"IoU: {iou:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}"
    )
    fig.suptitle(fig_title, color="white", fontsize=13, weight="bold", y=0.98)

    # 1. Original Image
    axes[0].imshow(orig_img)
    axes[0].set_title("(1) Original Image\n(512x512 Letterbox)", color="white", fontsize=11, weight="bold", pad=8)
    axes[0].axis("off")

    # 2. Ground-Truth Mask for Target Class Only
    axes[1].imshow(gt_only_canvas)
    axes[1].set_title(f"(2) Ground Truth ({target_class} Only)\n{gt_total:,} foreground pixels", color="white", fontsize=11, weight="bold", pad=8)
    axes[1].axis("off")

    # 3. Prediction Mask for Target Class Only
    axes[2].imshow(pred_only_canvas)
    axes[2].set_title(f"(3) Model Prediction (tau=0.5)\n{pred_total:,} predicted pixels", color="white", fontsize=11, weight="bold", pad=8)
    axes[2].axis("off")

    # 4. Diagnostic Overlay
    axes[3].imshow(overlay)
    axes[3].set_title("(4) Diagnostic Error Overlay\nTP (Green), FP (Red), FN (Cyan)", color="white", fontsize=11, weight="bold", pad=8)
    axes[3].axis("off")

    patch_tp = mpatches.Patch(color='#00FF00', label=f'TP: {tp:,} px')
    patch_fp = mpatches.Patch(color='#FF2828', label=f'FP: {fp:,} px')
    patch_fn = mpatches.Patch(color='#00E6FF', label=f'FN: {fn:,} px')
    leg = axes[3].legend(
        handles=[patch_tp, patch_fp, patch_fn],
        loc="lower left",
        fontsize=9,
        facecolor="#2b2b2b",
        edgecolor="white"
    )
    for text in leg.get_texts():
        text.set_color("white")

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, wspace=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated diagnostic visualization: {output_path.name} (GT={gt_total:,}, Pred={pred_total:,}, IoU={iou:.4f}, TP={tp:,}, FN={fn:,})")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    downloads_weak = Path("/Users/kundankumar/Downloads/weak_classes")
    output_dir = repo_root / "reports" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    val_dataset = DACL10KDataset(
        root_dir="/Users/kundankumar/Downloads/dacl10k_v2_devphase",
        split="validation",
        target_size=(512, 512),
        augment=False
    )

    cases = [
        # WConccor
        ("WConccor_dacl10k_v2_validation_0024.png", "WConccor", "MISSED (Zero TP / 100% False Negative)", "dacl10k_v2_validation_0024"),
        ("WConccor_dacl10k_v2_validation_0055.png", "WConccor", "MISSED (Zero TP / 100% False Negative)", "dacl10k_v2_validation_0055"),
        # Rockpocket
        ("Rockpocket_dacl10k_v2_validation_0017.png", "Rockpocket", "DETECTED (True Positive Clustered)", "dacl10k_v2_validation_0017"),
        ("Rockpocket_dacl10k_v2_validation_0016.png", "Rockpocket", "MISSED (Zero TP / 100% False Negative)", "dacl10k_v2_validation_0016"),
        # Crack
        ("Crack_dacl10k_v2_validation_0016.png", "Crack", "DETECTED (Partial True Positive)", "dacl10k_v2_validation_0016"),
        ("Crack_dacl10k_v2_validation_0011.png", "Crack", "MISSED (Zero TP / Thin Crack Boundary Failure)", "dacl10k_v2_validation_0011"),
        # ACrack
        ("ACrack_dacl10k_v2_validation_0026.png", "ACrack", "DETECTED (Strong Detection / Large Web)", "dacl10k_v2_validation_0026"),
        ("ACrack_dacl10k_v2_validation_0038.png", "ACrack", "DETECTED (100% Recall with Surface Over-Prediction)", "dacl10k_v2_validation_0038"),
        # Additional reference weak classes
        ("Cavity_dacl10k_v2_validation_0026.png", "Cavity", "DETECTED (True Positive Clustered)", "dacl10k_v2_validation_0026"),
        ("ExposedRebars_dacl10k_v2_validation_0034.png", "ExposedRebars", "DETECTED (Rebar Linear Structure)", "dacl10k_v2_validation_0034"),
    ]

    print("Generating class-specific 4-panel diagnostic visualizations with unoccluded GT...")
    for filename, target_class, category, sample_id in cases:
        src = downloads_weak / filename
        dest = output_dir / f"diagnostic_{target_class}_{sample_id}.png"
        create_class_diagnostic_panel(src, target_class, dest, category, sample_id, val_dataset)

    print(f"\nAll diagnostic visualizations successfully saved to: {output_dir}")


if __name__ == "__main__":
    main()
