"""
Script: scripts/train_class_imbalance_analysis.py
Description:
    Analyzes all 6,935 training annotations for DACL10K-v2:
    - Image occurrences and percentages
    - Total native and 512x512 letterbox foreground pixels
    - Polygon and mask area distributions (mean, median, percentiles)
    - Small (<1024 px), Medium (1024-9216 px), Large (>9216 px) defect breakdown
    - 19x19 class co-occurrence matrix and conditional probabilities
    - Training-only candidate class weights (Inverse, Sqrt, Median, Effective Samples, Clipped)
    - Saves reports/train_class_frequencies.csv and reports/train_cooccurrence_matrix.csv
"""

import os
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

DACL10K_CLASSES = [
    "ACrack", "Bearing", "Cavity", "Crack", "Drainage", "EJoint",
    "Efflorescence", "ExposedRebars", "Graffiti", "Hollowareas", "JTape",
    "PEquipment", "Restformwork", "Rockpocket", "Rust", "Spalling",
    "WConccor", "Weathering", "Wetspot"
]
NUM_CLASSES = len(DACL10K_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(DACL10K_CLASSES)}


def polygon_shoelace_area(pts):
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def main():
    repo_root = Path(__file__).resolve().parent.parent
    train_ann_dir = Path("/Users/kundankumar/Downloads/dacl10k_v2_devphase/annotations/train")
    if not train_ann_dir.exists():
        raise FileNotFoundError(f"Training annotations directory not found: {train_ann_dir}")

    ann_files = sorted([f for f in os.listdir(train_ann_dir) if f.endswith(".json")])
    total_train_images = len(ann_files)
    print(f"Total training annotation files found: {total_train_images}")
    assert total_train_images == 6935, f"Expected exactly 6935 training images, found {total_train_images}"

    target_h, target_w = 512, 512
    total_training_pixels_512 = total_train_images * target_h * target_w  # 1,817,968,640

    # Data structures
    images_per_class = defaultdict(int)
    shapes_per_class = defaultdict(int)
    poly_areas_native = defaultdict(list)
    poly_areas_512 = defaultdict(list)
    mask_pixels_512 = defaultdict(int)
    mask_pixels_native = defaultdict(int)

    # Size buckets (in 512x512 pixels): Small (<1024), Medium (1024-9216), Large (>9216)
    size_buckets_512 = {c: {"small": 0, "medium": 0, "large": 0} for c in DACL10K_CLASSES}
    size_buckets_native = {c: {"small": 0, "medium": 0, "large": 0} for c in DACL10K_CLASSES}

    # Co-occurrence: 19x19
    cooccurrence = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    t0 = time.time()
    print("Beginning comprehensive analysis of all 6,935 training images...")

    for idx, fname in enumerate(ann_files):
        fpath = train_ann_dir / fname
        with open(fpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        orig_h = data.get("imageHeight")
        orig_w = data.get("imageWidth")
        shapes = data.get("shapes", [])

        # Letterbox transformation parameters
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        present_classes_in_image = set()
        shapes_by_class_512 = defaultdict(list)
        shapes_by_class_native = defaultdict(list)

        for s in shapes:
            lbl = s.get("label")
            if lbl not in CLASS_TO_IDX:
                continue

            pts = np.array(s.get("points", []), dtype=np.float32)
            if len(pts) < 3:
                continue

            present_classes_in_image.add(lbl)
            shapes_per_class[lbl] += 1

            # Native polygon area
            area_nat = polygon_shoelace_area(pts)
            poly_areas_native[lbl].append(area_nat)
            if area_nat < 1024:
                size_buckets_native[lbl]["small"] += 1
            elif area_nat <= 9216:
                size_buckets_native[lbl]["medium"] += 1
            else:
                size_buckets_native[lbl]["large"] += 1

            # 512x512 polygon points
            pts_512 = pts.copy()
            pts_512[:, 0] = pts_512[:, 0] * scale + pad_left
            pts_512[:, 1] = pts_512[:, 1] * scale + pad_top
            area_512 = polygon_shoelace_area(pts_512)
            poly_areas_512[lbl].append(area_512)

            if area_512 < 1024:
                size_buckets_512[lbl]["small"] += 1
            elif area_512 <= 9216:
                size_buckets_512[lbl]["medium"] += 1
            else:
                size_buckets_512[lbl]["large"] += 1

            shapes_by_class_512[lbl].append(np.int32([np.round(pts_512)]))
            shapes_by_class_native[lbl].append(np.int32([np.round(pts)]))

        # Update image counts & co-occurrence
        present_indices = [CLASS_TO_IDX[c] for c in present_classes_in_image]
        for c in present_classes_in_image:
            images_per_class[c] += 1

        for i1 in present_indices:
            for i2 in present_indices:
                cooccurrence[i1, i2] += 1

        # Rasterize masks to get true unioned pixel counts (handling polygon overlaps)
        for c, poly_list_512 in shapes_by_class_512.items():
            mask_512 = np.zeros((target_h, target_w), dtype=np.uint8)
            cv2.fillPoly(mask_512, poly_list_512, color=1)
            mask_pixels_512[c] += int(mask_512.sum())

        for c, poly_list_nat in shapes_by_class_native.items():
            mask_nat = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.fillPoly(mask_nat, poly_list_nat, color=1)
            mask_pixels_native[c] += int(mask_nat.sum())

        if (idx + 1) % 1000 == 0 or (idx + 1) == total_train_images:
            elapsed = time.time() - t0
            print(f"Processed {idx + 1}/{total_train_images} images ({elapsed:.1f}s)...")

    total_time = time.time() - t0
    print(f"Analysis completed in {total_time:.2f}s!")

    # Assemble comprehensive per-class statistics DataFrame
    rows = []
    for c in DACL10K_CLASSES:
        img_count = images_per_class[c]
        img_pct = (img_count / total_train_images) * 100.0
        n_shapes = shapes_per_class[c]
        shapes_per_pos_img = (n_shapes / img_count) if img_count > 0 else 0.0

        pix_512 = mask_pixels_512[c]
        pix_512_pct = (pix_512 / total_training_pixels_512) * 100.0
        pix_native = mask_pixels_native[c]

        # Area percentiles in 512x512
        arr_512 = np.array(poly_areas_512[c]) if poly_areas_512[c] else np.array([0.0])
        mean_area_512 = float(np.mean(arr_512))
        std_area_512 = float(np.std(arr_512))
        p10_512 = float(np.percentile(arr_512, 10))
        p25_512 = float(np.percentile(arr_512, 25))
        p50_512 = float(np.percentile(arr_512, 50))  # Median
        p75_512 = float(np.percentile(arr_512, 75))
        p90_512 = float(np.percentile(arr_512, 90))
        p95_512 = float(np.percentile(arr_512, 95))
        p99_512 = float(np.percentile(arr_512, 99))
        max_area_512 = float(np.max(arr_512))

        # Size category percentages
        n_small = size_buckets_512[c]["small"]
        n_med = size_buckets_512[c]["medium"]
        n_large = size_buckets_512[c]["large"]
        tot_b = n_small + n_med + n_large if (n_small + n_med + n_large) > 0 else 1
        pct_small = (n_small / tot_b) * 100.0
        pct_med = (n_med / tot_b) * 100.0
        pct_large = (n_large / tot_b) * 100.0

        # Area percentiles in native resolution
        arr_nat = np.array(poly_areas_native[c]) if poly_areas_native[c] else np.array([0.0])
        mean_area_nat = float(np.mean(arr_nat))
        p50_nat = float(np.percentile(arr_nat, 50))

        rows.append({
            "class_index": CLASS_TO_IDX[c],
            "class_name": c,
            "image_count": img_count,
            "image_percentage": round(img_pct, 4),
            "total_shapes": n_shapes,
            "shapes_per_pos_img": round(shapes_per_pos_img, 2),
            "total_pixels_512": pix_512,
            "pixel_percentage_512": round(pix_512_pct, 6),
            "total_pixels_native": pix_native,
            "poly_mean_area_512": round(mean_area_512, 1),
            "poly_median_area_512": round(p50_512, 1),
            "poly_p10_512": round(p10_512, 1),
            "poly_p25_512": round(p25_512, 1),
            "poly_p75_512": round(p75_512, 1),
            "poly_p90_512": round(p90_512, 1),
            "poly_p95_512": round(p95_512, 1),
            "poly_p99_512": round(p99_512, 1),
            "poly_max_area_512": round(max_area_512, 1),
            "small_defects_count": n_small,
            "small_defects_pct": round(pct_small, 2),
            "medium_defects_count": n_med,
            "medium_defects_pct": round(pct_med, 2),
            "large_defects_count": n_large,
            "large_defects_pct": round(pct_large, 2),
            "poly_mean_area_native": round(mean_area_nat, 1),
            "poly_median_area_native": round(p50_nat, 1),
        })

    df = pd.DataFrame(rows)

    # Candidate Class Weights Calculation (TRAINING DATA ONLY)
    # 1. Inverse Image Frequency: N_total / (N_classes * N_c)
    # Normalized so mean weight = 1.0
    N_tot = total_train_images
    inv_img = [N_tot / (NUM_CLASSES * max(1, r["image_count"])) for r in rows]
    inv_img_norm = [w / np.mean(inv_img) for w in inv_img]

    # 2. Inverse Pixel Frequency (512x512): Total_fg_pixels / (NUM_CLASSES * pix_c)
    total_fg_pixels = sum(r["total_pixels_512"] for r in rows)
    inv_pix = [total_fg_pixels / (NUM_CLASSES * max(1, r["total_pixels_512"])) for r in rows]
    inv_pix_norm = [w / np.mean(inv_pix) for w in inv_pix]

    # 3. Square-Root Inverse Pixel Frequency: sqrt(total_fg_pixels / pix_c), normalized
    sqrt_inv_pix = [np.sqrt(total_fg_pixels / max(1, r["total_pixels_512"])) for r in rows]
    sqrt_inv_pix_norm = [w / np.mean(sqrt_inv_pix) for w in sqrt_inv_pix]

    # 4. Median Frequency Balancing: median(pix_512) / pix_c
    med_pix = np.median([r["total_pixels_512"] for r in rows])
    med_freq = [med_pix / max(1, r["total_pixels_512"]) for r in rows]
    med_freq_norm = [w / np.mean(med_freq) for w in med_freq]

    # 5. Effective Number of Samples (Cui et al. CVPR 2019):
    # E_n = (1 - beta^n) / (1 - beta)
    # Weight w = (1 - beta) / (1 - beta^n)
    beta = 0.999
    eff_weights = []
    for r in rows:
        n = max(1, r["image_count"])
        eff_w = (1.0 - beta) / (1.0 - (beta ** n))
        eff_weights.append(eff_w)
    eff_weights_norm = [w / np.mean(eff_weights) for w in eff_weights]

    # 6. Clipped Weights (Recommended Practical: Sqrt Inverse Pixel Frequency clipped to [0.2, 5.0])
    clipped_sqrt_weights = np.clip(sqrt_inv_pix_norm, 0.2, 5.0)
    clipped_sqrt_weights = [round(float(w), 4) for w in (clipped_sqrt_weights / np.mean(clipped_sqrt_weights))]

    # Add candidate weights to DataFrame
    df["weight_inv_image_norm"] = [round(w, 4) for w in inv_img_norm]
    df["weight_inv_pixel_raw"] = [round(w, 4) for w in inv_pix]
    df["weight_inv_pixel_norm"] = [round(w, 4) for w in inv_pix_norm]
    df["weight_sqrt_pixel_norm"] = [round(w, 4) for w in sqrt_inv_pix_norm]
    df["weight_median_freq"] = [round(w, 4) for w in med_freq]
    df["weight_effective_samples_beta999"] = [round(w, 4) for w in eff_weights_norm]
    df["weight_clipped_sqrt_recommended"] = clipped_sqrt_weights

    # Save to reports directory
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "train_class_frequencies.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved class frequencies CSV to: {csv_path}")

    # Save Co-occurrence Matrix
    cooc_df = pd.DataFrame(cooccurrence, index=DACL10K_CLASSES, columns=DACL10K_CLASSES)
    cooc_csv_path = reports_dir / "train_cooccurrence_matrix.csv"
    cooc_df.to_csv(cooc_csv_path)
    print(f"Saved co-occurrence matrix to: {cooc_csv_path}")

    # Compute Conditional Probability Matrix: P(Class_col | Class_row)
    cond_prob = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for i in range(NUM_CLASSES):
        diag = cooccurrence[i, i]  # total images with class i
        if diag > 0:
            cond_prob[i, :] = cooccurrence[i, :] / diag
    cond_df = pd.DataFrame(cond_prob, index=DACL10K_CLASSES, columns=DACL10K_CLASSES)
    cond_csv_path = reports_dir / "train_conditional_probabilities.csv"
    cond_df.to_csv(cond_csv_path)
    print(f"Saved conditional probability matrix to: {cond_csv_path}")

    # Display Top Summary
    print("\n" + "=" * 90)
    print("DACL10K-v2 TRAINING SPLIT CLASS IMBALANCE SUMMARY (6,935 IMAGES)")
    print("=" * 90)
    display_cols = [
        "class_name", "image_count", "image_percentage", "total_shapes",
        "total_pixels_512", "pixel_percentage_512", "poly_median_area_512",
        "small_defects_pct", "large_defects_pct", "weight_clipped_sqrt_recommended"
    ]
    print(df[display_cols].sort_values(by="image_count", ascending=False).to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
