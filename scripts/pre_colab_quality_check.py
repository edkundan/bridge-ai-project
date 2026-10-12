"""
Script: scripts/pre_colab_quality_check.py
Description:
    Pre-Colab Quality Checks (Task A):
    1. End-to-end class-to-channel mapping consistency.
    2. Lightweight full-dataset annotation audit (100% of train and validation sets, fast vector math).
    3. Optimized mask degradation and disappearance analysis (comparing original resolution vs 512x512 letterbox).
       - FAST MODE (default): Configurable representative subset (e.g., 100 train, 50 val).
       - FULL AUDIT MODE: Exhaustive audit across all 6,935 train and 975 val images.
    4. Caches expensive mask checks with configuration validation.
    5. Clean progress bars with tqdm.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Ensure project root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.mask_converter import MaskConverter, DACL10K_CLASSES
from src.data.dataset import DACL10KDataset, letterbox_resize
from src.segmentation.evaluate import SegmentationEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("quality_check")

DATASET_ROOT = Path("/Users/kundankumar/Downloads/dacl10k_v2_devphase")
CACHE_FILE = repo_root / "reports" / ".cache_quality_check.json"


def check_mapping_consistency() -> Dict[str, Any]:
    """Verify class-to-channel mapping across configuration, converter, dataset, and evaluator."""
    logger.info("=== CHECK 1: CLASS-TO-CHANNEL MAPPING CONSISTENCY ===")
    
    with open(repo_root / "configs/training.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg_classes = cfg["dataset"]["classes"]
    
    converter = MaskConverter()
    mc_classes = converter.class_names
    
    from src.data.dataset import DACL10K_CLASSES as DS_CLASSES
    evaluator = SegmentationEvaluator()
    eval_classes = evaluator.class_names
    
    assert len(cfg_classes) == 19, f"Config has {len(cfg_classes)} classes, expected 19"
    assert cfg_classes == list(DACL10K_CLASSES), "Config classes do not match DACL10K_CLASSES"
    assert mc_classes == list(DACL10K_CLASSES), "MaskConverter classes do not match DACL10K_CLASSES"
    assert list(DS_CLASSES) == list(DACL10K_CLASSES), "Dataset classes do not match DACL10K_CLASSES"
    assert eval_classes == list(DACL10K_CLASSES), "Evaluator classes do not match DACL10K_CLASSES"
    
    mapping = {idx: name for idx, name in enumerate(DACL10K_CLASSES)}
    logger.info("  All 19 channels verified identical across pipeline in canonical alphabetical order.")
    return mapping


def polygon_shoelace_area(points: np.ndarray) -> float:
    """Calculate polygon area using Green's theorem / Shoelace formula without rasterization."""
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def audit_annotations_lightweight(split: str) -> Dict[str, Any]:
    """
    Lightweight, fast audit across 100% of the dataset split without image rasterization.
    Parses JSON files, validates coordinates, counts polygons, and calculates Shoelace areas.
    """
    anno_dir = DATASET_ROOT / "annotations" / split
    anno_files = sorted(list(anno_dir.glob("*.json")))
    
    class_stats = {
        cls_name: {
            "polygon_count": 0,
            "images_present": 0,
            "total_area_shoelace": 0.0,
            "min_area": float("inf"),
            "max_area": 0.0,
        }
        for cls_name in DACL10K_CLASSES
    }
    
    empty_images = 0
    total_images = len(anno_files)
    total_polygons = 0
    
    desc = f"Lightweight Scan [{split.upper()} - 100% full split]"
    for json_path in tqdm(anno_files, desc=desc, unit="file"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        shapes = data.get("shapes", [])
        if not shapes:
            empty_images += 1
            continue
            
        seen_classes_in_img = set()
        for shape in shapes:
            label = shape.get("label")
            if label not in class_stats:
                continue
                
            pts = np.array(shape.get("points", []), dtype=np.float32)
            if len(pts) < 3:
                continue
                
            area = polygon_shoelace_area(pts)
            class_stats[label]["polygon_count"] += 1
            class_stats[label]["total_area_shoelace"] += area
            class_stats[label]["min_area"] = min(class_stats[label]["min_area"], area)
            class_stats[label]["max_area"] = max(class_stats[label]["max_area"], area)
            seen_classes_in_img.add(label)
            total_polygons += 1
            
        for label in seen_classes_in_img:
            class_stats[label]["images_present"] += 1
            
    # Normalize min_area
    for cls_name, s in class_stats.items():
        if s["min_area"] == float("inf"):
            s["min_area"] = 0.0
            
    return {
        "split": split,
        "total_files": total_images,
        "empty_annotation_files": empty_images,
        "annotated_files": total_images - empty_images,
        "total_polygons": total_polygons,
        "class_stats": class_stats,
    }


def audit_mask_degradation_optimized(
    split: str,
    max_samples: Optional[int] = None,
    target_size: Tuple[int, int] = (512, 512)
) -> Dict[str, Any]:
    """
    Optimized mask degradation check:
    - Only rasterizes channels for classes actually present in each sample.
    - Uses 2D single-channel rasterization instead of 19-channel 4K allocations.
    - Letterbox resizes with cv2.INTER_NEAREST to strictly test pixel retention.
    """
    anno_dir = DATASET_ROOT / "annotations" / split
    anno_files = sorted(list(anno_dir.glob("*.json")))
    
    if max_samples is not None and max_samples < len(anno_files):
        # Reproducible deterministic subset
        np.random.seed(42)
        sample_indices = np.linspace(0, len(anno_files) - 1, max_samples, dtype=int)
        anno_files = [anno_files[i] for i in sample_indices]
        is_subset = True
    else:
        is_subset = False
        
    converter = MaskConverter()
    target_h, target_w = target_size
    
    stats = {
        cls_name: {
            "orig_count": 0,           # Images where class mask had > 0 pixels at native resolution
            "resized_count": 0,        # Images where class mask still has > 0 pixels at 512x512
            "disappeared_count": 0,    # Images where class mask completely disappeared (0 pixels)
            "orig_pixels_total": 0,
            "resized_pixels_total": 0,
            "min_orig_pixels": float("inf"),
            "min_retained_pixels": float("inf"),
        }
        for cls_name in DACL10K_CLASSES
    }
    
    images_audited = 0
    images_with_disappearance = 0
    
    mode_label = f"Subsample ({len(anno_files)} imgs)" if is_subset else "Full Exhaustive"
    desc = f"Mask Degradation [{split.upper()} - {mode_label}]"
    
    t0 = time.time()
    for json_path in tqdm(anno_files, desc=desc, unit="img"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        orig_h = data.get("imageHeight", 0)
        orig_w = data.get("imageWidth", 0)
        shapes = data.get("shapes", [])
        
        if not shapes or orig_h <= 0 or orig_w <= 0:
            continue
            
        images_audited += 1
        
        # Group shapes by class
        shapes_by_class: Dict[str, List[np.ndarray]] = {}
        for s in shapes:
            lbl = s.get("label")
            if lbl not in converter.class_to_idx:
                continue
            pts = np.array(s.get("points", []), dtype=np.float32)
            if len(pts) >= 3:
                shapes_by_class.setdefault(lbl, []).append(pts)
                
        if not shapes_by_class:
            continue
            
        # Compute letterbox scaling geometry
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        
        img_had_disappearance = False
        
        # Test ONLY active classes in this image
        for cls_name, poly_list in shapes_by_class.items():
            # 1. Rasterize single 2D channel at original resolution
            orig_chan = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for pts in poly_list:
                pts_clipped = np.copy(pts)
                pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], 0, orig_w - 1)
                pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], 0, orig_h - 1)
                pts_int = np.int32([np.round(pts_clipped)])
                cv2.fillPoly(orig_chan, pts_int, color=1)
                
            orig_px = int(np.sum(orig_chan > 0))
            if orig_px == 0:
                continue
                
            # 2. Letterbox resize single 2D channel with INTER_NEAREST
            resized_chan_small = cv2.resize(orig_chan, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            padded_chan = cv2.copyMakeBorder(
                resized_chan_small, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
            resized_px = int(np.sum(padded_chan > 0))
            
            s = stats[cls_name]
            s["orig_count"] += 1
            s["orig_pixels_total"] += orig_px
            s["min_orig_pixels"] = min(s["min_orig_pixels"], orig_px)
            
            if resized_px > 0:
                s["resized_count"] += 1
                s["resized_pixels_total"] += resized_px
                s["min_retained_pixels"] = min(s["min_retained_pixels"], resized_px)
            else:
                s["disappeared_count"] += 1
                img_had_disappearance = True
                
        if img_had_disappearance:
            images_with_disappearance += 1
            
    # Clean up infinities
    for cls_name, s in stats.items():
        if s["min_orig_pixels"] == float("inf"):
            s["min_orig_pixels"] = 0
        if s["min_retained_pixels"] == float("inf"):
            s["min_retained_pixels"] = 0
            
    elapsed = time.time() - t0
    return {
        "split": split,
        "is_subset": is_subset,
        "samples_audited": images_audited,
        "total_files_in_split": len(anno_files),
        "images_with_disappearance": images_with_disappearance,
        "elapsed_seconds": round(elapsed, 2),
        "class_stats": stats,
    }


def get_cache_key(mode: str, train_samples: int, val_samples: int) -> str:
    key_str = f"{DATASET_ROOT}_{mode}_{train_samples}_{val_samples}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def load_cached_results(cache_key: str) -> Optional[Dict[str, Any]]:
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("cache_key") == cache_key:
            logger.info(f"Loaded cached quality check results from {CACHE_FILE}")
            return cache.get("data")
    except Exception as e:
        logger.warning(f"Could not load cache: {e}")
    return None


def save_cached_results(cache_key: str, data: Dict[str, Any]):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"cache_key": cache_key, "timestamp": time.time(), "data": data}, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save cache: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pre-Colab Quality Checks for DACL10K-v2")
    parser.add_argument("--fast", action="store_true", default=True, help="Fast mode with sample subsets (default)")
    parser.add_argument("--full", action="store_true", help="Full audit mode across all images")
    parser.add_argument("--train-samples", type=int, default=100, help="Train samples in fast mode (default: 100)")
    parser.add_argument("--val-samples", type=int, default=50, help="Val samples in fast mode (default: 50)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache and force re-execution")
    args = parser.parse_args()
    
    if args.full:
        mode = "full"
        train_samples = None
        val_samples = None
    else:
        mode = "fast"
        train_samples = args.train_samples
        val_samples = args.val_samples
        
    logger.info("======================================================================")
    logger.info(f"STARTING PRE-COLAB QUALITY AUDIT (MODE: {mode.upper()})")
    logger.info(f"Dataset root: {DATASET_ROOT}")
    if mode == "fast":
        logger.info(f"Mask check sample sizes: Train = {train_samples}, Val = {val_samples}")
    else:
        logger.info("Mask check sample sizes: 100% of Train and Val images (Exhaustive)")
    logger.info("======================================================================")
    
    start_total_time = time.time()
    
    # 1. Mapping consistency
    mapping = check_mapping_consistency()
    
    # Check cache for expensive mask degradation audit
    cache_key = get_cache_key(mode, train_samples or 6935, val_samples or 975)
    cached_data = None if args.no_cache else load_cached_results(cache_key)
    
    if cached_data is not None:
        logger.info("Using valid cached results for mask degradation checks.")
        val_mask_audit = cached_data["val_mask_audit"]
        train_mask_audit = cached_data["train_mask_audit"]
        val_anno_audit = cached_data["val_anno_audit"]
        train_anno_audit = cached_data["train_anno_audit"]
    else:
        # 2. Lightweight annotation audit (100% full dataset coverage, fast)
        logger.info("\n=== CHECK 2: LIGHTWEIGHT ANNOTATION SCAN (100% COMPLETE DATASET) ===")
        val_anno_audit = audit_annotations_lightweight("validation")
        train_anno_audit = audit_annotations_lightweight("train")
        
        # 3. Optimized mask degradation audit (Fast sample or Full)
        logger.info(f"\n=== CHECK 3: MASK DEGRADATION AUDIT ({mode.upper()} MODE) ===")
        val_mask_audit = audit_mask_degradation_optimized("validation", max_samples=val_samples)
        train_mask_audit = audit_mask_degradation_optimized("train", max_samples=train_samples)
        
        save_cached_results(cache_key, {
            "val_anno_audit": val_anno_audit,
            "train_anno_audit": train_anno_audit,
            "val_mask_audit": val_mask_audit,
            "train_mask_audit": train_mask_audit,
        })
        
    total_time = time.time() - start_total_time
    
    # Print clean summary report
    print("\n" + "=" * 80)
    print("                      PRE-COLAB QUALITY AUDIT REPORT")
    print("=" * 80)
    print(f"Execution Mode:       {mode.upper()}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Class Mapping:        19/19 channels verified identical across pipeline")
    print("-" * 80)
    print("1. DATASET ANNOTATION COVERAGE (100% EXHAUSTIVE SCAN):")
    print(f"   - Training Split:   {train_anno_audit['annotated_files']} annotated / {train_anno_audit['total_files']} total images ({train_anno_audit['empty_annotation_files']} empty)")
    print(f"   - Validation Split: {val_anno_audit['annotated_files']} annotated / {val_anno_audit['total_files']} total images ({val_anno_audit['empty_annotation_files']} empty)")
    print(f"   - Total Polygons:   {train_anno_audit['total_polygons']} train, {val_anno_audit['total_polygons']} val")
    print("-" * 80)
    print(f"2. MASK DISAPPEARANCE & DEGRADATION AUDIT (512x512 RESIZE, {mode.upper()}):")
    print(f"   - Validation Split: {val_mask_audit['samples_audited']} images audited.")
    print(f"     Images with >=1 disappearing class: {val_mask_audit['images_with_disappearance']}")
    print(f"   - Training Split:   {train_mask_audit['samples_audited']} images audited.")
    print(f"     Images with >=1 disappearing class: {train_mask_audit['images_with_disappearance']}")
    print("-" * 80)
    
    # Detailed breakdown for difficult thin/small defects
    thin_classes = ["Crack", "ACrack", "ExposedRebars", "Cavity", "EJoint", "WConccor"]
    print(f"{'Class Name':<15} | {'Val Orig':<9} | {'Val Retained':<12} | {'Val Lost':<8} | {'Train Orig':<10} | {'Train Retained':<14} | {'Train Lost':<10}")
    print("-" * 80)
    for cls_name in DACL10K_CLASSES:
        v_s = val_mask_audit["class_stats"][cls_name]
        t_s = train_mask_audit["class_stats"][cls_name]
        is_highlight = cls_name in thin_classes
        marker = "*" if is_highlight else " "
        print(f"{cls_name+marker:<15} | {v_s['orig_count']:<9} | {v_s['resized_count']:<12} | {v_s['disappeared_count']:<8} | {t_s['orig_count']:<10} | {t_s['resized_count']:<14} | {t_s['disappeared_count']:<10}")
        
    print("-" * 80)
    print("* Denotes critical thin/small defect classes specifically analyzed.")
    print("=" * 80)
    
    # Save permanent summary JSON
    summary_path = repo_root / "reports" / "pre_colab_quality_report.json"
    summary_data = {
        "mode": mode,
        "total_execution_time_seconds": round(total_time, 2),
        "mapping_verified": True,
        "annotation_audit_full": {
            "train": train_anno_audit,
            "validation": val_anno_audit,
        },
        "mask_degradation_audit": {
            "train": train_mask_audit,
            "validation": val_mask_audit,
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Detailed quality report saved to: {summary_path}")


if __name__ == "__main__":
    main()
