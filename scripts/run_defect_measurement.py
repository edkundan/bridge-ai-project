#!/usr/bin/env python3
"""
Script: scripts/run_defect_measurement.py
Description:
    Runs Stage 2 Defect Measurement and Severity Assessment Foundation.
    Takes an input image (or folder of images), applies trained DeepLabV3+
    segmentation checkpoint (or precomputed probabilities), extracts geometric
    measurements, generates overlays, and outputs structured JSON.

Usage:
    python scripts/run_defect_measurement.py \
        --image data/raw/dacl10k_v2_devphase/images/validation/dacl10k_v2_validation_0001.jpg \
        --checkpoint experiments/segmentation_baseline/checkpoints/best_model.pth \
        --output-dir outputs/stage2_measurements \
        --threshold 0.5 \
        --min-area 10
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import torch

from src.data.dataset import DACL10K_CLASSES
from src.measurement import (
    DefectMeasurementPipeline,
    DefectVisualizer,
    LiteratureGroundedSeverityEvaluator,
)
from src.segmentation.predict import BridgeDefectPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_defect_measurement")


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 2 Defect Measurement and Severity Assessment Foundation."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single input image file.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to directory of images to process in batch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pth). If None, runs in simulation/dry-run mode.",
    )
    parser.add_argument(
        "--probabilities",
        type=str,
        default=None,
        help="Optional path to precomputed probability map (.npy) of shape (19, H, W).",
    )
    parser.add_argument(
        "--roi-mask",
        type=str,
        default=None,
        help="Optional path to binary ROI mask image (.png/.jpg) of shape (H, W).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stage2_measurements",
        help="Directory to save measurement JSON and overlay visualizations.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold in [0.0, 1.0] for defect detection (default: 0.5).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Minimum connected component pixel area to filter noise (default: 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: 'auto', 'cuda', 'mps', or 'cpu'.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Measurement Pipeline & Visualizer
    pipeline = DefectMeasurementPipeline(
        threshold=args.threshold,
        min_component_area_pixels=args.min_area,
    )
    visualizer = DefectVisualizer()
    severity_evaluator = LiteratureGroundedSeverityEvaluator()

    # Determine image list
    image_paths: List[Path] = []
    if args.image:
        image_paths.append(Path(args.image))
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            image_paths.extend(image_dir.glob(ext))
        image_paths = sorted(image_paths)
    elif args.probabilities:
        # Probability only mode
        pass
    else:
        logger.info("No --image or --image-dir specified. Generating synthetic demonstration.")
        # Create a synthetic demonstration image
        synth_img = np.ones((512, 512, 3), dtype=np.uint8) * 180
        # Draw simulated concrete texture
        cv2.putText(synth_img, "CONCRETE SURFACE", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        synth_path = out_dir / "synthetic_demo_image.jpg"
        cv2.imwrite(str(synth_path), cv2.cvtColor(synth_img, cv2.COLOR_RGB2BGR))
        image_paths.append(synth_path)

    # Initialize Predictor if checkpoint is provided
    predictor = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        predictor = BridgeDefectPredictor(
            checkpoint_path=args.checkpoint,
            device=args.device,
            threshold=args.threshold,
        )
    elif args.checkpoint:
        logger.warning(f"Checkpoint not found at: {args.checkpoint}. Running in mock mode.")

    # Optional ROI mask
    roi_mask = None
    if args.roi_mask and os.path.exists(args.roi_mask):
        roi_raw = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
        if roi_raw is not None:
            roi_mask = (roi_raw > 127).astype(np.uint8)
            logger.info(f"Loaded ROI mask with {np.sum(roi_mask > 0)} pixels.")

    # Process images
    total_processed = 0
    for img_path in image_paths:
        logger.info(f"Processing image: {img_path.name}")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.error(f"Failed to read image: {img_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        if predictor is not None:
            pred_res = predictor.predict_image(rgb, return_orig_size=True)
            prob_maps = pred_res["probability_maps"]
        elif args.probabilities and os.path.exists(args.probabilities):
            prob_maps = np.load(args.probabilities)
        else:
            # Deterministic synthetic probability map for demonstration
            prob_maps = np.zeros((19, h, w), dtype=np.float32)
            # Simulate a crack
            for i in range(100, 300):
                x = int(150 + 20 * np.sin(i / 20.0))
                prob_maps[3, i, max(0, x-1):min(w, x+2)] = 0.88  # Crack (idx 3)
            # Simulate a spall
            cv2.circle(prob_maps[15], (350, 250), 35, 0.92, -1)  # Spalling (idx 15)
            # Simulate exposed rebar inside the spall
            prob_maps[7, 235:265, 345:355] = 0.95  # ExposedRebars (idx 7)

        # Run measurement pipeline
        measurement = pipeline.measure_probabilities(
            probabilities=prob_maps,
            image_id=img_path.stem,
            roi_mask=roi_mask,
        )

        # Run severity assessment
        severity = severity_evaluator.evaluate(measurement)

        # Export JSON outputs
        json_path = out_dir / f"{img_path.stem}_measurement.json"
        with open(json_path, "w") as f:
            f.write(measurement.to_json(indent=2))

        severity_json_path = out_dir / f"{img_path.stem}_severity.json"
        with open(severity_json_path, "w") as f:
            json.dump(severity.to_dict(), f, indent=2)

        # Render and save visual overlay
        overlay_path = out_dir / f"{img_path.stem}_overlay.png"
        visualizer.save_overlay(
            save_path=overlay_path,
            image_rgb=rgb,
            measurement=measurement,
            probability_maps=prob_maps,
            roi_mask=roi_mask,
        )

        logger.info(
            f"Image {img_path.name}: Detected {measurement.total_detected_classes} classes "
            f"({measurement.get_detected_classes()}) | Total defect area: {measurement.total_defect_image_area_pct:.2f}% | "
            f"Highest Criticality Tier: {severity.highest_criticality_tier}"
        )
        logger.info(f"Saved JSON: {json_path}")
        logger.info(f"Saved Overlay: {overlay_path}")
        total_processed += 1

    logger.info(f"Completed processing {total_processed} images. Results saved in {out_dir}")


if __name__ == "__main__":
    main()
