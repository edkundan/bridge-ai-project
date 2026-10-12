import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.path import Path as MplPath
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CLASSES = [
    "ACrack", "Bearing", "Cavity", "Crack", "Drainage", "EJoint",
    "Efflorescence", "ExposedRebars", "Graffiti", "Hollowareas", "JTape",
    "PEquipment", "Restformwork", "Rockpocket", "Rust", "Spalling",
    "WConccor", "Weathering", "Wetspot"
]

# Generate a visually distinct color palette for the 19 classes
# Using a fixed seed to keep colors consistent across runs
np.random.seed(42)
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))[:19] # tab20 has 20 distinct colors
CLASS_COLORS = {cls: COLORS[i] for i, cls in enumerate(CLASSES)}

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DACL10K-v2 dataset annotations")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml",
                        help="Path to dataset config file")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Override dataset root directory")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to visualize (e.g., train, val, test)")
    parser.add_argument("--output-dir", type=str, default="outputs/dataset_visualizations",
                        help="Directory to save visualizations")
    return parser.parse_args()

def calculate_centroid(points: List[List[float]]) -> Tuple[float, float]:
    """Calculate the centroid of a polygon given its points."""
    arr = np.array(points)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))

def visualize_sample(image_path: Path, annotation_path: Path, output_path: Path):
    """Visualize a single image and its annotations, then save it."""
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return
    if not annotation_path.exists():
        logger.warning(f"Annotation not found: {annotation_path}")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return

    try:
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load annotation {annotation_path}: {e}")
        return

    shapes = annotation.get("shapes", [])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    present_classes = set()

    for shape in shapes:
        label = shape.get("label")
        points = shape.get("points")
        shape_type = shape.get("shape_type", "polygon")

        if label not in CLASS_COLORS:
            continue
            
        if shape_type != "polygon" or not points or len(points) < 3:
            continue

        present_classes.add(label)
        color = CLASS_COLORS[label]

        # Draw filled semi-transparent polygon
        poly = mpatches.Polygon(points, closed=True, facecolor=color, alpha=0.4, edgecolor=color, linewidth=2)
        ax.add_patch(poly)

        # Calculate centroid and add text
        cx, cy = calculate_centroid(points)
        ax.text(cx, cy, label, color="white", fontsize=8, weight="bold",
                ha="center", va="center", bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1))

    # Add legend
    if present_classes:
        legend_patches = [mpatches.Patch(color=CLASS_COLORS[cls], label=cls) for cls in sorted(present_classes)]
        ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.15, 1), title="Classes")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved visualization to {output_path}")

def main():
    args = parse_args()
    random.seed(42)

    dataset_root = args.dataset_root
    if not dataset_root and os.path.exists(args.config):
        try:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            dataset_root = config.get("dataset", {}).get("root_dir", None)
        except Exception as e:
            logger.warning(f"Failed to read dataset root from config {args.config}: {e}")
    
    if not dataset_root:
        logger.error("Dataset root must be provided either via --dataset-root or in the config file.")
        return

    root_path = Path(dataset_root)
    split = args.split
    images_dir = root_path / "images" / split
    annotations_dir = root_path / "annotations" / split

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return
    if not annotations_dir.exists():
        logger.error(f"Annotations directory not found: {annotations_dir}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all annotations
    annotation_files = list(annotations_dir.glob("*.json"))
    if not annotation_files:
        logger.error(f"No annotation files found in {annotations_dir}")
        return

    # Randomly select N samples
    num_samples = min(args.samples, len(annotation_files))
    selected_annotations = random.sample(annotation_files, num_samples)

    logger.info(f"Selected {num_samples} samples from {split} split for visualization.")

    for ann_path in selected_annotations:
        # Assuming image filename matches annotation filename but with .jpg extension
        image_name = ann_path.stem + ".jpg"
        image_path = images_dir / image_name
        output_path = output_dir / f"{ann_path.stem}_vis.png"
        
        visualize_sample(image_path, ann_path, output_path)

if __name__ == "__main__":
    main()
