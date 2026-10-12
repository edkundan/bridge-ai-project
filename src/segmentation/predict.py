"""
Module: src.segmentation.predict
Description:
    Inference pipeline for DACL10K-v2 bridge defect segmentation.
    Produces 19 independent probability maps per image and thresholded binary masks.
    
    Features:
    - Preprocesses input image using aspect-ratio preserving letterbox resizing.
    - Runs model forward pass with sigmoid activation.
    - Inverts letterbox transformation back to original image resolution.
    - Saves 19-channel probability maps (.npy / .npz).
    - Generates colored overlay visualizations with class legends.
"""

import argparse
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
import torch
import yaml

from src.data.dataset import DACL10K_CLASSES, IMAGENET_MEAN, IMAGENET_STD, letterbox_resize
from src.segmentation.model import BridgeSegmentationModel

logger = logging.getLogger(__name__)

# Fixed visually distinct colormap for qualitative prediction overlays
np.random.seed(42)
OVERLAY_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))[:19]


class BridgeDefectPredictor:
    """
    Inference engine that generates 19 independent class probability maps.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        device: str = "auto",
        target_size: Tuple[int, int] = (512, 512),
        threshold: float = 0.5,
        architecture: str = "DeepLabV3Plus",
        encoder_name: str = "resnet34",
    ):
        """
        Args:
            checkpoint_path: Path to trained weights (.pth).
            model: Optional pre-instantiated model.
            device: 'auto', 'mps', 'cuda', or 'cpu'.
            target_size: Model input resolution (H, W).
            threshold: Probability threshold for binary masks.
            architecture: Model architecture name.
            encoder_name: Backbone encoder name.
        """
        self.target_size = target_size
        self.threshold = threshold
        self.class_names = list(DACL10K_CLASSES)
        self.num_classes = len(self.class_names)

        # Device selection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Predictor initialized on device: {self.device}")

        # Initialize model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = BridgeSegmentationModel(
                architecture=architecture,
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                num_classes=self.num_classes,
            ).to(self.device)

            if checkpoint_path is not None and os.path.exists(checkpoint_path):
                chk = torch.load(checkpoint_path, map_location=self.device)
                state_dict = chk.get("model_state_dict", chk)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {checkpoint_path}")
            else:
                logger.warning("No checkpoint loaded; model running with uninitialized/initial weights.")

        self.model.eval()

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        return_orig_size: bool = True,
    ) -> Dict[str, Any]:
        """
        Predicts 19-channel probability map for an input image.
        
        Args:
            image_input: Path to image or RGB np.ndarray (H, W, 3).
            return_orig_size: If True, inverts letterbox to restore original resolution.
            
        Returns:
            Dictionary with:
                - probability_maps: np.ndarray (C, H, W) in [0.0, 1.0]
                - binary_masks: np.ndarray (C, H, W) in {0, 1}
                - detected_classes: List of classes with detected pixels
                - original_image: RGB np.ndarray
        """
        if isinstance(image_input, (str, Path)):
            bgr = cv2.imread(str(image_input))
            if bgr is None:
                raise IOError(f"Could not read image at: {image_input}")
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input

        orig_h, orig_w = img_rgb.shape[:2]

        # Aspect-ratio preserving letterbox resize
        padded_img, _, t_info = letterbox_resize(img_rgb, None, target_size=self.target_size)

        # Normalize and convert to PyTorch tensor
        img_norm = (padded_img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs_padded = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (19, H_pad, W_pad)

        if return_orig_size:
            # Invert letterbox padding
            pad_top = t_info["pad_top"]
            new_h = t_info["new_h"]
            pad_left = t_info["pad_left"]
            new_w = t_info["new_w"]

            unpadded = probs_padded[:, pad_top : pad_top + new_h, pad_left : pad_left + new_w]
            # Resize back to original (orig_h, orig_w)
            probs_orig = np.zeros((self.num_classes, orig_h, orig_w), dtype=np.float32)
            for c in range(self.num_classes):
                probs_orig[c] = cv2.resize(unpadded[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            final_probs = probs_orig
        else:
            final_probs = probs_padded

        binary_masks = (final_probs >= self.threshold).astype(np.uint8)

        detected_classes = []
        for c in range(self.num_classes):
            if np.sum(binary_masks[c]) > 0:
                detected_classes.append(self.class_names[c])

        return {
            "probability_maps": final_probs,
            "binary_masks": binary_masks,
            "detected_classes": detected_classes,
            "original_image": img_rgb,
        }

    def visualize_prediction(
        self,
        prediction_result: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Creates an overlay visualization of the detected defect classes.
        """
        img_rgb = prediction_result["original_image"].copy()
        binary_masks = prediction_result["binary_masks"]
        detected = prediction_result["detected_classes"]

        overlay = img_rgb.copy()
        legend_patches = []

        for c_idx, c_name in enumerate(self.class_names):
            if np.sum(binary_masks[c_idx]) > 0:
                color = (OVERLAY_COLORS[c_idx][:3] * 255).astype(np.uint8)
                mask_c = binary_masks[c_idx] > 0
                colored_mask = np.zeros_like(img_rgb)
                colored_mask[mask_c] = color
                overlay = cv2.addWeighted(overlay, 0.75, colored_mask, 0.25, 0)
                legend_patches.append(mpatches.Patch(color=OVERLAY_COLORS[c_idx], label=c_name))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(overlay)
        ax.axis("off")
        title_str = f"DACL10K Defect Detection ({len(detected)} classes detected)" if detected else "No defects detected"
        ax.set_title(title_str, fontsize=12, weight="bold")

        if legend_patches:
            ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.25, 1), title="Defect Classes")

        plt.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Visualization saved to: {save_path}")
        plt.close(fig)
        return overlay


def main():
    parser = argparse.ArgumentParser(description="Predict DACL10K-v2 Bridge Defects")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    predictor = BridgeDefectPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )

    result = predictor.predict_image(args.image)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_stem = Path(args.image).stem
    # Save probability maps
    npz_path = out_dir / f"{img_stem}_probs.npz"
    np.savez_compressed(
        npz_path,
        probability_maps=result["probability_maps"],
        binary_masks=result["binary_masks"],
        classes=DACL10K_CLASSES,
    )
    logger.info(f"Saved 19 probability maps to {npz_path}")

    # Save visual overlay
    vis_path = out_dir / f"{img_stem}_overlay.png"
    predictor.visualize_prediction(result, save_path=vis_path)


if __name__ == "__main__":
    main()
