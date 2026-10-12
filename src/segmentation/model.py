"""
Module: src.segmentation.model
Description:
    Semantic segmentation model architecture for DACL10K-v2 bridge defect detection.
    
    Selected Baseline:
        DeepLabV3+ with a ResNet-34 backbone (pretrained on ImageNet).
        
    Scientific Rationale:
        1. Multi-scale defect morphology: DACL10K-v2 contains both thin, high-aspect-ratio
           defects (e.g., hairline cracks, exposed rebars) and extensive areal damage
           (e.g., weathering, rust, spalling). DeepLabV3+ employs Atrous Spatial Pyramid
           Pooling (ASPP) with parallel dilation rates (6, 12, 18) to capture multi-scale
           spatial contexts without sacrificing feature map resolution.
        2. Boundary preservation: The decoder module incorporates a low-level feature skip
           connection from the encoder stage 1, recovering sharp defect contours.
        3. Computational balance: ResNet-34 (~21.5M parameters) provides strong representation
           while maintaining low memory footprint and high training throughput on macOS/MPS/CPU.
        4. Independent multi-label output: Produces 19 independent logits (channels) mapped
           via element-wise sigmoid, strictly avoiding cross-class softmax competition.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


class BridgeSegmentationModel(nn.Module):
    """
    Multi-label semantic segmentation baseline for bridge defect detection.
    Outputs 19 independent class logits.
    """

    def __init__(
        self,
        architecture: str = "DeepLabV3Plus",
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        num_classes: int = 19,
    ):
        """
        Args:
            architecture: Segmentation model architecture name ('DeepLabV3Plus', 'Unet', 'FPN').
            encoder_name: Feature extractor backbone (e.g., 'resnet34').
            encoder_weights: Pretrained weights name (e.g., 'imagenet' or None).
            in_channels: Input image channel count (3 for RGB).
            num_classes: Number of defect classes (19 for DACL10K-v2).
        """
        super().__init__()
        self.architecture_name = architecture
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Instantiate primary model
        if architecture.lower() in ["deeplabv3plus", "deeplabv3+"]:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # Critical: raw logits for multi-label BCEWithLogitsLoss
            )
        elif architecture.lower() == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,
            )
        elif architecture.lower() == "fpn":
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,
            )
        else:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Supported options: ['DeepLabV3Plus', 'Unet', 'FPN']."
            )

        logger.info(
            f"Initialized {architecture} with {encoder_name} backbone "
            f"(weights={encoder_weights}, in_channels={in_channels}, classes={num_classes})."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw logits.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W).
            
        Returns:
            logits: Unnormalized logits of shape (B, 19, H, W).
        """
        return self.model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates independent class probabilities via element-wise sigmoid.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W).
            
        Returns:
            probabilities: Float tensor of shape (B, 19, H, W) in [0, 1].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Generates binary multi-label segmentation masks based on probability threshold.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W).
            threshold: Probability decision threshold (default: 0.5).
            
        Returns:
            binary_masks: Byte tensor of shape (B, 19, H, W) with values {0, 1}.
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).to(torch.uint8)

    def get_parameter_counts(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


def build_model(config: Dict[str, Any]) -> BridgeSegmentationModel:
    """
    Factory function to construct model from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'model' subsection.
        
    Returns:
        BridgeSegmentationModel instance.
    """
    m_cfg = config.get("model", {})
    return BridgeSegmentationModel(
        architecture=m_cfg.get("architecture", "DeepLabV3Plus"),
        encoder_name=m_cfg.get("encoder_name", "resnet34"),
        encoder_weights=m_cfg.get("encoder_weights", "imagenet"),
        in_channels=m_cfg.get("in_channels", 3),
        num_classes=m_cfg.get("num_classes", 19),
    )
