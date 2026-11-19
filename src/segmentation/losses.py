"""
Module: src.segmentation.losses
Description:
    Loss functions for multi-label semantic bridge defect segmentation on DACL10K-v2.
    
    Design Principles:
    - Multi-label independence: Evaluates each of the 19 channels as an independent
      binary classification task per pixel.
    - Stable sigmoid formulations: Operates directly on raw model logits.
    - Supports:
        1. BCEWithLogitsLoss (stable baseline, optional pos_weight per class).
        2. MultiLabelSoftDiceLoss (addresses extreme foreground-background spatial sparsity).
        3. CombinedBCEDiceLoss (blends pixel-level cross-entropy with overlap optimization).
        4. MultiLabelFocalLoss (dynamically down-weights easy background pixels).
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiLabelSoftDiceLoss(nn.Module):
    """
    Multi-label Soft Dice Loss computed independently per channel.
    
    Formula per channel c:
        Dice_c = (2 * sum(p_c * y_c) + eps) / (sum(p_c) + sum(y_c) + eps)
        Loss_c = 1.0 - Dice_c
        Loss = mean_c(Loss_c)
    where p_c = sigmoid(logits_c).
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw unnormalized model outputs.
            targets: (B, C, H, W) binary ground truth tensor in {0, 1}.
            
        Returns:
            Scalar loss tensor.
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions: (B, C, H*W)
        b, c = logits.shape[:2]
        probs_flat = probs.view(b, c, -1)
        targets_flat = targets.view(b, c, -1)

        # Intersection and union per sample and channel
        intersection = (probs_flat * targets_flat).sum(dim=2)
        cardinality = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice_score

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        return dice_loss


class ClassWeightedBCEDiceLoss(nn.Module):
    """
    Channel-level class-weighted BCE + Soft Dice loss for multi-label segmentation.
    Weights each class channel c by w_c:
        Loss = sum_c(w_c * (BCE_c + Dice_c)) / sum_c(w_c)
    
    Formula per channel c:
        BCE_c = mean over batch and spatial pixels of BCEWithLogits (reduction='mean')
        Dice_c = mean over batch of (1.0 - SoftDice_c)
    
    Note:
        This is channel-level class weighting applied uniformly across all pixels
        for each class channel c, NOT positive-only BCE weighting (no pos_weight is added).
    """

    def __init__(
        self,
        class_weights: Union[List[float], torch.Tensor],
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if isinstance(class_weights, list):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        elif not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(list(class_weights), dtype=torch.float32)

        if class_weights.ndim != 1:
            raise ValueError(f"class_weights must be 1D, got ndim={class_weights.ndim}")
        if len(class_weights) != 19:
            raise ValueError(f"Expected 19 class weights, got {len(class_weights)}")

        self.register_buffer("class_weights", class_weights.float())
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw unnormalized model outputs.
            targets: (B, C, H, W) binary ground truth tensor in {0, 1}.
        Returns:
            Scalar weighted loss tensor.
        """
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}")

        b, c, h, w = logits.shape
        if c != len(self.class_weights):
            raise ValueError(f"Expected {len(self.class_weights)} channels, got logits with {c} channels")

        # 1. Per-channel BCE loss (averaged over B, H, W) -> shape (C,)
        bce_all = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce_c = bce_all.mean(dim=(0, 2, 3))

        # 2. Per-channel Soft Dice loss (computed per sample then averaged over B) -> shape (C,)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(b, c, -1)
        targets_flat = targets.view(b, c, -1)
        intersection = (probs_flat * targets_flat).sum(dim=2)
        cardinality = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_c = (1.0 - dice_score).mean(dim=0)

        # 3. Channel combined loss -> shape (C,)
        loss_c = self.bce_weight * bce_c + self.dice_weight * dice_c

        # 4. Weighted combination normalized by sum of weights
        weights = self.class_weights.to(logits.device)
        return (weights * loss_c).sum() / weights.sum()


class CombinedBCEDiceLoss(nn.Module):
    """
    Hybrid loss combining Binary Cross Entropy and Soft Dice Loss.
    Combines pixel-level calibration (BCE) with region overlap optimization (Dice).
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = MultiLabelSoftDiceLoss(eps=eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class MultiLabelFocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for multi-label dense prediction.
    Down-weights well-classified background pixels, focusing gradients on hard defects.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def create_loss_function(
    loss_type: str = "bce",
    pos_weight: Optional[torch.Tensor] = None,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    class_weights: Optional[Union[List[float], torch.Tensor]] = None,
) -> nn.Module:
    """
    Factory function for loss modules.
    
    Args:
        loss_type: 'bce', 'dice', 'combined_bce_dice', 'class_weighted_bce_dice', or 'focal'.
        pos_weight: Optional (19,) positive class weights for BCEWithLogitsLoss.
        bce_weight: Scaling factor for BCE in combined loss.
        dice_weight: Scaling factor for Dice in combined loss.
        focal_gamma: Focusing parameter for Focal loss.
        focal_alpha: Class balance parameter for Focal loss.
        class_weights: Optional (19,) channel weights for ClassWeightedBCEDiceLoss.
        
    Returns:
        Instantiated nn.Module loss.
    """
    loss_type = loss_type.lower()
    if loss_type == "bce":
        logger.info(f"Using BCEWithLogitsLoss (pos_weight={pos_weight is not None}).")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "dice":
        logger.info("Using MultiLabelSoftDiceLoss.")
        return MultiLabelSoftDiceLoss()
    elif loss_type in ["combined", "combined_bce_dice", "bce_dice"]:
        logger.info(
            f"Using CombinedBCEDiceLoss (bce_w={bce_weight}, dice_w={dice_weight}, "
            f"pos_weight={pos_weight is not None})."
        )
        return CombinedBCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
        )
    elif loss_type in ["class_weighted_bce_dice", "weighted_bce_dice", "sqrt_weighted_bcedice"]:
        if class_weights is None:
            raise ValueError("class_weights must be provided when loss_type is 'class_weighted_bce_dice'")
        logger.info(
            f"Using ClassWeightedBCEDiceLoss (bce_w={bce_weight}, dice_w={dice_weight}, "
            f"num_classes={len(class_weights)})."
        )
        return ClassWeightedBCEDiceLoss(
            class_weights=class_weights,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )
    elif loss_type == "focal":
        logger.info(f"Using MultiLabelFocalLoss (alpha={focal_alpha}, gamma={focal_gamma}).")
        return MultiLabelFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. Expected 'bce', 'dice', 'combined', 'class_weighted_bce_dice', or 'focal'."
        )
