"""
Script: src.segmentation.smoke_test
Description:
    End-to-end smoke test on real DACL10K-v2 dataset samples.
    
    Verifies:
    1. Loading real images and matching JSON annotations.
    2. 19-channel multi-label binary mask shapes (19, H, W).
    3. Visualizing input image and active mask defect channels.
    4. Forward pass producing (B, 19, H, W) logits.
    5. Loss calculation (finite and non-negative).
    6. Backward pass and gradient flow.
    7. Optimizer step and model parameter weight updates (norm diff > 0).
"""

import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.dataset import DACL10KDataset, DACL10K_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from src.segmentation.losses import CombinedBCEDiceLoss
from src.segmentation.model import BridgeSegmentationModel
from src.segmentation.train import select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("smoke_test")


def run_smoke_test(config_path: str = "configs/training.yaml") -> bool:
    logger.info("=" * 60)
    logger.info("EXECUTING STAGE 1 SMOKE TEST ON REAL DATASET SAMPLES")
    logger.info("=" * 60)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    root_dir = config["dataset"]["root_dir"]
    target_size = tuple(config["training"]["image_size"])
    output_dir = Path("outputs/smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load several images and masks
    logger.info("Step 1: Loading 4 real images and annotations from DACL10K-v2 train split...")
    dataset = DACL10KDataset(
        root_dir=root_dir,
        split="train",
        target_size=target_size,
        augment=False,
        max_samples=4,
        return_metadata=True,
    )
    assert len(dataset) >= 4, f"Expected at least 4 samples, got {len(dataset)}"
    logger.info(f"Loaded {len(dataset)} samples successfully.")

    # 2. Verify 19-channel mask shape
    logger.info("Step 2: Verifying 19-channel mask shape...")
    batch_images = []
    batch_masks = []
    sample_metas = []

    for i in range(4):
        img_t, mask_t, meta = dataset[i]
        logger.info(
            f"  Sample {i} [{meta['image_id']}]: "
            f"Image tensor = {tuple(img_t.shape)}, "
            f"Mask tensor = {tuple(mask_t.shape)}, "
            f"Active classes = {meta.get('classes_present', [])}"
        )
        assert img_t.shape == (3, target_size[0], target_size[1]), f"Unexpected image shape: {img_t.shape}"
        assert mask_t.shape == (19, target_size[0], target_size[1]), f"Unexpected mask shape: {mask_t.shape}"
        
        # Verify mask is binary {0, 1}
        vals = set(torch.unique(mask_t).numpy())
        assert vals.issubset({0.0, 1.0}), f"Mask must be binary {0.0, 1.0}, found {vals}"

        batch_images.append(img_t)
        batch_masks.append(mask_t)
        sample_metas.append(meta)

    images_batch = torch.stack(batch_images, dim=0) # (4, 3, 512, 512)
    masks_batch = torch.stack(batch_masks, dim=0)   # (4, 19, 512, 512)
    logger.info(f"Batch shapes: Images = {tuple(images_batch.shape)}, Masks = {tuple(masks_batch.shape)}")

    # 3. Visualize image and selected mask channels
    logger.info("Step 3: Visualizing image and active mask channels...")
    vis_path = output_dir / "smoke_test_visualization.png"

    # Select the sample with highest defect presence
    active_idx = 0
    max_active = -1
    for i, meta in enumerate(sample_metas):
        cnt = len(meta.get("classes_present", []))
        if cnt > max_active:
            max_active = cnt
            active_idx = i

    sample_img = images_batch[active_idx].numpy().transpose(1, 2, 0)
    sample_img = np.clip((sample_img * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255).astype(np.uint8)
    sample_mask = masks_batch[active_idx].numpy()

    # Find active channels
    active_channels = [c for c in range(19) if np.sum(sample_mask[c]) > 0]
    if not active_channels:
        active_channels = [0, 3, 14]  # fallback to ACrack, Crack, Rust

    num_panels = 1 + min(3, len(active_channels))
    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))
    axes[0].imshow(sample_img)
    axes[0].set_title(f"Input: {sample_metas[active_idx]['image_id']}", fontsize=11)
    axes[0].axis("off")

    for panel_idx, ch in enumerate(active_channels[:3]):
        ax = axes[panel_idx + 1]
        ch_name = DACL10K_CLASSES[ch]
        pos_pixels = int(np.sum(sample_mask[ch]))
        ax.imshow(sample_mask[ch], cmap="hot")
        ax.set_title(f"Channel {ch} ({ch_name}): {pos_pixels} px", fontsize=11)
        ax.axis("off")

    plt.suptitle("Smoke Test: Input Image & Multi-Label Mask Channels", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(vis_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved visualization to: {vis_path}")

    # 4. Run one forward pass
    logger.info("Step 4: Running one forward pass...")
    device = select_device(config["training"].get("device", "auto"))
    logger.info(f"Target execution device: {device}")

    # Build model (load without pretrained weights download for smoke test if offline, or with imagenet)
    model = BridgeSegmentationModel(
        architecture="DeepLabV3Plus",
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=19,
    ).to(device)
    model.train()

    inputs = images_batch.to(device)
    targets = masks_batch.to(device)

    logits = model(inputs)
    logger.info(f"Forward output logits shape: {tuple(logits.shape)}")
    assert logits.shape == (4, 19, target_size[0], target_size[1]), f"Wrong logits shape: {logits.shape}"

    # 5. Calculate loss
    logger.info("Step 5: Calculating Combined BCE + Soft Dice Loss...")
    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    loss = criterion(logits, targets)
    loss_val = loss.item()
    logger.info(f"Computed loss value: {loss_val:.6f}")
    assert torch.isfinite(loss), f"Loss is non-finite: {loss_val}"
    assert loss_val > 0.0, f"Loss should be positive: {loss_val}"

    # 6. Backward pass and optimizer step
    logger.info("Step 6: Running backward pass and optimizer step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Track parameter weights before step
    test_param = next(p for p in model.parameters() if p.requires_grad)
    weight_before = test_param.clone().detach()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 7. Verify model parameters update
    logger.info("Step 7: Verifying model parameter updates...")
    weight_after = test_param.clone().detach()
    delta_norm = torch.norm(weight_after - weight_before).item()
    logger.info(f"Parameter delta L2-norm after 1 optimizer step: {delta_norm:.6e}")
    assert delta_norm > 0.0, "Model weights failed to update!"

    logger.info("=" * 60)
    logger.info("SMOKE TEST PASSED WITH FLYING COLORS!")
    logger.info("All 7 steps verified: data loading, 19-ch masks, visualization,")
    logger.info("forward pass, loss calculation, backward pass, and parameter updates.")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
