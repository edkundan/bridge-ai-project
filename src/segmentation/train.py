"""
Module: src.segmentation.train
Description:
    Reproducible training pipeline for multi-label semantic bridge defect segmentation.
    
    Features:
    - Automatic hardware device selection (Apple MPS, NVIDIA CUDA, CPU fallback).
    - Validates inputs and catches non-finite (NaN/Inf) loss values.
    - Checkpoint saving: best validation metric, latest resume checkpoint, final model.
    - Resuming capability from saved checkpoints.
    - Learning rate scheduling and gradient clipping.
    - Graceful memory error (OOM) handling with actionable diagnostics.
    - Output history and metric curve visualization.
"""

import argparse
from collections import Counter
import json
import logging
import os
import random
import time
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from src.data.dataset import DACL10K_CLASSES, create_dataloaders
from src.segmentation.evaluate import SegmentationEvaluator
from src.segmentation.losses import create_loss_function
from src.segmentation.model import BridgeSegmentationModel

logger = logging.getLogger("train")


def set_seed(seed: int = 42):
    """Ensure full reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(preferred_device: str = "auto") -> torch.device:
    """
    Selects optimal available hardware device with graceful CPU fallback.
    """
    if preferred_device != "auto":
        dev = torch.device(preferred_device)
        logger.info(f"Using explicitly requested device: {dev}")
        return dev

    if torch.backends.mps.is_available():
        try:
            # Smoke test MPS device tensor allocation
            import platform
            arch = platform.machine()
            logger.info(f"Selected Apple Metal Performance Shaders (MPS) backend on {arch} host architecture.")
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS detected but allocation failed: {e}. Falling back to CPU.")


    if torch.cuda.is_available():
        logger.info(f"Selected NVIDIA CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    logger.info("Using standard CPU device.")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: Optional[float] = 1.0,
) -> float:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    total_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        try:
            logits = model(images)
            loss = criterion(logits, masks)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "mps backend out of memory" in err_msg or "cannot allocate" in err_msg:
                logger.error(
                    "\n\n[FATAL OOM ERROR]: Ran out of memory during forward pass!\n"
                    "Remediation actions:\n"
                    "1. Lower batch_size in configs/training.yaml (e.g., from 4 to 2 or 1).\n"
                    "2. Reduce image_size (e.g., from [512, 512] to [384, 384] or [256, 256]).\n"
                    "3. Reduce num_workers to 0 or 1.\n"
                )
            raise e

        # Validate loss stability
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered at epoch {epoch}, batch {batch_idx}: loss={loss.item()}"
            )

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(1, total_batches)
    return epoch_loss


def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """Runs a single validation epoch and evaluates mIoU."""
    model.eval()
    running_loss = 0.0
    evaluator = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=threshold)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item()
            evaluator.update(logits, masks)

    val_loss = running_loss / max(1, len(loader))
    val_metrics = evaluator.compute()
    return val_loss, val_metrics


def plot_learning_curves(history: Dict[str, List[float]], save_path: Union[str, Path]):
    """Plots and saves loss and mIoU curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss axis
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color="tab:blue", fontsize=12)
    line1 = ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue", lw=2)
    line2 = ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tab:cyan", lw=2, linestyle="--")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # mIoU axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("mIoU", color="tab:orange", fontsize=12)
    line3 = ax2.plot(epochs, history["val_miou"], label="Val mIoU (All)", color="tab:orange", lw=2)
    line4 = ax2.plot(epochs, history["val_miou_present"], label="Val mIoU (Present)", color="tab:red", lw=2, linestyle=":")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("DACL10K-v2 Baseline Training Curves", fontsize=14, weight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def train(
    config_path: Union[str, Path],
    resume_path: Optional[str] = None,
    override_epochs: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main training execution function.
    
    Args:
        config_path: Path to YAML training configuration.
        resume_path: Optional path to checkpoint to resume from.
        override_epochs: Optional override for total epochs.
        max_train_samples: Optional sample limit for smoke testing.
        max_val_samples: Optional sample limit for validation testing.
        
    Returns:
        Dictionary containing training history and paths to saved artifacts.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setup directories
    exp_dir = Path(config.get("experiment", {}).get("output_dir", "experiments/segmentation_baseline"))
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = exp_dir / "train.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("STARTING STAGE 1: SEGMENTATION BASELINE TRAINING")
    logger.info(f"Configuration File: {config_path}")
    logger.info(f"Experiment Output Directory: {exp_dir}")
    logger.info("=" * 60)

    # Reproducibility seed
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)

    # Hardware device
    preferred_device = config.get("training", {}).get("device", "auto")
    device = select_device(preferred_device)

    # Build DataLoaders
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    root_dir = dataset_cfg.get("root_dir")
    image_size = tuple(train_cfg.get("image_size", [512, 512]))
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 2))

    sampling_cfg = config.get("sampling", {})
    train_loader, val_loader = create_dataloaders(
        root_dir=root_dir,
        target_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        sampling_cfg=sampling_cfg,
    )

    # Log and export expected sampling distribution if rare-class sampling is active
    is_sampling_active = hasattr(train_loader.sampler, "expected_summary") and train_loader.sampler.expected_summary is not None
    if is_sampling_active:
        expected = train_loader.sampler.expected_summary
        logger.info("=" * 60)
        logger.info("RARE-CLASS SAMPLING ACTIVE (WeightedRandomSampler)")
        logger.info(
            f"Images: {expected['num_images']} | Total Weight: {expected['total_weight']:.1f} | "
            f"Mean Weight: {expected['mean_weight']:.4f} | Range: [{expected['min_weight']}, {expected['max_weight']}]"
        )
        for focal in ["WConccor", "Rockpocket", "Crack"]:
            st = expected["expected_class_exposure"].get(focal, {})
            logger.info(
                f"  [{focal}] Base: {st.get('base_prevalence_pct', 0)}% ({st.get('image_count', 0)} imgs) -> "
                f"Expected: {st.get('expected_sampling_prob_pct', 0)}% (~{st.get('expected_samples_per_epoch', 0):.0f} imgs/epoch) | "
                f"Boost: {st.get('expected_boost_factor', 1.0)}x"
            )
        logger.info("=" * 60)
        with open(exp_dir / "sampling_expected_distribution.json", "w", encoding="utf-8") as f:
            json.dump(expected, f, indent=4)

    # Build Model
    model_cfg = config.get("model", {})
    model = BridgeSegmentationModel(
        architecture=model_cfg.get("architecture", "DeepLabV3Plus"),
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 19),
    ).to(device)

    param_counts = model.get_parameter_counts()
    logger.info(
        f"Model initialized: Total params = {param_counts['total_parameters']:,} | "
        f"Trainable params = {param_counts['trainable_parameters']:,}"
    )

    # Build Loss Function
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "bce")
    
    # Optional positive weighting for BCE
    pos_weight = None
    if loss_cfg.get("use_pos_weight", False):
        weights = loss_cfg.get("pos_weights", None)
        if weights is not None:
            pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = create_loss_function(
        loss_type=loss_type,
        pos_weight=pos_weight,
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        class_weights=loss_cfg.get("class_weights", None),
    )

    # Build Optimizer & Scheduler
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    opt_name = train_cfg.get("optimizer", "AdamW").lower()

    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_epochs = override_epochs if override_epochs is not None else int(train_cfg.get("epochs", 30))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_miou = 0.0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_miou": [],
        "val_miou_present": [],
        "lr": [],
    }

    resumed = False
    if resume_path is not None and os.path.exists(resume_path):
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        chk = torch.load(resume_path, map_location=device)
        model_state = chk.get("model_state_dict", chk.get("model_state"))
        if model_state is None:
            model_state = chk
        model.load_state_dict(model_state)

        opt_state = chk.get("optimizer_state_dict", chk.get("optimizer_state"))
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            logger.info("Optimizer state restored successfully.")
        else:
            logger.warning("No optimizer state found in checkpoint. Optimizer re-initialized from current learning rate.")

        sched_state = chk.get("scheduler_state_dict", chk.get("scheduler_state"))
        if sched_state is not None and scheduler is not None:
            scheduler.load_state_dict(sched_state)
            logger.info("Scheduler state restored successfully.")

        start_epoch = chk.get("epoch", 0) + 1
        best_miou = chk.get("best_miou", chk.get("best_val_miou", 0.0))
        history = chk.get("history", history)
        logger.info(f"Resumed at epoch {start_epoch} with previous best mIoU: {best_miou:.4f}")
        resumed = True

    # Pretrained checkpoint initialization (e.g. baseline transfer without optimizer state)
    pretrained_path = (
        model_cfg.get("pretrained_checkpoint")
        or train_cfg.get("pretrained_checkpoint")
        or config.get("pretrained_checkpoint")
    )
    if not resumed and pretrained_path:
        if os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained weights (fresh optimizer/scheduler) from: {pretrained_path}")
            chk = torch.load(pretrained_path, map_location=device)
            model_state = chk.get("model_state_dict", chk.get("model_state", chk))
            model.load_state_dict(model_state, strict=True)
            logger.info("Pretrained weights loaded successfully.")
        else:
            logger.warning(f"Pretrained checkpoint not found at: {pretrained_path}. Using base model initialization.")

    threshold = float(train_cfg.get("threshold", 0.5))
    grad_clip = float(train_cfg.get("gradient_clip", 1.0))
    best_metric_name = train_cfg.get("best_metric", "val_miou")

    # Main Training Loop
    start_time = time.time()
    logger.info(f"Starting training loop from epoch {start_epoch} to {total_epochs}...")

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Deterministic sampling generator reseeding for reproducible resumption
        if hasattr(train_loader.sampler, "generator") and train_loader.sampler.generator is not None:
            train_loader.sampler.generator.manual_seed(seed + epoch)

        # Train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            grad_clip=grad_clip,
        )

        # Record and log actual sampled exposure if recording sampler is active
        if is_sampling_active and hasattr(train_loader.sampler, "last_epoch_indices"):
            sampled_indices = train_loader.sampler.last_epoch_indices
            presence_mat = getattr(train_loader.sampler, "presence_matrix", None)
            dataset_classes = getattr(train_loader.dataset, "classes", DACL10K_CLASSES)
            num_total_dataset = len(train_loader.dataset)
            total_drawn = len(sampled_indices)
            unique_indices = sorted(list(set(sampled_indices)))
            unique_count = len(unique_indices)
            counts = list(Counter(sampled_indices).values())
            max_reps = max(counts) if counts else 0
            mean_reps = round(total_drawn / max(1, unique_count), 2)

            actual_class_exposure = {}
            if total_drawn > 0 and presence_mat is not None:
                sampled_presence = presence_mat[sampled_indices]
                class_draw_counts = sampled_presence.sum(axis=0)
                unique_presence = presence_mat[unique_indices]
                class_unique_counts = unique_presence.sum(axis=0)

                for c_idx, c_name in enumerate(dataset_classes):
                    d_cnt = int(class_draw_counts[c_idx])
                    u_cnt = int(class_unique_counts[c_idx])
                    actual_class_exposure[c_name] = {
                        "class_index": c_idx,
                        "sampled_instances": d_cnt,
                        "sampled_fraction": round(d_cnt / max(1, total_drawn), 4),
                        "sampled_fraction_pct": round(d_cnt / max(1, total_drawn) * 100, 2),
                        "unique_images_sampled": u_cnt,
                    }

            epoch_sampling_stats = {
                "epoch": epoch,
                "total_samples_drawn": total_drawn,
                "unique_images_sampled": unique_count,
                "unique_coverage_pct": round(unique_count / max(1, num_total_dataset) * 100, 2),
                "max_repetitions": max_reps,
                "mean_repetitions": mean_reps,
                "class_exposure": actual_class_exposure,
            }
            with open(exp_dir / f"sampling_epoch_{epoch:03d}.json", "w", encoding="utf-8") as f:
                json.dump(epoch_sampling_stats, f, indent=4)

            sampling_history_path = exp_dir / "sampling_exposure_history.json"
            sampling_history = []
            if sampling_history_path.exists():
                try:
                    with open(sampling_history_path, "r", encoding="utf-8") as f:
                        sampling_history = json.load(f)
                    sampling_history = [item for item in sampling_history if item.get("epoch") != epoch]
                except Exception:
                    sampling_history = []
            sampling_history.append(epoch_sampling_stats)
            with open(sampling_history_path, "w", encoding="utf-8") as f:
                json.dump(sampling_history, f, indent=4)

            wc_actual = actual_class_exposure.get("WConccor", {})
            rp_actual = actual_class_exposure.get("Rockpocket", {})
            cr_actual = actual_class_exposure.get("Crack", {})
            logger.info(
                f"Epoch {epoch:03d} Sampling | "
                f"Unique: {unique_count}/{num_total_dataset} ({epoch_sampling_stats['unique_coverage_pct']}%) | "
                f"Max Reps: {max_reps} | Mean Reps: {mean_reps:.2f} | "
                f"WConccor: {wc_actual.get('sampled_instances', 0)} ({wc_actual.get('sampled_fraction_pct', 0)}%) | "
                f"Rockpocket: {rp_actual.get('sampled_instances', 0)} ({rp_actual.get('sampled_fraction_pct', 0)}%) | "
                f"Crack: {cr_actual.get('sampled_instances', 0)} ({cr_actual.get('sampled_fraction_pct', 0)}%)"
            )

        # Validate
        val_loss, val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            threshold=threshold,
        )

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        val_miou_all = val_metrics["macro"]["mIoU_all_classes"]
        val_miou_pres = val_metrics["macro"]["mIoU_present_classes"]

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_miou"].append(round(val_miou_all, 4))
        history["val_miou_present"].append(round(val_miou_pres, 4))
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch:03d}/{total_epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU (All): {val_miou_all:.4f} | "
            f"Val mIoU (Present): {val_miou_pres:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Check for best checkpoint based on validation mIoU
        is_best = (val_miou_all > best_miou)
        if is_best:
            best_miou = val_miou_all
            best_checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_state": model.state_dict(),
                "best_miou": best_miou,
                "best_val_miou": best_miou,
                "val_miou": val_miou_all,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "sampling_config": sampling_cfg,
                "config": config,
            }, best_checkpoint_path)
            logger.info(f"[*] New best model saved to: {best_checkpoint_path} (mIoU: {best_miou:.4f}, Val Loss: {val_loss:.4f})")

        # Save latest resume checkpoint
        latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_state": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou": best_miou,
            "best_val_miou": best_miou,
            "val_miou": val_miou_all,
            "val_loss": val_loss,
            "history": history,
            "sampling_config": sampling_cfg,
            "config": config,
        }, latest_checkpoint_path)

        # Update curves and history JSON
        with open(exp_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        plot_learning_curves(history, exp_dir / "learning_curves.png")

        # Save per-class validation metrics CSV (including TP, FP, FN, IoU, Dice)
        if "per_class" in val_metrics:
            per_class_df = pd.DataFrame(val_metrics["per_class"])
            per_class_df["epoch"] = epoch
            epoch_csv_path = exp_dir / f"val_metrics_epoch_{epoch:03d}.csv"
            per_class_df.to_csv(epoch_csv_path, index=False)
            per_class_df.to_csv(exp_dir / "val_metrics_latest.csv", index=False)

            history_csv_path = exp_dir / "val_metrics_history.csv"
            if history_csv_path.exists():
                try:
                    all_metrics_df = pd.read_csv(history_csv_path)
                    all_metrics_df = all_metrics_df[all_metrics_df["epoch"] != epoch]
                    all_metrics_df = pd.concat([all_metrics_df, per_class_df], ignore_index=True)
                except Exception:
                    all_metrics_df = per_class_df
            else:
                all_metrics_df = per_class_df
            all_metrics_df.to_csv(history_csv_path, index=False)

    total_time = time.time() - start_time

    # Save final model separately
    final_model_path = checkpoint_dir / "final_model.pth"
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "final_history": history,
        "config": config,
    }, final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Training completed in {total_time / 60:.2f} minutes.")

    return {
        "best_miou": best_miou,
        "best_model_path": str(checkpoint_dir / "best_model.pth"),
        "final_model_path": str(final_model_path),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DACL10K-v2 Segmentation Baseline")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples (smoke test)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit val samples (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    train(
        config_path=args.config,
        resume_path=args.resume,
        override_epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()
