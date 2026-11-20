"""
Module: src.data.dataset
Description:
    PyTorch Dataset and DataLoader pipelines for DACL10K-v2 multi-label bridge
    defect segmentation.
    
    Key Features:
    - Aspect-ratio preserving letterbox resizing to avoid crushing thin cracks.
    - Synchronized image and multi-channel mask transformations.
    - 19 independent binary channels per mask.
    - Strict handling of train/validation/testdev splits (no testdev in supervised train/val).
    - Reproducible DataLoader worker seeding.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.mask_converter import DACL10K_CLASSES, MaskConverter

logger = logging.getLogger(__name__)

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_resize(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    target_size: Tuple[int, int] = (512, 512),
    pad_value: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Resizes image and mask while strictly preserving original aspect ratio.
    Pads the remaining boundaries with pad_value to achieve target_size (H, W).
    
    Args:
        image: np.ndarray (H, W, 3) uint8 or float32.
        mask: Optional np.ndarray (H, W, C) uint8 or float32.
        target_size: Tuple of (target_height, target_width).
        pad_value: Value for image border padding.
        
    Returns:
        padded_image: np.ndarray of shape (target_height, target_width, 3).
        padded_mask: np.ndarray of shape (target_height, target_width, C) or None.
        transform_info: Dict with scale, pad_top, pad_left, orig_h, orig_w.
    """
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]

    # Compute scale factor to fit within target bounding box
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    # Resize image with bilinear interpolation
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute symmetric padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_img = cv2.copyMakeBorder(
        resized_img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value]
    )

    padded_mask = None
    if mask is not None:
        # Resize mask using NEAREST neighbor to preserve strictly binary {0, 1} values
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask retains 3D shape (H, W, C) even if C=1
        if mask.ndim == 3 and resized_mask.ndim == 2:
            resized_mask = resized_mask[:, :, np.newaxis]

        padded_mask = cv2.copyMakeBorder(
            resized_mask, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        if mask.ndim == 3 and padded_mask.ndim == 2:
            padded_mask = padded_mask[:, :, np.newaxis]

    transform_info = {
        "scale": scale,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
    }

    return padded_img, padded_mask, transform_info


class DACL10KDataset(Dataset):
    """
    DACL10K-v2 Multi-Label Semantic Segmentation Dataset.
    Loads bridge images and matching LabelMe JSON polygon annotations,
    returning normalized image tensors and 19-channel binary mask tensors.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        target_size: Tuple[int, int] = (512, 512),
        classes: Optional[List[str]] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,
        sample_filenames: Optional[List[str]] = None,
        return_metadata: bool = False,
    ):
        """
        Args:
            root_dir: Path to DACL10K-v2 dataset root directory.
            split: Dataset split ('train', 'validation', or 'testdev').
            target_size: (height, width) for model inputs.
            classes: Optional custom class list (default: 19 DACL10K classes).
            augment: If True, applies random horizontal/vertical flips.
            max_samples: Optional limit on number of samples (useful for smoke tests).
            sample_filenames: Optional explicit list of image/json filenames to load.
            return_metadata: If True, __getitem__ returns (image, mask, metadata).
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_size = tuple(target_size)
        self.classes = classes if classes is not None else list(DACL10K_CLASSES)
        self.augment = augment
        self.return_metadata = return_metadata

        self.mask_converter = MaskConverter(class_names=self.classes)
        self.num_classes = len(self.classes)

        # Enforce that testdev split is never used for supervised training/validation
        if self.split not in ["train", "validation", "testdev"]:
            raise ValueError(f"Split '{self.split}' unrecognized. Expected 'train', 'validation', or 'testdev'.")

        self.image_dir = self.root_dir / "images" / self.split
        self.anno_dir = self.root_dir / "annotations" / self.split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        if self.split == "testdev":
            logger.info("Initializing dataset for 'testdev' split (unannotated test partition).")
            # Discover images only
            self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))
            self.anno_paths = [None] * len(self.image_paths)
        else:
            if not self.anno_dir.exists():
                raise FileNotFoundError(f"Annotation directory does not exist: {self.anno_dir}")
            
            # Match images with annotations by stem
            all_images = {p.stem: p for p in self.image_dir.glob("*.jpg")}
            all_annos = {p.stem: p for p in self.anno_dir.glob("*.json")}

            common_stems = sorted(list(set(all_images.keys()) & set(all_annos.keys())))
            
            if len(common_stems) == 0:
                raise RuntimeError(f"No matching image-annotation pairs found in split '{self.split}'.")

            self.image_paths = [all_images[s] for s in common_stems]
            self.anno_paths = [all_annos[s] for s in common_stems]

        # Explicit filename filtering if specified
        if sample_filenames is not None:
            stems_to_keep = [Path(f).stem for f in sample_filenames]
            filtered_images = []
            filtered_annos = []
            img_map = {p.stem: p for p in self.image_paths}
            ann_map = {p.stem: p for p in self.anno_paths if p is not None}
            for s in stems_to_keep:
                if s in img_map:
                    filtered_images.append(img_map[s])
                    filtered_annos.append(ann_map.get(s, None))
            self.image_paths = filtered_images
            self.anno_paths = filtered_annos

        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            self.anno_paths = self.anno_paths[:max_samples]


        logger.info(
            f"DACL10KDataset [{self.split}]: {len(self.image_paths)} samples loaded. "
            f"Target size: {self.target_size}. Augmentation: {self.augment}."
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_augmentations(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random flips consistently to image and mask."""
        # Random horizontal flip (p = 0.5)
        if random.random() > 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1, :])

        # Random vertical flip (p = 0.5)
        if random.random() > 0.5:
            image = np.ascontiguousarray(image[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :, :])

        return image, mask

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        img_path = self.image_paths[idx]
        anno_path = self.anno_paths[idx]

        # Read image via OpenCV (BGR -> RGB)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise IOError(f"Failed to read image at: {img_path}")
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        metadata = {
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "orig_height": orig_h,
            "orig_width": orig_w,
        }

        # Build multi-label mask
        if anno_path is not None:
            mask_np, anno_meta = self.mask_converter.json_to_multilabel_mask(
                anno_path, height=orig_h, width=orig_w
            )
            metadata.update(anno_meta)
        else:
            # Unannotated (e.g. testdev)
            mask_np = np.zeros((orig_h, orig_w, self.num_classes), dtype=np.uint8)
            metadata["is_empty"] = True
            metadata["classes_present"] = []

        # Geometry preservation via Letterbox resize
        padded_img, padded_mask, t_info = letterbox_resize(
            image_rgb, mask_np, target_size=self.target_size
        )
        metadata["transform_info"] = t_info

        # Augmentation (training split only)
        if self.augment and padded_mask is not None:
            padded_img, padded_mask = self._apply_augmentations(padded_img, padded_mask)

        # Normalize image: float32 in [0, 1] then ImageNet standardization
        img_float = padded_img.astype(np.float32) / 255.0
        img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        # HWC -> CHW tensor: (3, H, W)
        image_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Mask tensor: HWC -> CHW tensor: (19, H, W) with values in {0.0, 1.0}
        mask_tensor = torch.from_numpy(padded_mask.transpose(2, 0, 1)).float()

        if self.return_metadata:
            return image_tensor, mask_tensor, metadata
        return image_tensor, mask_tensor


def seed_worker(worker_id: int):
    """Seed DataLoader worker processes for deterministic batch generation."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def custom_collate_fn(batch: List[Any]) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]]:
    """
    Collate function supporting both (image, mask) pairs and (image, mask, metadata) tuples
    where metadata contains variable-length lists of present classes.
    """
    if len(batch[0]) == 3:
        images = torch.stack([item[0] for item in batch], dim=0)
        masks = torch.stack([item[1] for item in batch], dim=0)
        metas = [item[2] for item in batch]
        return images, masks, metas
    else:
        images = torch.stack([item[0] for item in batch], dim=0)
        masks = torch.stack([item[1] for item in batch], dim=0)
        return images, masks


class RecordingWeightedRandomSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler that records the sampled indices for each epoch iteration.
    Enables exact monitoring of unique image coverage, repetition counts, and actual
    per-class exposure without disrupting PyTorch's native sampling loop.
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(weights=weights, num_samples=num_samples, replacement=replacement, generator=generator)
        self.last_epoch_indices: List[int] = []
        self.presence_matrix: Optional[np.ndarray] = None
        self.expected_summary: Optional[Dict[str, Any]] = None
        self.weights_list: List[float] = list(weights)

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        self.last_epoch_indices = rand_tensor.tolist()
        yield from iter(self.last_epoch_indices)


def compute_rare_class_sampling_weights(
    dataset: DACL10KDataset,
    sampling_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], np.ndarray, Dict[str, Any]]:
    """
    Computes per-image sampling weights for rare-class-aware oversampling on the training split.
    Determines class presence from the actual training annotation JSON files using canonical classes.
    
    Formula:
        weight = min(max_weight, base_weight + wconccor_boost * has_WConccor
                                            + rockpocket_boost * has_Rockpocket
                                            + crack_boost * has_Crack)
    Default parameters:
        base_weight = 1.0
        wconccor_boost = 2.0
        rockpocket_boost = 1.5
        crack_boost = 0.5
        max_weight = 4.0
        
    Args:
        dataset: DACL10KDataset instance (strictly aligns with dataset.anno_paths).
        sampling_cfg: Optional dictionary of sampling parameters.
        
    Returns:
        weights: List[float] of length len(dataset), strictly 1:1 aligned with dataset indices.
        presence_matrix: np.ndarray of shape (len(dataset), num_classes) with binary indicators {0, 1}.
        summary: Dict containing expected sampling probabilities and boost factors.
    """
    cfg = sampling_cfg or {}
    base_weight = float(cfg.get("base_weight", 1.0))
    wc_boost = float(cfg.get("wconccor_boost", 2.0))
    rp_boost = float(cfg.get("rockpocket_boost", 1.5))
    cr_boost = float(cfg.get("crack_boost", 0.5))
    max_weight = float(cfg.get("max_weight", 4.0))

    class_names = dataset.classes
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    num_samples = len(dataset)

    presence_matrix = np.zeros((num_samples, num_classes), dtype=np.uint8)
    weights = []

    wc_idx = class_to_idx.get("WConccor")
    rp_idx = class_to_idx.get("Rockpocket")
    cr_idx = class_to_idx.get("Crack")

    for i, anno_path in enumerate(dataset.anno_paths):
        has_wc = 0
        has_rp = 0
        has_cr = 0

        if anno_path is not None and os.path.exists(anno_path):
            try:
                with open(anno_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                labels = {s.get("label") for s in data.get("shapes", [])}
                for l in labels:
                    if l in class_to_idx:
                        presence_matrix[i, class_to_idx[l]] = 1
                if wc_idx is not None:
                    has_wc = int(presence_matrix[i, wc_idx])
                if rp_idx is not None:
                    has_rp = int(presence_matrix[i, rp_idx])
                if cr_idx is not None:
                    has_cr = int(presence_matrix[i, cr_idx])
            except Exception as e:
                logger.warning(f"Failed to read annotation at {anno_path}: {e}")

        raw_w = base_weight + wc_boost * has_wc + rp_boost * has_rp + cr_boost * has_cr
        w = min(max_weight, raw_w)
        weights.append(float(w))

    total_w = sum(weights)
    mean_w = total_w / max(1, num_samples)

    # Compute expected probabilities and exposure per class
    expected_stats = {}
    for c_idx, c_name in enumerate(class_names):
        c_count = int(presence_matrix[:, c_idx].sum())
        c_base_prob = c_count / max(1, num_samples)
        c_weight_sum = float(sum(w for w, has_c in zip(weights, presence_matrix[:, c_idx]) if has_c > 0))
        c_expected_prob = c_weight_sum / max(1e-8, total_w)
        c_expected_count = c_expected_prob * num_samples
        boost_factor = c_expected_prob / max(1e-8, c_base_prob) if c_base_prob > 0 else 1.0

        expected_stats[c_name] = {
            "class_index": c_idx,
            "image_count": c_count,
            "base_prevalence": round(c_base_prob, 4),
            "base_prevalence_pct": round(c_base_prob * 100, 2),
            "expected_sampling_prob": round(c_expected_prob, 4),
            "expected_sampling_prob_pct": round(c_expected_prob * 100, 2),
            "expected_samples_per_epoch": round(c_expected_count, 1),
            "expected_boost_factor": round(boost_factor, 2),
        }

    summary = {
        "num_images": num_samples,
        "total_weight": round(total_w, 2),
        "mean_weight": round(mean_w, 4),
        "min_weight": round(min(weights) if weights else 1.0, 2),
        "max_weight": round(max(weights) if weights else 1.0, 2),
        "expected_class_exposure": expected_stats,
    }

    return weights, presence_matrix, summary


def create_dataloaders(
    root_dir: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 42,
    pin_memory: bool = False,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    sampling_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to build training and validation DataLoaders.
    Supports optional rare-class oversampling for training via WeightedRandomSampler.
    Validation loader remains strictly unweighted, without replacement or augmentation.
    
    Args:
        root_dir: DACL10K-v2 dataset root path.
        target_size: (H, W) model resolution.
        batch_size: Mini-batch size.
        num_workers: Number of worker processes.
        seed: Random seed for worker initialization.
        pin_memory: PyTorch pin_memory flag.
        max_train_samples: Optional subset limit for training.
        max_val_samples: Optional subset limit for validation.
        sampling_cfg: Optional configuration dictionary for rare-class sampling.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = DACL10KDataset(
        root_dir=root_dir,
        split="train",
        target_size=target_size,
        augment=True,
        max_samples=max_train_samples,
    )

    val_dataset = DACL10KDataset(
        root_dir=root_dir,
        split="validation",
        target_size=target_size,
        augment=False,
        max_samples=max_val_samples,
    )

    # Configure training sampler if enabled
    sampling_enabled = bool(sampling_cfg and sampling_cfg.get("enabled", False))
    if sampling_enabled:
        weights, presence_matrix, summary = compute_rare_class_sampling_weights(train_dataset, sampling_cfg)
        sampler_gen = torch.Generator()
        sampler_gen.manual_seed(seed)
        replacement = bool(sampling_cfg.get("replacement", True))

        train_sampler = RecordingWeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=replacement,
            generator=sampler_gen,
        )
        train_sampler.presence_matrix = presence_matrix
        train_sampler.expected_summary = summary
        train_sampler.seed = seed

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            drop_last=True if len(train_dataset) > batch_size else False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True if len(train_dataset) > batch_size else False,
        )

    # Validation DataLoader is strictly unweighted, non-replacement, shuffle=False
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    return train_loader, val_loader
