"""
Script: scripts/diagnostic_audit.py
Stage 1A Diagnostic Audit for DACL10K-v2 Segmentation Model
- Audit class frequencies, polygon areas in validation set
- Load best_model.pth checkpoint and inspect per-class probabilities
- Analyze zero-prediction classes at thresholds 0.1, 0.2, 0.3, 0.5
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.mask_converter import DACL10K_CLASSES

ZERO_PRED_CLASSES = {'ACrack', 'Cavity', 'Crack', 'JTape', 'Restformwork', 'Rockpocket', 'WConccor'}
ANNO_DIR = Path('/Users/kundankumar/Downloads/dacl10k_v2_devphase/annotations/validation')
CKPT_PATHS = [
    Path('/Users/kundankumar/Documents/untitled folder 2/bridge-ai-project/experiments/segmentation_baseline/checkpoints/best_model.pth'),
    Path('/content/drive/MyDrive/bridge_ai_experiments/segmentation_baseline/checkpoints/best_model.pth'),
]
DATASET_ROOT = Path('/Users/kundankumar/Downloads/dacl10k_v2_devphase')


# =====================================================================
# STEP 1: Class frequency audit (all 975 val annotations)
# =====================================================================
print("=" * 70)
print("STEP 1: CLASS FREQUENCY AUDIT ON VALIDATION SET (ALL 975 IMAGES)")
print("=" * 70)
anno_files = sorted(list(ANNO_DIR.glob('*.json')))
class_img_count = {c: 0 for c in DACL10K_CLASSES}
class_polygon_areas = {c: [] for c in ZERO_PRED_CLASSES}
total_scanned = 0

for jf in anno_files:
    with open(jf) as f:
        data = json.load(f)
    shapes = data.get('shapes', [])
    img_h = data.get('imageHeight', 1)
    img_w = data.get('imageWidth', 1)
    seen = set()
    for s in shapes:
        lbl = s.get('label', '')
        if lbl in DACL10K_CLASSES:
            seen.add(lbl)
        if lbl in ZERO_PRED_CLASSES:
            pts = s.get('points', [])
            if len(pts) >= 3:
                arr = np.array(pts, dtype=np.float32)
                x, y = arr[:, 0], arr[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                pct = area / (img_h * img_w) * 100
                class_polygon_areas[lbl].append(pct)
    for lbl in seen:
        class_img_count[lbl] += 1
    total_scanned += 1

print(f"Total val images scanned: {total_scanned}")
print()
hdr = f"{'Class':<15} | {'Images':>7} | {'% of val':>8}"
print(hdr)
print('-' * len(hdr))
for c in DACL10K_CLASSES:
    marker = '  *** ZERO-PRED' if c in ZERO_PRED_CLASSES else ''
    print(f"{c:<15} | {class_img_count[c]:>7} | {class_img_count[c]/total_scanned*100:>7.1f}%{marker}")

print()
print("Polygon area stats for zero-pred classes (% of native image resolution, before resize):")
hdr2 = f"{'Class':<15} | {'Polygons':>8} | {'Mean%':>7} | {'Median%':>8} | {'Min%':>8} | {'Max%':>7}"
print(hdr2)
print('-' * len(hdr2))
for c in ZERO_PRED_CLASSES:
    areas = class_polygon_areas[c]
    if areas:
        print(f"{c:<15} | {len(areas):>8} | {np.mean(areas):>7.3f} | {np.median(areas):>8.4f} | {np.min(areas):>8.5f} | {np.max(areas):>7.2f}")
    else:
        print(f"{c:<15} | {'0':>8} | {'N/A':>7} | {'N/A':>8} | {'N/A':>8} | {'N/A':>7}")


# =====================================================================
# STEP 2: Try to load checkpoint
# =====================================================================
print()
print("=" * 70)
print("STEP 2: CHECKPOINT INSPECTION")
print("=" * 70)
ckpt_path = None
for p in CKPT_PATHS:
    if p.exists():
        ckpt_path = p
        break

if ckpt_path is None:
    print("[SKIP] No checkpoint found at expected paths:")
    for p in CKPT_PATHS:
        print(f"  {p}")
    print("  Place best_model.pth at experiments/segmentation_baseline/checkpoints/best_model.pth")
    print("  to enable Steps 2–4 of the diagnostic.")
    sys.exit(0)

print(f"Found checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')
ckpt_keys = list(ckpt.keys())
print(f"Checkpoint top-level keys: {ckpt_keys}")
if 'epoch' in ckpt:
    print(f"Saved from epoch: {ckpt['epoch']}")
if 'val_miou' in ckpt:
    print(f"Recorded val mIoU: {ckpt['val_miou']:.4f}")
if 'val_dice' in ckpt:
    print(f"Recorded val Dice: {ckpt['val_dice']:.4f}")

# Checkpoint from Colab training saves with key 'model_state_dict'
# (see train.py checkpoint saving logic)
model_state = (ckpt.get('model_state_dict')
               or ckpt.get('model_state')
               or ckpt.get('state_dict'))
if model_state is None:
    # Checkpoint IS the state dict (legacy format)
    model_state = ckpt
print(f"Model state dict key used: {[k for k in ckpt.keys() if 'model' in k.lower() or 'state' in k.lower()]}")
if isinstance(model_state, dict):
    out_keys = [k for k in model_state.keys() if 'segmentation_head' in k.lower() or 'final_conv' in k.lower() or 'head' in k.lower()]
    print(f"\nOutput layer keys: {out_keys[:8]}")
    for k in out_keys[:4]:
        v = model_state[k]
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

# =====================================================================
# STEP 3: Load model and check per-class probabilities on a few val images
# =====================================================================
print()
print("=" * 70)
print("STEP 3: PER-CLASS PROBABILITY ANALYSIS ON SAMPLE VAL IMAGES")
print("=" * 70)
try:
    from src.segmentation.model import BridgeSegmentationModel
    from src.data.dataset import DACL10KDataset, custom_collate_fn
    from torch.utils.data import DataLoader

    device = torch.device('cpu')
    model = BridgeSegmentationModel(
        architecture='DeepLabV3Plus',
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        num_classes=19,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    print("Model loaded successfully from checkpoint.")

    # Load 8 val images containing zero-pred classes for inspection
    val_ds = DACL10KDataset(
        root_dir=DATASET_ROOT,
        split='validation',
        target_size=(512, 512),
        augment=False,
        return_metadata=True,
    )

    # Find samples that contain any of the zero-pred classes
    sample_indices = []
    for i in range(len(val_ds)):
        anno_path = val_ds.anno_paths[i]
        if anno_path is None:
            continue
        with open(anno_path) as f:
            data = json.load(f)
        labels = {s['label'] for s in data.get('shapes', [])}
        if labels & ZERO_PRED_CLASSES:
            sample_indices.append(i)
        if len(sample_indices) >= 16:
            break

    print(f"\nFound {len(sample_indices)} val images containing zero-pred classes.")
    print("Inspecting per-class probabilities at thresholds [0.1, 0.2, 0.3, 0.5]:")

    # Per-class accumulators
    cls_max_prob  = {c: [] for c in DACL10K_CLASSES}
    cls_mean_prob = {c: [] for c in DACL10K_CLASSES}
    cls_pos_pixels = {tau: {c: 0 for c in DACL10K_CLASSES} for tau in [0.1, 0.2, 0.3, 0.5]}

    with torch.no_grad():
        for idx in sample_indices:
            img_t, mask_t, meta = val_ds[idx]
            img_b = img_t.unsqueeze(0).to(device)
            logits = model(img_b)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (19, 512, 512)

            for ci, cname in enumerate(DACL10K_CLASSES):
                ch = probs[ci]
                cls_max_prob[cname].append(float(ch.max()))
                cls_mean_prob[cname].append(float(ch.mean()))
                for tau in [0.1, 0.2, 0.3, 0.5]:
                    cls_pos_pixels[tau][cname] += int((ch >= tau).sum())

    print()
    hdr3 = f"{'Class':<15} | {'MaxProb(med)':>12} | {'MeanProb(med)':>13} | {'Pos@0.1':>8} | {'Pos@0.2':>8} | {'Pos@0.3':>8} | {'Pos@0.5':>8}"
    print(hdr3)
    print('-' * len(hdr3))
    for cname in DACL10K_CLASSES:
        max_p_med = float(np.median(cls_max_prob[cname])) if cls_max_prob[cname] else 0.0
        mean_p_med = float(np.median(cls_mean_prob[cname])) if cls_mean_prob[cname] else 0.0
        pos01 = cls_pos_pixels[0.1][cname]
        pos02 = cls_pos_pixels[0.2][cname]
        pos03 = cls_pos_pixels[0.3][cname]
        pos05 = cls_pos_pixels[0.5][cname]
        marker = '  ***' if cname in ZERO_PRED_CLASSES else ''
        print(f"{cname:<15} | {max_p_med:>12.4f} | {mean_p_med:>13.5f} | {pos01:>8,} | {pos02:>8,} | {pos03:>8,} | {pos05:>8,}{marker}")

    # =====================================================================
    # STEP 4: Check class presence in actual GT masks from these samples
    # =====================================================================
    print()
    print("=" * 70)
    print("STEP 4: GT MASK PRESENCE CHECK IN SAMPLE IMAGES")
    print("=" * 70)
    print(f"{'Class':<15} | {'GT imgs with class':>18} | {'Max GT Pixels':>15}")
    print('-' * 55)
    cls_gt_imgs = {c: 0 for c in DACL10K_CLASSES}
    cls_gt_max_px = {c: 0 for c in DACL10K_CLASSES}
    for idx in sample_indices:
        img_t, mask_t, meta = val_ds[idx]
        mask_np = mask_t.numpy()  # (19, 512, 512)
        for ci, cname in enumerate(DACL10K_CLASSES):
            px = int(mask_np[ci].sum())
            if px > 0:
                cls_gt_imgs[cname] += 1
                cls_gt_max_px[cname] = max(cls_gt_max_px[cname], px)

    for cname in DACL10K_CLASSES:
        marker = '  ***' if cname in ZERO_PRED_CLASSES else ''
        print(f"{cname:<15} | {cls_gt_imgs[cname]:>18} | {cls_gt_max_px[cname]:>15,}{marker}")

except Exception as e:
    import traceback
    print(f"[ERROR] {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("DIAGNOSTIC AUDIT COMPLETE")
print("=" * 70)
