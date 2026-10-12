"""
Script: scripts/generate_colab_notebook.py
Description:
    Generates notebooks/DACL10K_Colab_Training.ipynb with complete structure,
    markdown explanations, and executable PyTorch code for Google Colab execution.
"""

import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
notebook_path = repo_root / "notebooks" / "DACL10K_Colab_Training.ipynb"

cells = []


def add_markdown(source: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })


def add_code(source: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })


# Title & Header
add_markdown("""# Bridge Defect Segmentation — Stage 1 Full Training on DACL10K-v2
### Explainable AI-Based Bridge Defect Severity Assessment and Risk-Based Maintenance Prioritization

**Model:** DeepLabV3+ with ResNet-34 Encoder (ImageNet Pretrained)  
**Output:** 19 Independent Multi-Label Binary Segmentation Channels  
**Resolution:** $512 \\times 512$ Aspect-Ratio Preserving Letterbox  
**Environment:** Google Colab (NVIDIA T4 / V100 / A100 GPU)

---
> **Important Scientific Note:**  
> The earlier Stage 1A experiment evaluated overfit capacity on a 16-sample training diagnostic subset.  
> This notebook executes full supervised baseline training on the complete **DACL10K-v2 training split (6,935 images)** and evaluates generalization strictly on the **validation split (975 images)**.
""")

# Section 1
add_markdown("""## Section 1: Environment and GPU Verification
This section inspects the runtime environment, verifies PyTorch versions, detects available hardware accelerators, and enforces that CUDA is active.""")

add_code("""import os
import sys
import platform
import torch

print("=" * 60)
print("ENVIRONMENT & HARDWARE ACCELERATOR VERIFICATION")
print("=" * 60)
print(f"Python Version:   {platform.python_version()}")
print(f"PyTorch Version:  {torch.__version__}")
print(f"OS Platform:      {platform.platform()}")

# Verify CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA Available:   {cuda_available}")

if not cuda_available:
    raise RuntimeError(
        "\\n[CRITICAL ERROR] No GPU accelerator detected!\\n"
        "Please enable a GPU runtime before running this notebook:\\n"
        "  1. Go to top menu: Runtime -> Change runtime type\\n"
        "  2. Under 'Hardware accelerator', select 'T4 GPU' or 'A100 GPU'\\n"
        "  3. Click 'Save' and re-run this cell."
    )

gpu_count = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0)
gpu_props = torch.cuda.get_device_properties(0)
total_vram_gb = gpu_props.total_memory / (1024 ** 3)
cuda_capability = f"{gpu_props.major}.{gpu_props.minor}"

print(f"GPU Device:       {gpu_name} (x{gpu_count})")
print(f"Compute Cap:      {cuda_capability}")
print(f"Total VRAM:       {total_vram_gb:.2f} GB")
print("=" * 60)
print("GPU accelerator verified successfully!")""")

add_markdown("""## Section 2: Repository & Code Setup
Configure your project code in Google Colab.

**Supports 3 flexible options:**
1. **Option A (Google Drive — Easiest, No GitHub required):** Place `bridge-ai-project-code.zip` or the `bridge-ai-project` folder in your Google Drive root (`MyDrive/`).
2. **Option B (Direct Upload):** Drag and drop `bridge-ai-project-code.zip` into Colab's left-sidebar Files panel.
3. **Option C (GitHub Clone):** Set `REPO_URL` to your GitHub repository URL.""")

add_code("""# ==============================================================================
# SECTION 2: REPOSITORY & CODE SETUP
# Supports Google Drive, direct zip upload, or GitHub clone
# ==============================================================================
import os
import sys
import shutil
from pathlib import Path

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

PROJECT_DIR = "/content/bridge-ai-project"
REPO_URL = "https://github.com/your-username/bridge-ai-project.git" # Update if using GitHub

code_found = False

# --- OPTION A: Check if code exists on Google Drive ---
drive_zip_path = "/content/drive/MyDrive/bridge-ai-project-code.zip"
drive_project_dir = "/content/drive/MyDrive/bridge-ai-project"

if os.path.exists(drive_zip_path):
    print(f"Found code zip on Google Drive: {drive_zip_path}")
    print("Unpacking to /content/bridge-ai-project...")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    !unzip -q -o {drive_zip_path} -d {PROJECT_DIR}
    code_found = True
elif os.path.exists(drive_project_dir) and os.path.exists(f"{drive_project_dir}/src/data/dataset.py"):
    print(f"Found code directory on Google Drive: {drive_project_dir}")
    print("Syncing to /content/bridge-ai-project...")
    !rsync -avq --exclude='experiments' --exclude='outputs' {drive_project_dir}/ {PROJECT_DIR}/
    code_found = True

# --- OPTION B: Check if zip was uploaded directly to /content ---
colab_zip_path = "/content/bridge-ai-project-code.zip"
if not code_found and os.path.exists(colab_zip_path):
    print(f"Found uploaded zip at {colab_zip_path}. Unpacking...")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    !unzip -q -o {colab_zip_path} -d {PROJECT_DIR}
    code_found = True

# --- OPTION C: Clone from GitHub ---
if not code_found:
    if "your-username" not in REPO_URL and "github.com" in REPO_URL:
        print(f"Cloning from GitHub repository: {REPO_URL}...")
        !git clone {REPO_URL} {PROJECT_DIR}
        if os.path.exists(f"{PROJECT_DIR}/src/data/dataset.py"):
            code_found = True
    else:
        print("\\n" + "=" * 70)
        print("[SETUP REQUIRED] How to provide your project code to Colab:")
        print("Choose ANY of these 3 easy methods:")
        print("  Method 1: Upload 'bridge-ai-project-code.zip' to your Google Drive root (MyDrive/)")
        print("  Method 2: Drag & drop 'bridge-ai-project-code.zip' into Colab's left sidebar file panel")
        print("  Method 3: Set REPO_URL to your GitHub repository address and re-run")
        print("=" * 70 + "\\n")

# Add repository root to Python path
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Install required dependencies
print("\\nInstalling project dependencies...")
!pip install -q segmentation-models-pytorch albumentations pyyaml opencv-python-headless tqdm pandas matplotlib

# Verify required core files exist
required_files = [
    f"{PROJECT_DIR}/src/data/dataset.py",
    f"{PROJECT_DIR}/src/data/mask_converter.py",
    f"{PROJECT_DIR}/src/segmentation/model.py",
    f"{PROJECT_DIR}/src/segmentation/losses.py",
    f"{PROJECT_DIR}/src/segmentation/evaluate.py",
    f"{PROJECT_DIR}/configs/training.yaml",
]

for rf in required_files:
    assert os.path.exists(rf), (
        f"Missing required file: {rf}\\n"
        f"Please upload 'bridge-ai-project-code.zip' to Colab or Google Drive!"
    )

print("\\nAll repository files and dependencies verified successfully!")""")

# Section 3
add_markdown("""## Section 3: Dataset Setup & Verification
Mount Google Drive and verify the DACL10K-v2 dataset split locations.  
**Strict Safety Rule:** The dataset is never uploaded to GitHub or hardcoded to Mac paths.""")

add_code("""# ==============================================================================
# CONFIGURABLE DATASET PATHS
# ==============================================================================
# Option A: Dataset already unzipped in Google Drive
# Example: "/content/drive/MyDrive/dacl10k_v2_devphase"
DRIVE_DATASET_DIR = "/content/drive/MyDrive/dacl10k_v2_devphase"

# Option B (Recommended for High I/O Speed): Fast local SSD path
# If your dataset is a zip file in Drive, unzip to /content/dataset for 10x faster loading!
USE_LOCAL_FAST_STORAGE = False
ZIP_FILE_PATH_ON_DRIVE = "/content/drive/MyDrive/dacl10k_v2_devphase.zip"
LOCAL_DATASET_DIR = "/content/dataset/dacl10k_v2_devphase"

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

if USE_LOCAL_FAST_STORAGE:
    if not os.path.exists(LOCAL_DATASET_DIR):
        print(f"Extracting dataset from {ZIP_FILE_PATH_ON_DRIVE} to local fast SSD...")
        os.makedirs("/content/dataset", exist_ok=True)
        !unzip -q -o {ZIP_FILE_PATH_ON_DRIVE} -d /content/dataset
    DATASET_ROOT = LOCAL_DATASET_DIR
else:
    DATASET_ROOT = DRIVE_DATASET_DIR

print(f"\\nUsing Dataset Root: {DATASET_ROOT}")

# Verify file counts
from pathlib import Path
train_img_dir = Path(DATASET_ROOT) / "images/train"
train_anno_dir = Path(DATASET_ROOT) / "annotations/train"
val_img_dir = Path(DATASET_ROOT) / "images/validation"
val_anno_dir = Path(DATASET_ROOT) / "annotations/validation"

assert train_img_dir.exists(), f"Train image directory not found: {train_img_dir}"
assert train_anno_dir.exists(), f"Train annotation directory not found: {train_anno_dir}"
assert val_img_dir.exists(), f"Val image directory not found: {val_img_dir}"
assert val_anno_dir.exists(), f"Val annotation directory not found: {val_anno_dir}"

train_imgs = list(train_img_dir.glob("*.jpg"))
train_annos = list(train_anno_dir.glob("*.json"))
val_imgs = list(val_img_dir.glob("*.jpg"))
val_annos = list(val_anno_dir.glob("*.json"))

print("-" * 50)
print(f"Train Images:       {len(train_imgs)} (Expected: 6,935)")
print(f"Train Annotations:  {len(train_annos)} (Expected: 6,935)")
print(f"Val Images:         {len(val_imgs)} (Expected: 975)")
print(f"Val Annotations:    {len(val_annos)} (Expected: 975)")
print("-" * 50)

if len(train_imgs) == 6935 and len(val_imgs) == 975:
    print("Dataset verification PASSED with exact expected image counts!")
else:
    print(f"[WARNING] Detected {len(train_imgs)} train and {len(val_imgs)} val images. Verify that dataset unzipping is complete.")""")

# Section 4
add_markdown("""## Section 4: Configuration & Hyperparameters
Inspect and customize the Colab training configuration. All 19 classes are verified for consistent alphabetical indexing.""")

add_code("""import yaml

# ==============================================================================
# CONFIGURABLE TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 8          # Recommended: 8 for T4 (16GB), 16 for A100 (40GB)
IMAGE_SIZE = (512, 512) # Height, Width letterbox resize
LEARNING_RATE = 0.0001  # AdamW base learning rate
NUM_WORKERS = 4         # Parallel DataLoader workers
TOTAL_EPOCHS = 25       # Full training baseline duration
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/bridge_ai_experiments/segmentation_baseline"

os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{DRIVE_OUTPUT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{DRIVE_OUTPUT_DIR}/visualizations", exist_ok=True)

# Load base config
config_path = f"{PROJECT_DIR}/configs/colab_training.yaml"
if not os.path.exists(config_path):
    config_path = f"{PROJECT_DIR}/configs/training.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Update config with active Colab settings
config["dataset"]["root_dir"] = DATASET_ROOT
config["training"]["batch_size"] = BATCH_SIZE
config["training"]["image_size"] = list(IMAGE_SIZE)
config["training"]["learning_rate"] = LEARNING_RATE
config["training"]["num_workers"] = NUM_WORKERS
config["training"]["epochs"] = TOTAL_EPOCHS
config["training"]["device"] = "cuda"
config["experiment"]["output_dir"] = DRIVE_OUTPUT_DIR

# Verify 19 canonical DACL10K classes
from src.data.mask_converter import DACL10K_CLASSES
assert len(config["dataset"]["classes"]) == 19
assert config["dataset"]["classes"] == list(DACL10K_CLASSES)

print("Active Colab Configuration:")
print(f"  Device:         {config['training']['device']}")
print(f"  Batch Size:     {config['training']['batch_size']}")
print(f"  Image Size:     {config['training']['image_size']}")
print(f"  Learning Rate:  {config['training']['learning_rate']}")
print(f"  Total Epochs:   {config['training']['epochs']}")
print(f"  Output Dir:     {config['experiment']['output_dir']}")
print(f"  Class Count:    {len(DACL10K_CLASSES)} independent multi-label channels")""")

# Section 5
add_markdown("""## Section 5: Training Preflight Diagnostics
Verifies data loading, tensor dimensions, strictly binary mask values $\{0, 1\}$, forward pass through DeepLabV3+, loss calculation, and checkpoint saving.""")

add_code("""import torch
from torch.utils.data import DataLoader
from src.data.dataset import DACL10KDataset, custom_collate_fn
from src.segmentation.model import BridgeSegmentationModel
from src.segmentation.losses import CombinedBCEDiceLoss

print("=== RUNNING PREFLIGHT DIAGNOSTICS ===")

# 1. Test Dataset Loader
preflight_ds = DACL10KDataset(
    root_dir=DATASET_ROOT,
    split="train",
    target_size=IMAGE_SIZE,
    augment=False,
    max_samples=4
)
preflight_loader = DataLoader(
    preflight_ds,
    batch_size=4,
    shuffle=False,
    collate_fn=custom_collate_fn
)

images, masks = next(iter(preflight_loader))
print(f"1. Image Batch Shape: {tuple(images.shape)} | dtype: {images.dtype}")
print(f"2. Mask Batch Shape:  {tuple(masks.shape)} | dtype: {masks.dtype}")

assert images.shape == (4, 3, 512, 512), f"Unexpected image shape: {images.shape}"
assert masks.shape == (4, 19, 512, 512), f"Unexpected mask shape: {masks.shape}"

# Verify strictly binary masks {0, 1}
unique_vals = torch.unique(masks).tolist()
print(f"3. Mask Unique Values: {unique_vals}")
assert set(unique_vals).issubset({0.0, 1.0}), f"Masks contain non-binary values: {unique_vals}"

# 4. Test Forward Pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BridgeSegmentationModel(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    num_classes=19
).to(device)

images_cuda = images.to(device)
masks_cuda = masks.to(device)

model.eval()
with torch.no_grad():
    logits = model(images_cuda)

print(f"4. Output Logits Shape: {tuple(logits.shape)} | Expected: (4, 19, 512, 512)")
assert logits.shape == (4, 19, 512, 512), f"Unexpected logits shape: {logits.shape}"

# 5. Test Loss Function
criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
loss = criterion(logits, masks_cuda)
print(f"5. Computed Preflight Loss: {loss.item():.4f}")
assert torch.isfinite(loss), "Preflight loss is NaN or infinite!"

# 6. Test Checkpoint Writing Permissions
test_ckpt_path = f"{DRIVE_OUTPUT_DIR}/checkpoints/preflight_test.pth"
torch.save({"model_state": model.state_dict(), "test": True}, test_ckpt_path)
assert os.path.exists(test_ckpt_path), "Failed to save checkpoint to Drive!"
os.remove(test_ckpt_path)
print("6. Google Drive write permissions confirmed!")
print("\\nPREFLIGHT CHECKS PASSED WITH FLYING COLORS!")""")

# Section 6
add_markdown("""## Section 6: Short Timing Benchmark
Benchmarks 10 real training batches to measure throughput (samples/sec) and estimate full-dataset epoch duration.  
> **Notice:** The estimated epoch duration is an extrapolation and will vary with disk I/O and GPU thermal throttling.""")

add_code("""import time
from tqdm import tqdm

print("=== BENCHMARKING BATCH THROUGHPUT ON CUDA ===")

bench_ds = DACL10KDataset(
    root_dir=DATASET_ROOT,
    split="train",
    target_size=IMAGE_SIZE,
    augment=True,
    max_samples=BATCH_SIZE * 10
)
bench_loader = DataLoader(
    bench_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=custom_collate_fn,
    pin_memory=True
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
model.train()

# Warmup batch
images_b, masks_b = next(iter(bench_loader))
images_b, masks_b = images_b.to(device, non_blocking=True), masks_b.to(device, non_blocking=True)
optimizer.zero_grad()
out = model(images_b)
loss = criterion(out, masks_b)
loss.backward()
optimizer.step()
torch.cuda.synchronize()

# Benchmark 10 batches
num_bench_batches = 10
start_time = time.time()

for idx, (images_b, masks_b) in enumerate(tqdm(bench_loader, total=num_bench_batches, desc="Benchmarking")):
    if idx >= num_bench_batches:
        break
    images_b = images_b.to(device, non_blocking=True)
    masks_b = masks_b.to(device, non_blocking=True)
    
    optimizer.zero_grad()
    out = model(images_b)
    loss = criterion(out, masks_b)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

total_bench_time = time.time() - start_time
total_samples_bench = num_bench_batches * BATCH_SIZE
samples_per_sec = total_samples_bench / total_bench_time
sec_per_batch = total_bench_time / num_bench_batches

# DACL10K-v2 full training set: 6,935 images
total_train_samples = 6935
est_sec_per_epoch = total_train_samples / samples_per_sec
est_min_per_epoch = est_sec_per_epoch / 60.0

print("\\n" + "=" * 60)
print("BENCHMARK RESULTS (EXTRAPOLATION ESTIMATE)")
print("=" * 60)
print(f"Measured Throughput:       {samples_per_sec:.2f} samples/second")
print(f"Average Batch Duration:    {sec_per_batch:.3f} seconds (batch size {BATCH_SIZE})")
print(f"Estimated Epoch Duration:  ~{est_min_per_epoch:.1f} minutes ({est_sec_per_epoch:.0f}s)")
print(f"Estimated {TOTAL_EPOCHS}-Epoch Run:     ~{(est_min_per_epoch * TOTAL_EPOCHS) / 60.0:.1f} hours")
print("=" * 60)
print("NOTE: This estimate reflects GPU compute throughput. Validation evaluation")
print("and Google Drive I/O may add slight variance.")""")

# Section 7
add_markdown("""## Section 7: Full-Dataset Training Loop
Executes full supervised baseline training on `dacl10k_v2_train` (6,935 images) and validates on `dacl10k_v2_validation` (975 images).  
- **Resumption Support:** Automatically resumes from `latest_checkpoint.pth` if a previous session was interrupted.  
- **Best Model Checkpoint:** Saved as `best_model.pth` based on validation mIoU.  
- **Safety Switch:** Set `EXECUTE_TRAINING = True` to launch full training.""")

add_code("""# ==============================================================================
# SAFETY SWITCH: Set to True when you are ready to start training
# ==============================================================================
EXECUTE_TRAINING = False

if not EXECUTE_TRAINING:
    print("[SAFETY GUARD] EXECUTE_TRAINING is set to False.")
    print("Review your configuration above, then set EXECUTE_TRAINING = True to start.")
else:
    import json
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from src.segmentation.evaluate import SegmentationEvaluator

    print(f"Starting Full Baseline Training ({TOTAL_EPOCHS} Epochs)...")
    
    # 1. Datasets & Loaders
    train_dataset = DACL10KDataset(
        root_dir=DATASET_ROOT,
        split="train",
        target_size=IMAGE_SIZE,
        augment=True
    )
    val_dataset = DACL10KDataset(
        root_dir=DATASET_ROOT,
        split="validation",
        target_size=IMAGE_SIZE,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # 2. Model, Loss, Optimizer, Scheduler
    model = BridgeSegmentationModel(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=19
    ).to(device)
    
    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_miou": [],
        "val_dice": [],
        "epoch_time": []
    }
    
    start_epoch = 1
    best_val_miou = -1.0
    
    # Check for existing checkpoint to resume
    resume_path = f"{DRIVE_OUTPUT_DIR}/checkpoints/latest_checkpoint.pth"
    history_path = f"{DRIVE_OUTPUT_DIR}/training_history.json"
    
    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_miou = checkpoint.get("best_val_miou", -1.0)
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        print(f"Resumed at Epoch {start_epoch}/{TOTAL_EPOCHS} with Best Val mIoU = {best_val_miou:.4f}")
        
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        epoch_start = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss_acc = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{TOTAL_EPOCHS:02d} [Train]")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_acc += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = train_loss_acc / max(1, train_steps)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss_acc = 0.0
        val_steps = 0
        evaluator = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=0.5)
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{TOTAL_EPOCHS:02d} [Val]"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss_acc += loss.item()
                val_steps += 1
                
                # Update metrics with sigmoid probabilities applied
                evaluator.update(logits, masks, is_logits=True)
                
        avg_val_loss = val_loss_acc / max(1, val_steps)
        val_metrics = evaluator.compute()
        val_miou = val_metrics["macro"]["mIoU_all_classes"]
        val_dice = val_metrics["macro"]["macro_dice_all_classes"]
        
        epoch_dur = time.time() - epoch_start
        
        # Record history
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_miou"].append(round(val_miou, 4))
        history["val_dice"].append(round(val_dice, 4))
        history["epoch_time"].append(round(epoch_dur, 2))
        
        print(f"\\nEpoch {epoch:02d}/{TOTAL_EPOCHS:02d} Summary | Duration: {epoch_dur:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Val mIoU:   {val_miou:.4f} | Val Dice: {val_dice:.4f}")
        
        # Save latest resumable checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_miou": best_val_miou,
            "config": config,
        }, f"{DRIVE_OUTPUT_DIR}/checkpoints/latest_checkpoint.pth")
        
        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_miou": val_miou,
                "val_dice": val_dice,
                "val_metrics": val_metrics,
                "config": config,
            }, f"{DRIVE_OUTPUT_DIR}/checkpoints/best_model.pth")
            print(f"  --> NEW BEST MODEL SAVED! (Val mIoU: {best_val_miou:.4f})")
            
        with open(f"{DRIVE_OUTPUT_DIR}/training_history.json", "w") as f:
            json.dump(history, f, indent=2)
            
    print("\\n=======================================================")
    print("TRAINING RUN COMPLETE!")
    print(f"Best Validation mIoU: {best_val_miou:.4f}")
    print(f"Artifacts saved in:   {DRIVE_OUTPUT_DIR}")
    print("=======================================================")""")

# Section 8
add_markdown("""## Section 8: Validation Evaluation and Qualitative Visualizations
Evaluates the best saved checkpoint on the held-out validation set (975 images), generates per-class metrics tables, and renders side-by-side 4-panel visual comparisons:
1. Original letterboxed bridge photograph
2. Ground-truth multi-label mask overlay
3. Predicted multi-label mask ($\tau = 0.5$)
4. Translucent prediction overlay on image""")

add_code("""import pandas as pd
import matplotlib.pyplot as plt
from src.segmentation.evaluate import save_qualitative_comparison

best_ckpt_path = f"{DRIVE_OUTPUT_DIR}/checkpoints/best_model.pth"

if not os.path.exists(best_ckpt_path):
    print(f"[NOTE] Best checkpoint not found at {best_ckpt_path}. Run Section 7 first.")
else:
    print(f"Loading best checkpoint from {best_ckpt_path}...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    eval_model = BridgeSegmentationModel(encoder_name="resnet34", num_classes=19, weights=None).to(device)
    eval_model.load_state_dict(ckpt["model_state"])
    eval_model.eval()
    
    val_dataset = DACL10KDataset(root_dir=DATASET_ROOT, split="validation", target_size=IMAGE_SIZE, augment=False, return_metadata=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate_fn)
    
    evaluator = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=0.5)
    
    print("Evaluating complete validation split (975 images)...")
    with torch.no_grad():
        for images, masks, metas in tqdm(val_loader, desc="Validation Evaluation"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            sample_ids = [m["image_id"] for m in metas]
            
            logits = eval_model(images)
            evaluator.update(logits, masks, sample_ids=sample_ids, is_logits=True)
            
    val_results = evaluator.compute()
    
    # Save CSV and JSON
    csv_path = f"{DRIVE_OUTPUT_DIR}/val_per_class_metrics.csv"
    json_path = f"{DRIVE_OUTPUT_DIR}/val_summary_metrics.json"
    evaluator.save_results(csv_path=csv_path, json_path=json_path)
    
    print("\\n" + "=" * 70)
    print("HELD-OUT VALIDATION SET PERFORMANCE (DACL10K-v2, 975 IMAGES)")
    print("=" * 70)
    df = pd.DataFrame(val_results["per_class"])
    display_cols = ["class_name", "iou", "dice", "precision", "recall", "gt_pixels", "pred_pixels"]
    print(df[display_cols].to_string(index=False))
    print("-" * 70)
    print(f"Validation mIoU (All 19 Classes):        {val_results['macro']['mIoU_all_classes']:.4f}")
    print(f"Validation mIoU (Present Classes Only):  {val_results['macro']['mIoU_present_classes']:.4f}")
    print(f"Validation Macro Dice:                   {val_results['macro']['macro_dice_all_classes']:.4f}")
    print("=" * 70)
    
    # Render Qualitative Visualizations
    print("\\nGenerating 4-panel visual comparisons on sample validation images...")
    vis_dir = f"{DRIVE_OUTPUT_DIR}/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    for i in range(min(5, len(val_dataset))):
        img_t, mask_t, meta = val_dataset[i]
        with torch.no_grad():
            log_t = eval_model(img_t.unsqueeze(0).to(device))
            prob_t = torch.sigmoid(log_t).squeeze(0).cpu()
            pred_bin = (prob_t >= 0.5).numpy().astype(np.uint8).transpose(1, 2, 0)
            
        save_qualitative_comparison(
            image_tensor=img_t,
            gt_mask=mask_t.numpy().transpose(1, 2, 0),
            pred_mask=pred_bin,
            sample_id=meta["image_id"],
            output_dir=vis_dir,
            class_names=DACL10K_CLASSES
        )
    print(f"Visualizations saved to {vis_dir}!")""")

# Section 9
add_markdown("""## Section 9: Artifact Export & Download Instructions
Verify generated artifacts in Google Drive and download checkpoints or summary reports to your local computer.""")

add_code("""print("=" * 60)
print("TRAINING ARTIFACTS IN GOOGLE DRIVE:")
print("=" * 60)
!ls -lh {DRIVE_OUTPUT_DIR}/checkpoints
!ls -lh {DRIVE_OUTPUT_DIR}

# Download helper for Colab
from google.colab import files

def download_artifact(rel_path):
    full_path = os.path.join(DRIVE_OUTPUT_DIR, rel_path)
    if os.path.exists(full_path):
        print(f"Downloading {full_path}...")
        files.download(full_path)
    else:
        print(f"File not found: {full_path}")

print("\\nTo download the best model checkpoint to your Mac:")
print(">>> download_artifact('checkpoints/best_model.pth')")
print("\\nTo download the validation CSV metrics:")
print(">>> download_artifact('val_per_class_metrics.csv')")""")

# Save notebook
notebook_dict = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"Notebook successfully created at: {notebook_path}")
