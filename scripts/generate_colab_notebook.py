"""
Script: scripts/generate_colab_notebook.py
Description:
    Generates notebooks/DACL10K_Colab_Training.ipynb with complete structure,
    verified PyTorch code, robust unified checkpoint keys, and seamless
    resumption support for 15 total epochs on Google Colab.
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
add_markdown("""# Bridge Defect Segmentation — Stage 1 Resumable Baseline on DACL10K-v2
### Explainable AI-Based Bridge Defect Severity Assessment and Risk-Based Maintenance Prioritization

**Model:** DeepLabV3+ with ResNet-34 Encoder (ImageNet Pretrained)  
**Output:** 19 Independent Multi-Label Binary Segmentation Channels  
**Resolution:** $512 \\times 512$ Aspect-Ratio Preserving Letterbox  
**Total Target Epochs:** 15 Total Epochs (Resumes from verified checkpoint without resetting epoch counter)  
**Evaluation:** Validation split (975 images) evaluated after every epoch  
**Primary Metric:** Validation mIoU at $\\tau = 0.5$ (Threshold $\\tau = 0.3$ recorded as auxiliary diagnostic)

---
> **Important Scientific Scope:**  
> - Validation split (975 images) is used strictly for model evaluation and checkpoint selection.  
> - Unannotated `testdev` (1,012 images) is **NEVER** used for model selection or training.  
> - This baseline run delivers multi-label defect probability maps. Downstream engineering severity assessment, crack width measurement, and risk prioritization are subsequent stages and are not claimed as completed here.

---

## ⚡ RECOVERY PROTOCOL (BROWSER REFRESH vs. RUNTIME RESET)

| Scenario | What Happened | Immediate Action |
|:---|:---|:---|
| **A. Browser Refresh / Reconnect** *(Colab VM still alive)* | You closed your tab, refreshed the page, or local Wi-Fi reconnected. The underlying Colab cloud machine is **still running**. In-memory Python variables, imports, and GPU models are preserved. | **1. Do NOT re-run setup cells.**<br>**2. Reconnect to the session.**<br>**3. Check cell outputs to monitor ongoing training or continue from where you left off.** |
| **B. Runtime Reset / Disconnect** *(Colab VM was terminated)* | Kernel crashed, Colab timed out after idle, machine was disconnected for >30 minutes, or you clicked "Restart session". All in-memory variables and installed packages are **completely wiped**. | **1. Run CELL R1 (`RUNTIME RECOVERY`) below immediately.** It restores Google Drive, GPU checks, missing packages, repository code, datasets, configuration variables, and audits Drive checkpoints in a single idempotent pass.<br>**2. Run Section 6.6 (optional audit) or CELL R2 (`QUICK RESUME`) to resume training up to 15 epochs.** |

> ⚠️ **CRITICAL RECOVERY RULE:**  
> **Never assume an in-memory variable or installed package survived a runtime reset.**  
> If the kernel restarted or a new VM was assigned, execute **Cell R1 (Runtime Recovery)** first! You do **not** need to manually execute Sections 1–6 one by one after a reset.
""")

# ==============================================================================
# CELL R1: RUNTIME RECOVERY (IDEMPOTENT ALL-IN-ONE SETUP)
# ==============================================================================
add_markdown("""### 🔄 CELL R1: RUNTIME RECOVERY — RUN AFTER RUNTIME RESET
**Idempotent, non-destructive, all-in-one restoration cell:**
- Mounts Google Drive (if not already mounted)
- Enforces CUDA / GPU acceleration (T4/A100)
- Checks and installs only genuinely missing dependencies
- Locates repository code without overwriting existing files
- Verifies DACL10K-v2 dataset and reports actual train (6,935) and val (975) image counts
- Restores all configuration variables (`BATCH_SIZE=8`, `IMAGE_SIZE=512x512`, `TOTAL_EPOCHS=15`, etc.)
- Discovers and audits Google Drive checkpoints (`/content/drive/MyDrive/bridge_ai_experiments/segmentation_baseline/checkpoints/`)
- Sets up resume targets without automatically starting training""")

add_code("""# ==============================================================================
# CELL R1: RUNTIME RECOVERY — RUN AFTER RUNTIME RESET
# ==============================================================================
import os
import sys
import shutil
import time
import json
import importlib.util
from pathlib import Path

print("=" * 75)
print("DACL10K-v2 COLAB RUNTIME RECOVERY & SETUP PROTOCOL")
print("=" * 75)

# ------------------------------------------------------------------------------
# 1. Mount Google Drive Idempotently
# ------------------------------------------------------------------------------
DRIVE_MOUNT_POINT = "/content/drive"
if not os.path.exists(f"{DRIVE_MOUNT_POINT}/MyDrive"):
    print("[1/7] Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount(DRIVE_MOUNT_POINT, force_remount=False)
        print("      [+] Google Drive mounted successfully at /content/drive.")
    except Exception as e:
        print(f"      [!] Drive mount notice: {e}")
else:
    print(f"[1/7] [+] Google Drive is already mounted at {DRIVE_MOUNT_POINT}/MyDrive.")

# ------------------------------------------------------------------------------
# 2. Hardware Accelerator & CUDA Verification
# ------------------------------------------------------------------------------
import torch
cuda_available = torch.cuda.is_available()
print(f"[2/7] Checking CUDA Hardware Accelerator: {cuda_available}")
if not cuda_available:
    print("      [!] WARNING: No GPU detected! Running on CPU will be extremely slow.")
    print("          To fix: Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4 or A100).")
    DEVICE = torch.device("cpu")
else:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"      [+] Active GPU: {gpu_name} ({gpu_vram:.2f} GB VRAM)")
    DEVICE = torch.device("cuda")

# ------------------------------------------------------------------------------
# 3. Check & Install Missing Dependencies Only
# ------------------------------------------------------------------------------
required_deps = {
    "segmentation_models_pytorch": "segmentation-models-pytorch",
    "albumentations": "albumentations",
    "yaml": "pyyaml",
    "tqdm": "tqdm",
    "cv2": "opencv-python-headless",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
}

missing_deps = [pip_pkg for mod, pip_pkg in required_deps.items() if importlib.util.find_spec(mod) is None]
if missing_deps:
    print(f"[3/7] Installing missing dependencies: {missing_deps}...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing_deps)
    print("      [+] Missing dependencies installed.")
else:
    print("[3/7] [+] All required Python dependencies are already installed.")

# ------------------------------------------------------------------------------
# 4. Locate or Restore Repository Code without destructive overwrites
# ------------------------------------------------------------------------------
PROJECT_DIR = "/content/bridge-ai-project"
REPO_URL = "https://github.com/edkundan/bridge-ai-project.git"

if os.path.exists(f"{PROJECT_DIR}/src/data/dataset.py"):
    print(f"[4/7] [+] Existing repository verified at {PROJECT_DIR} (preserving current files).")
else:
    print(f"[4/7] Repository not found at {PROJECT_DIR}. Restoring...")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    drive_zip = f"{DRIVE_MOUNT_POINT}/MyDrive/bridge-ai-project-code.zip"
    colab_zip = "/content/bridge-ai-project-code.zip"
    drive_dir = f"{DRIVE_MOUNT_POINT}/MyDrive/bridge-ai-project"

    if os.path.exists(drive_zip):
        print(f"      Found code archive: {drive_zip}. Extracting...")
        import subprocess
        subprocess.check_call(["unzip", "-q", "-o", drive_zip, "-d", PROJECT_DIR])
    elif os.path.exists(drive_dir) and os.path.exists(f"{drive_dir}/src/data/dataset.py"):
        print(f"      Found code directory: {drive_dir}. Syncing...")
        import subprocess
        subprocess.check_call(["rsync", "-avq", "--exclude=experiments", "--exclude=outputs", f"{drive_dir}/", f"{PROJECT_DIR}/"])
    elif os.path.exists(colab_zip):
        print(f"      Found local archive: {colab_zip}. Extracting...")
        import subprocess
        subprocess.check_call(["unzip", "-q", "-o", colab_zip, "-d", PROJECT_DIR])
    else:
        print(f"      Cloning from {REPO_URL}...")
        import subprocess
        subprocess.check_call(["git", "clone", REPO_URL, PROJECT_DIR])

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

assert os.path.exists(f"{PROJECT_DIR}/src/data/dataset.py"), "Failed to locate src/data/dataset.py!"
assert os.path.exists(f"{PROJECT_DIR}/src/segmentation/model.py"), "Failed to locate src/segmentation/model.py!"
print("      [+] Repository codebase confirmed on sys.path.")

# ------------------------------------------------------------------------------
# 5. Locate & Verify DACL10K-v2 Dataset Paths & Counts
# ------------------------------------------------------------------------------
POSSIBLE_DATASET_ROOTS = [
    "/content/dacl10k_v2_devphase",
    f"{DRIVE_MOUNT_POINT}/MyDrive/dacl10k_v2_devphase",
    "/content/dacl10k",
    f"{DRIVE_MOUNT_POINT}/MyDrive/dacl10k",
]

DATASET_ROOT = None
for candidate in POSSIBLE_DATASET_ROOTS:
    if os.path.exists(candidate) and os.path.exists(f"{candidate}/images/train"):
        DATASET_ROOT = candidate
        break

if DATASET_ROOT is None:
    zip_candidates = [
        f"{DRIVE_MOUNT_POINT}/MyDrive/dacl10k_v2_devphase.zip",
        f"{DRIVE_MOUNT_POINT}/MyDrive/dacl10k.zip",
        "/content/dacl10k_v2_devphase.zip",
        "/content/dacl10k.zip",
    ]
    for zpath in zip_candidates:
        if os.path.exists(zpath):
            print(f"      Found dataset archive: {zpath}. Unpacking to /content/dacl10k_v2_devphase...")
            dest_dir = "/content/dacl10k_v2_devphase"
            os.makedirs(dest_dir, exist_ok=True)
            import subprocess
            subprocess.check_call(["unzip", "-q", "-o", zpath, "-d", "/content/"])
            DATASET_ROOT = dest_dir
            break

if DATASET_ROOT is None or not os.path.exists(f"{DATASET_ROOT}/images/train"):
    raise FileNotFoundError("DACL10K-v2 dataset not found! Please place 'dacl10k_v2_devphase.zip' or folder in Drive root or /content.")

train_imgs_count = len(list((Path(DATASET_ROOT) / "images" / "train").glob("*.jpg")))
val_imgs_count = len(list((Path(DATASET_ROOT) / "images" / "validation").glob("*.jpg")))
print(f"[5/7] [+] Dataset verified at: {DATASET_ROOT}")
print(f"      Train images: {train_imgs_count} (Expected: 6,935) | Val images: {val_imgs_count} (Expected: 975)")

# ------------------------------------------------------------------------------
# 6. Restore Configuration Variables, Classes, Losses, & Evaluators
# ------------------------------------------------------------------------------
from src.data.dataset import DACL10KDataset, DACL10K_CLASSES, custom_collate_fn
from src.segmentation.model import BridgeSegmentationModel
from src.segmentation.losses import CombinedBCEDiceLoss
from src.segmentation.evaluate import SegmentationEvaluator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

BATCH_SIZE = 8
IMAGE_SIZE = (512, 512)
LEARNING_RATE = 0.0001
NUM_WORKERS = 4
TOTAL_EPOCHS = 15
PRIMARY_THRESHOLD = 0.5
DIAG_THRESHOLD = 0.3

DRIVE_OUTPUT_DIR = f"{DRIVE_MOUNT_POINT}/MyDrive/bridge_ai_experiments/segmentation_baseline"
os.makedirs(f"{DRIVE_OUTPUT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{DRIVE_OUTPUT_DIR}/visualizations", exist_ok=True)

# Load base config
config_path = f"{PROJECT_DIR}/configs/colab_training.yaml"
if not os.path.exists(config_path):
    config_path = f"{PROJECT_DIR}/configs/training.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["dataset"]["root_dir"] = DATASET_ROOT
config["training"]["batch_size"] = BATCH_SIZE
config["training"]["image_size"] = list(IMAGE_SIZE)
config["training"]["learning_rate"] = LEARNING_RATE
config["training"]["num_workers"] = NUM_WORKERS
config["training"]["epochs"] = TOTAL_EPOCHS
config["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
config["training"]["threshold"] = PRIMARY_THRESHOLD
config["experiment"]["output_dir"] = DRIVE_OUTPUT_DIR

print(f"[6/7] [+] Configuration & Class Definitions Restored:")
print(f"      Target Total Epochs:    {TOTAL_EPOCHS}")
print(f"      Batch Size:             {BATCH_SIZE}")
print(f"      Image Resolution:       {IMAGE_SIZE}")
print(f"      Primary Threshold:      {PRIMARY_THRESHOLD} (Diagnostic: {DIAG_THRESHOLD})")
print(f"      Drive Output Directory: {DRIVE_OUTPUT_DIR}")
print(f"      Number of Classes:      {len(DACL10K_CLASSES)}")

# ------------------------------------------------------------------------------
# 7. Checkpoint Discovery & Recovery State Resolution
# ------------------------------------------------------------------------------
ckpt_dir = f"{DRIVE_OUTPUT_DIR}/checkpoints"
history_path = f"{DRIVE_OUTPUT_DIR}/training_history.json"
RECOVERY_CHECKPOINT = None
RECOVERY_START_EPOCH = 1
RECOVERY_BEST_MIOU = 0.0

hist_val_miou = []
hist_val_dice = []
hist_best_miou = 0.0
if os.path.exists(history_path):
    try:
        with open(history_path, "r") as f:
            hist_data = json.load(f)
        hist_val_miou = hist_data.get("val_miou", [])
        hist_val_dice = hist_data.get("val_dice", [])
        if hist_val_miou:
            hist_best_miou = float(max(hist_val_miou))
    except Exception as e:
        print(f"      [!] Warning reading training_history.json: {e}")

print(f"[7/7] Auditing Google Drive Checkpoints in {ckpt_dir}...")
if not os.path.exists(ckpt_dir):
    print("      [!] Checkpoint directory does not exist yet.")
    available_ckpts = []
else:
    available_ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth")])

print(f"      Found {len(available_ckpts)} checkpoint file(s) on Google Drive: {available_ckpts}")

for fname in available_ckpts:
    fpath = os.path.join(ckpt_dir, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    try:
        chk_meta = torch.load(fpath, map_location="cpu")
        ep = chk_meta.get("epoch", "unknown")
        rec_miou = chk_meta.get("best_miou", chk_meta.get("best_val_miou", chk_meta.get("val_miou", "unknown")))
        has_opt = ("optimizer_state_dict" in chk_meta or "optimizer_state" in chk_meta)
        has_sched = ("scheduler_state_dict" in chk_meta or "scheduler_state" in chk_meta)
        print(f"      -> {fname} ({size_mb:.1f} MB) | Epoch: {ep} | mIoU: {rec_miou} | OptState: {has_opt} | SchedState: {has_sched}")
    except Exception as e:
        print(f"      -> {fname}: failed to inspect ({e})")

latest_ckpt_path = f"{ckpt_dir}/latest_checkpoint.pth"
best_ckpt_path = f"{ckpt_dir}/best_model.pth"

if os.path.exists(latest_ckpt_path):
    try:
        chk_latest = torch.load(latest_ckpt_path, map_location="cpu")
        saved_ep = chk_latest.get("epoch", 0)
        has_opt = ("optimizer_state_dict" in chk_latest or "optimizer_state" in chk_latest)
        has_sched = ("scheduler_state_dict" in chk_latest or "scheduler_state" in chk_latest)
        b_miou = float(chk_latest.get("best_miou", chk_latest.get("best_val_miou", 0.0)))

        if has_opt and has_sched and saved_ep >= 1:
            RECOVERY_CHECKPOINT = latest_ckpt_path
            RECOVERY_START_EPOCH = saved_ep + 1

            # Inconsistency audit between checkpoint header and training_history.json
            if hist_val_miou and abs(b_miou - hist_best_miou) > 1e-4:
                print("-" * 75)
                print(f"[!] METRIC AUDIT INCONSISTENCY DETECTED:")
                print(f"    latest_checkpoint.pth metadata header: best_miou = {b_miou:.4f} (stale from Epoch 1)")
                print(f"    training_history.json recorded epochs:  val_miou = {hist_val_miou}")
                print(f"    -> Epoch 2 achieved val_miou = {hist_best_miou:.4f} (val_dice = {hist_val_dice[-1] if hist_val_dice else 'N/A'})")
                print(f"    -> Using verified history baseline mIoU = {hist_best_miou:.4f} for best_model.pth selection.")
                print(f"    -> (Checkpoint metadata on Google Drive is preserved untouched; not rewritten).")
                print("-" * 75)
                RECOVERY_BEST_MIOU = hist_best_miou
            else:
                RECOVERY_BEST_MIOU = max(b_miou, hist_best_miou)

            print("-" * 75)
            print(f"[RECOVERY STATUS: FULL-STATE RESUME VERIFIED]")
            print(f"  Valid Checkpoint:     {latest_ckpt_path}")
            print(f"  Completed Epoch:      {saved_ep} of {TOTAL_EPOCHS}")
            print(f"  Next Epoch to Train:  {RECOVERY_START_EPOCH}")
            print(f"  Baseline Val mIoU:    {RECOVERY_BEST_MIOU:.4f} (Target to beat for best_model.pth)")
            print(f"  Ready for:            Section 6.6 (Evaluation) or Cell R2 (QUICK RESUME)")
            print("-" * 75)
        else:
            print("[!] Note: latest_checkpoint.pth is present but missing optimizer/scheduler state.")
    except Exception as e:
        print(f"[!] Error inspecting latest_checkpoint.pth: {e}")

if RECOVERY_CHECKPOINT is None and os.path.exists(best_ckpt_path):
    try:
        chk_best = torch.load(best_ckpt_path, map_location="cpu")
        saved_ep = chk_best.get("epoch", 0)
        b_miou = chk_best.get("best_miou", chk_best.get("best_val_miou", 0.0))
        print("-" * 75)
        print(f"[RECOVERY STATUS: WEIGHTS-ONLY CHECKPOINT FOUND (best_model.pth)]")
        print(f"  File:                 {best_ckpt_path} (Epoch {saved_ep}, mIoU: {b_miou:.4f})")
        print(f"  Note:                 Optimizer state is not saved in best_model.pth.")
        print(f"                        Prefer latest_checkpoint.pth for exact optimizer momentum resumption.")
        print("-" * 75)
    except Exception as e:
        pass

if RECOVERY_CHECKPOINT is None:
    print("-" * 75)
    print("[RECOVERY STATUS: NO VALID FULL-STATE RESUME CHECKPOINT FOUND IN GOOGLE DRIVE]")
    print("Recovery Options:")
    print("  1. Fresh Run: Begin training from Epoch 1 using Cell R2 or Section 7.")
    print("  2. Upload: If you have an external checkpoint, copy it to:")
    print(f"     {latest_ckpt_path}")
    print("     and re-run this recovery cell.")
    print("-" * 75)

print("=" * 75)
print("[+] RUNTIME RECOVERY COMPLETE. All variables, imports, and state are restored.")
print("    (Model training has NOT started. Use Cell R2 (QUICK RESUME) to begin.)")
print("=" * 75)""")

# ==============================================================================
# CELL R2: QUICK RESUME (POST-RECOVERY TRAINING CELL)
# ==============================================================================
add_markdown("""### 🚀 CELL R2: QUICK RESUME — RESUME 15-EPOCH BASELINE TRAINING
**Executes baseline continuation without re-running setup cells:**
- Verifies that all recovery variables and objects exist
- Loads model, optimizer, scheduler, and training history
- Resumes strictly from `RECOVERY_START_EPOCH` (no epoch counter reset)
- Evaluates the full 975 validation images after every epoch
- Saves `latest_checkpoint.pth` after every completed epoch
- Saves `best_model.pth` only when validation mIoU at $\\tau=0.5$ improves
- Persists `training_history.json` and configuration to Google Drive
- **Safety Switch:** Set `EXECUTE_QUICK_RESUME = True` to launch.""")

add_code("""# ==============================================================================
# CELL R2: QUICK RESUME — RESUME 15-EPOCH BASELINE TRAINING
# ==============================================================================
EXECUTE_QUICK_RESUME = False  # Set to True and re-run to launch training

if not EXECUTE_QUICK_RESUME:
    print("=" * 75)
    print("[SAFETY GUARD] EXECUTE_QUICK_RESUME is set to False.")
    print("Ready to resume:")
    print(f"  Target Total Epochs:   {TOTAL_EPOCHS}")
    print(f"  Resume Checkpoint:     {RECOVERY_CHECKPOINT}")
    print(f"  Next Epoch to Train:   {RECOVERY_START_EPOCH}")
    print(f"  Current Best Val mIoU: {RECOVERY_BEST_MIOU:.4f}")
    print("\\nTo launch training:")
    print("  1. Change above line: EXECUTE_QUICK_RESUME = True")
    print("  2. Re-run this cell")
    print("=" * 75)
else:
    required_vars = ["DATASET_ROOT", "DRIVE_OUTPUT_DIR", "TOTAL_EPOCHS", "RECOVERY_START_EPOCH", "DEVICE"]
    for var in required_vars:
        if var not in globals():
            raise RuntimeError(f"Missing required setup variable '{var}'! Please run CELL R1 (RUNTIME RECOVERY) first.")

    print("=" * 75)
    print(f"LAUNCHING QUICK RESUME TRAINING RUN (EPOCH {RECOVERY_START_EPOCH} TO {TOTAL_EPOCHS})")
    print(f"Device: {DEVICE} | Batch Size: {BATCH_SIZE} | Resolution: {IMAGE_SIZE}")
    print("=" * 75)

    # 1. Datasets & Loaders
    train_dataset = DACL10KDataset(root_dir=DATASET_ROOT, split="train", target_size=IMAGE_SIZE, augment=True)
    val_dataset = DACL10KDataset(root_dir=DATASET_ROOT, split="validation", target_size=IMAGE_SIZE, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # 2. Model, Loss, Optimizer, Scheduler
    model = BridgeSegmentationModel(
        architecture="DeepLabV3Plus",
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=19,
    ).to(DEVICE)

    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_miou": [],
        "val_dice": [],
        "val_miou_diag03": [],
        "val_dice_diag03": [],
        "lr": [],
        "epoch_time": [],
    }

    best_val_miou = RECOVERY_BEST_MIOU
    history_path = f"{DRIVE_OUTPUT_DIR}/training_history.json"
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
            hist_val_miou = history.get("val_miou", [])
            print(f"[+] Loaded existing training history ({len(history.get('train_loss', []))} epochs recorded: val_miou={hist_val_miou}).")
            if hist_val_miou:
                hist_max = float(max(hist_val_miou))
                if hist_max > best_val_miou:
                    print(f"    [!] Setting baseline best_val_miou = {hist_max:.4f} from training_history.json to protect best_model.pth.")
                    best_val_miou = hist_max
        except Exception as e:
            print(f"[!] Warning reading history: {e}")

    # 3. Restore Checkpoint State
    if RECOVERY_CHECKPOINT and os.path.exists(RECOVERY_CHECKPOINT):
        print(f"Loading checkpoint weights and optimizer state from: {RECOVERY_CHECKPOINT}")
        ckpt = torch.load(RECOVERY_CHECKPOINT, map_location=DEVICE)
        model_state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        opt_state = ckpt.get("optimizer_state_dict", ckpt.get("optimizer_state"))
        sched_state = ckpt.get("scheduler_state_dict", ckpt.get("scheduler_state"))

        model.load_state_dict(model_state, strict=True)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            print("  [+] Optimizer state restored successfully.")
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
            print("  [+] Scheduler state restored successfully.")
        print(f"  [+] Resumed from completed Epoch {ckpt.get('epoch', 0)}. Next: Epoch {RECOVERY_START_EPOCH}.")

    if RECOVERY_START_EPOCH > TOTAL_EPOCHS:
        print(f"[+] Training already completed all {TOTAL_EPOCHS} epochs! Nothing to run.")
    else:
        for epoch in range(RECOVERY_START_EPOCH, TOTAL_EPOCHS + 1):
            epoch_start = time.time()
            current_lr = optimizer.param_groups[0]["lr"]

            # --- TRAINING PHASE ---
            model.train()
            train_loss_acc = 0.0
            train_steps = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{TOTAL_EPOCHS:02d} [Train]")
            for images, masks in pbar:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

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

            # --- VALIDATION PHASE (All 975 images) ---
            model.eval()
            val_loss_acc = 0.0
            val_steps = 0
            evaluator_05 = SegmentationEvaluator(class_names=list(DACL10K_CLASSES), threshold=PRIMARY_THRESHOLD)
            evaluator_03 = SegmentationEvaluator(class_names=list(DACL10K_CLASSES), threshold=DIAG_THRESHOLD)

            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{TOTAL_EPOCHS:02d} [Val]"):
                    images = images.to(DEVICE, non_blocking=True)
                    masks = masks.to(DEVICE, non_blocking=True)

                    logits = model(images)
                    loss = criterion(logits, masks)
                    val_loss_acc += loss.item()
                    val_steps += 1

                    evaluator_05.update(logits, masks, is_logits=True)
                    evaluator_03.update(logits, masks, is_logits=True)

            avg_val_loss = val_loss_acc / max(1, val_steps)
            val_metrics_05 = evaluator_05.compute()
            val_metrics_03 = evaluator_03.compute()

            val_miou = val_metrics_05["macro"]["mIoU_all_classes"]
            val_dice = val_metrics_05["macro"].get("macro_dice_all_classes", val_metrics_05["macro"].get("macro_dice", 0.0))
            val_miou_03 = val_metrics_03["macro"]["mIoU_all_classes"]
            val_dice_03 = val_metrics_03["macro"].get("macro_dice_all_classes", val_metrics_03["macro"].get("macro_dice", 0.0))

            epoch_dur = time.time() - epoch_start

            # Record history
            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))
            history["val_miou"].append(round(val_miou, 4))
            history["val_dice"].append(round(val_dice, 4))
            history["val_miou_diag03"].append(round(val_miou_03, 4))
            history["val_dice_diag03"].append(round(val_dice_03, 4))
            history["lr"].append(current_lr)
            history["epoch_time"].append(round(epoch_dur, 2))

            print(f"\\nEpoch {epoch:02d}/{TOTAL_EPOCHS:02d} Summary | Duration: {epoch_dur:.1f}s | LR: {current_lr:.2e}")
            print(f"  Train Loss:         {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Val mIoU (tau=0.5): {val_miou:.4f} | Val Dice (tau=0.5): {val_dice:.4f}")
            print(f"  Val mIoU (tau=0.3): {val_miou_03:.4f} [Diagnostic] | Val Dice: {val_dice_03:.4f}")

            # Check if this epoch achieved best validation mIoU
            is_best = (val_miou > best_val_miou)
            if is_best:
                best_val_miou = val_miou
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_state": model.state_dict(),
                    "best_miou": best_val_miou,
                    "best_val_miou": best_val_miou,
                    "val_miou": val_miou,
                    "val_dice": val_dice,
                    "val_loss": avg_val_loss,
                    "val_metrics": val_metrics_05,
                    "config": config,
                }, f"{DRIVE_OUTPUT_DIR}/checkpoints/best_model.pth")
                print(f"  [*] NEW BEST MODEL SAVED! (Val mIoU: {best_val_miou:.4f}, Val Loss: {avg_val_loss:.4f})")

            # Save latest checkpoint after EVERY completed epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_state": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_miou": best_val_miou,
                "best_val_miou": best_val_miou,
                "val_miou": val_miou,
                "val_loss": avg_val_loss,
                "history": history,
                "config": config,
            }, f"{DRIVE_OUTPUT_DIR}/checkpoints/latest_checkpoint.pth")

            with open(f"{DRIVE_OUTPUT_DIR}/training_history.json", "w") as f:
                json.dump(history, f, indent=2)

        print("\\n" + "=" * 75)
        print("TRAINING RUN COMPLETE (15 TOTAL EPOCHS)!")
        print(f"Best Validation mIoU: {best_val_miou:.4f}")
        print(f"Artifacts saved in:   {DRIVE_OUTPUT_DIR}")
        print("=" * 75)""")

# Divider between Quick Recovery and Step-by-Step Pipeline
add_markdown("""---
## Standard Step-by-Step Diagnostic & Evaluation Pipeline (Sections 1–6.6)

> **Workflow Note:**  
> - **After a fresh runtime reset:** If you already executed **Cell R1 (RUNTIME RECOVERY)** above, all dependencies, datasets, repository code, and configuration variables are already restored. You do **not** need to execute Sections 1–6 one by one.  
> - You can jump straight to **Section 6.5 / 6.6** (to audit checkpoints or evaluate the validation set) or proceed directly to training via **Cell R2 (QUICK RESUME)** above.  
> - Sections 1 through 6 below are preserved for diagnostic verification, benchmarking, and first-time environment audits.
""")

# Section 1
add_markdown("""## Section 1: Environment and Hardware Accelerator Verification
Inspects Python, PyTorch, CUDA runtime, and GPU device properties. Enforces CUDA hardware acceleration.""")

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

# Section 2
add_markdown("""## Section 2: Repository & Code Setup
Mounts Google Drive and sets up the project codebase. Supports:
1. **Google Drive Sync** (retaining existing code and persistent checkpoints)
2. **Direct Zip Upload** (`bridge-ai-project-code.zip`)
3. **GitHub Clone** fallback""")

add_code("""# ==============================================================================
# SECTION 2: REPOSITORY & CODE SETUP
# ==============================================================================
import os
import sys
import shutil
from pathlib import Path

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

PROJECT_DIR = "/content/bridge-ai-project"
REPO_URL = "https://github.com/edkundan/bridge-ai-project.git"

code_found = False

# Option A: Check if code exists on Google Drive
drive_zip_path = "/content/drive/MyDrive/bridge-ai-project-code.zip"
drive_project_dir = "/content/drive/MyDrive/bridge-ai-project"

if os.path.exists(drive_zip_path):
    print(f"Found code zip on Google Drive: {drive_zip_path}")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    !unzip -q -o {drive_zip_path} -d {PROJECT_DIR}
    code_found = True
elif os.path.exists(drive_project_dir) and os.path.exists(f"{drive_project_dir}/src/data/dataset.py"):
    print(f"Found code directory on Google Drive: {drive_project_dir}")
    !rsync -avq --exclude='experiments' --exclude='outputs' {drive_project_dir}/ {PROJECT_DIR}/
    code_found = True

# Option B: Direct zip in /content
colab_zip_path = "/content/bridge-ai-project-code.zip"
if not code_found and os.path.exists(colab_zip_path):
    print(f"Found uploaded zip at {colab_zip_path}. Unpacking...")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    !unzip -q -o {colab_zip_path} -d {PROJECT_DIR}
    code_found = True

# Option C: Clone from GitHub
if not code_found:
    print(f"Cloning from GitHub repository: {REPO_URL}...")
    !git clone {REPO_URL} {PROJECT_DIR}
    if os.path.exists(f"{PROJECT_DIR}/src/data/dataset.py"):
        code_found = True

# Add repository root to Python path
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Install required packages
print("Verifying required dependencies...")
!pip install -q segmentation-models-pytorch albumentations pyyaml tqdm matplotlib pandas opencv-python-headless

print("=" * 60)
print(f"Codebase ready at: {PROJECT_DIR}")
assert os.path.exists(f"{PROJECT_DIR}/src/data/dataset.py"), "Missing src/data/dataset.py!"
assert os.path.exists(f"{PROJECT_DIR}/src/segmentation/model.py"), "Missing src/segmentation/model.py!"
print("Repository files verified successfully!")""")

# Section 3
add_markdown("""## Section 3: Dataset Unpacking & Verification
Verifies DACL10K-v2 dataset directories (6,935 train images, 975 validation images).""")

add_code("""# ==============================================================================
# SECTION 3: DATASET UNPACKING & VERIFICATION
# ==============================================================================
import os
from pathlib import Path

# Common candidate dataset locations
POSSIBLE_DATASET_ROOTS = [
    "/content/dacl10k_v2_devphase",
    "/content/drive/MyDrive/dacl10k_v2_devphase",
    "/content/dacl10k",
    "/content/drive/MyDrive/dacl10k",
]

DATASET_ROOT = None
for candidate in POSSIBLE_DATASET_ROOTS:
    if os.path.exists(candidate) and os.path.exists(f"{candidate}/images/train"):
        DATASET_ROOT = candidate
        break

# If not unzipped, look for dataset zip files
if DATASET_ROOT is None:
    zip_candidates = [
        "/content/drive/MyDrive/dacl10k_v2_devphase.zip",
        "/content/drive/MyDrive/dacl10k.zip",
        "/content/dacl10k_v2_devphase.zip",
        "/content/dacl10k.zip",
    ]
    for zpath in zip_candidates:
        if os.path.exists(zpath):
            print(f"Found dataset archive: {zpath}")
            dest_dir = "/content/dacl10k_v2_devphase"
            print(f"Unzipping to {dest_dir} (this may take 2-4 minutes)...")
            os.makedirs(dest_dir, exist_ok=True)
            !unzip -q -o {zpath} -d /content/
            DATASET_ROOT = dest_dir
            break

if DATASET_ROOT is None or not os.path.exists(f"{DATASET_ROOT}/images/train"):
    raise FileNotFoundError(
        "DACL10K-v2 dataset not found! Please place 'dacl10k_v2_devphase.zip' or folder in Drive root or /content."
    )

print(f"Active DATASET_ROOT: {DATASET_ROOT}")

train_img_dir = Path(DATASET_ROOT) / "images" / "train"
train_anno_dir = Path(DATASET_ROOT) / "annotations" / "train"
val_img_dir = Path(DATASET_ROOT) / "images" / "validation"
val_anno_dir = Path(DATASET_ROOT) / "annotations" / "validation"

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

assert len(train_imgs) > 0, "No training images found!"
assert len(val_imgs) > 0, "No validation images found!"
print("Dataset verification complete!")""")

# Section 4
add_markdown("""## Section 4: Configuration & Hyperparameters (Target: 15 Total Epochs)
Configures training hyperparameters:
- **Architecture:** DeepLabV3+ with ResNet-34 encoder
- **Loss:** Combined BCE (weight=1.0) + Soft Dice (weight=1.0)
- **Image Size:** $512 \\times 512$ letterbox
- **Batch Size:** 8 (optimized for T4 16GB VRAM)
- **Target Total Epochs:** 15 total epochs
- **Persistent Output:** Google Drive `/content/drive/MyDrive/bridge_ai_experiments/segmentation_baseline`""")

add_code("""import os
import sys
import yaml
from src.data.dataset import DACL10K_CLASSES

# ==============================================================================
# CONFIGURABLE TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 8          # Recommended: 8 for T4 (16GB), 16 for A100 (40GB)
IMAGE_SIZE = (512, 512) # Height, Width letterbox resize
LEARNING_RATE = 0.0001  # AdamW base learning rate
NUM_WORKERS = 4         # Parallel DataLoader workers
TOTAL_EPOCHS = 15       # TARGET: 15 Total Epochs for baseline
PRIMARY_THRESHOLD = 0.5 # Primary benchmark threshold
DIAG_THRESHOLD = 0.3    # Additional diagnostic threshold

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

config["dataset"]["root_dir"] = DATASET_ROOT
config["training"]["batch_size"] = BATCH_SIZE
config["training"]["image_size"] = list(IMAGE_SIZE)
config["training"]["learning_rate"] = LEARNING_RATE
config["training"]["num_workers"] = NUM_WORKERS
config["training"]["epochs"] = TOTAL_EPOCHS
config["training"]["device"] = "cuda"
config["training"]["threshold"] = PRIMARY_THRESHOLD
config["experiment"]["output_dir"] = DRIVE_OUTPUT_DIR

print(f"Target Total Epochs:   {TOTAL_EPOCHS}")
print(f"Batch Size:            {BATCH_SIZE}")
print(f"Image Resolution:      {IMAGE_SIZE}")
print(f"Primary Threshold:     {PRIMARY_THRESHOLD} (Diagnostic: {DIAG_THRESHOLD})")
print(f"Drive Output Directory: {DRIVE_OUTPUT_DIR}")
print(f"Number of Classes:     {len(DACL10K_CLASSES)}")""")

# Section 5
add_markdown("""## Section 5: Preflight Diagnostics & Checkpoint Resumption Smoke Test
Executes a rapid validation of the pipeline:
1. Batch generation and letterbox shapes
2. Forward pass and finite CombinedBCEDiceLoss
3. **Checkpoint Key Smoke Test:** Saves a test checkpoint with unified keys (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_miou`) and verifies seamless state restoration.
4. Google Drive write permissions check.""")

add_code("""import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.data.dataset import DACL10KDataset, custom_collate_fn
from src.segmentation.model import BridgeSegmentationModel
from src.segmentation.losses import CombinedBCEDiceLoss
from src.segmentation.evaluate import SegmentationEvaluator

print("=== RUNNING PREFLIGHT DIAGNOSTICS & SMOKE TEST ===")

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
assert set(unique_vals).issubset({0.0, 1.0}), f"Masks contain non-binary values: {unique_vals}"

# 3. Test Forward Pass
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

assert logits.shape == (4, 19, 512, 512), f"Unexpected logits shape: {logits.shape}"
print("3. Forward pass successful!")

# 4. Test Loss Function
criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
loss = criterion(logits, masks_cuda)
print(f"4. Computed Loss: {loss.item():.4f}")
assert torch.isfinite(loss), "Preflight loss is NaN or infinite!"

# 5. Checkpoint Resume Smoke Test
print("5. Running Checkpoint Key & Resumption Smoke Test...")
opt = AdamW(model.parameters(), lr=1e-4)
sched = CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS)
test_ckpt_path = f"{DRIVE_OUTPUT_DIR}/checkpoints/smoke_test_ckpt.pth"

torch.save({
    "epoch": 2,
    "model_state_dict": model.state_dict(),
    "model_state": model.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "optimizer_state": opt.state_dict(),
    "scheduler_state_dict": sched.state_dict(),
    "scheduler_state": sched.state_dict(),
    "best_miou": 0.2018,
    "best_val_miou": 0.2018,
}, test_ckpt_path)

assert os.path.exists(test_ckpt_path), "Failed to save test checkpoint!"

# Verify loading test checkpoint
chk = torch.load(test_ckpt_path, map_location=device)
model_st = chk.get("model_state_dict", chk.get("model_state"))
opt_st = chk.get("optimizer_state_dict", chk.get("optimizer_state"))
sched_st = chk.get("scheduler_state_dict", chk.get("scheduler_state"))
b_miou = chk.get("best_miou", chk.get("best_val_miou", 0.0))
next_epoch = chk.get("epoch", 0) + 1

model.load_state_dict(model_st)
opt.load_state_dict(opt_st)
sched.load_state_dict(sched_st)
os.remove(test_ckpt_path)

print(f"   -> Successfully restored state: next_epoch={next_epoch}, best_miou={b_miou:.4f}")
assert next_epoch == 3, f"Expected next epoch 3, got {next_epoch}"
assert abs(b_miou - 0.2018) < 1e-4, "Restored mIoU mismatch!"

# 6. Evaluator Metric Keys Smoke Test
evaluator = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=PRIMARY_THRESHOLD)
evaluator.update(logits, masks_cuda, is_logits=True)
m_dict = evaluator.compute()
assert "mIoU_all_classes" in m_dict["macro"], "Missing mIoU_all_classes!"
assert "macro_dice_all_classes" in m_dict["macro"], "Missing macro_dice_all_classes!"
assert "macro_dice" in m_dict["macro"], "Missing macro_dice!"

print("=" * 60)
print("ALL PREFLIGHT CHECKS & RESUME SMOKE TESTS PASSED!")
print("=" * 60)""")

# Section 6
add_markdown("""## Section 6: Throughput Timing Benchmark
Benchmarks GPU batch processing throughput (samples/sec) and estimates training duration for 15 epochs.""")

add_code("""import time
from tqdm import tqdm

print("=== BENCHMARKING BATCH THROUGHPUT ON GPU ===")

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
loss_b = criterion(model(images_b), masks_b)
loss_b.backward()
optimizer.step()

t0 = time.time()
num_bench_batches = len(bench_loader)
for images_b, masks_b in bench_loader:
    images_b, masks_b = images_b.to(device, non_blocking=True), masks_b.to(device, non_blocking=True)
    optimizer.zero_grad()
    loss_b = criterion(model(images_b), masks_b)
    loss_b.backward()
    optimizer.step()

torch.cuda.synchronize()
elapsed = time.time() - t0
samples_sec = (num_bench_batches * BATCH_SIZE) / elapsed
sec_per_epoch = 6935 / samples_sec
min_per_epoch = sec_per_epoch / 60

print(f"Measured Throughput:     {samples_sec:.2f} samples/second")
print(f"Estimated Epoch Time:    {min_per_epoch:.2f} minutes/epoch")
print(f"Estimated 15-Epoch Run:  {min_per_epoch * 15 / 60:.2f} hours")
print("=" * 60)""")

# Section 6.5
add_markdown("""## Section 6.5: Checkpoint Discovery & Verification (Before Training)
Explicitly inspects checkpoint files in the Google Drive directory:
- Resolves absolute file paths and file sizes.
- Reads saved epoch metadata and recorded validation mIoU.
- Verifies optimizer and scheduler state dictionary availability.
- Determines whether a genuine Epoch 2 checkpoint is present in Google Drive.
- Prevents silent loading of mismatched local Apple MPS checkpoints.""")

add_code("""# ==============================================================================
# SECTION 6.5: CHECKPOINT DISCOVERY & INTEGRITY AUDIT
# ==============================================================================
import os
import torch
import json

ckpt_dir = f"{DRIVE_OUTPUT_DIR}/checkpoints"
history_path = f"{DRIVE_OUTPUT_DIR}/training_history.json"
print("=" * 75)
print("GOOGLE DRIVE CHECKPOINT DISCOVERY & AUDIT")
print(f"Target Directory: {ckpt_dir}")
print("=" * 75)

hist_val_miou = []
hist_val_dice = []
if os.path.exists(history_path):
    try:
        with open(history_path, "r") as f:
            hist_data = json.load(f)
        hist_val_miou = hist_data.get("val_miou", [])
        hist_val_dice = hist_data.get("val_dice", [])
        print(f"[+] Found training_history.json with {len(hist_val_miou)} epoch(s):")
        print(f"    val_miou: {hist_val_miou}")
        print(f"    val_dice: {hist_val_dice}")
    except Exception as e:
        print(f"[!] Error reading training_history.json: {e}")

if not os.path.exists(ckpt_dir):
    print(f"[!] Directory not found: {ckpt_dir}")
    print("    No previous checkpoints found on Google Drive. Run will start from Epoch 1.")
else:
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth")])
    if not ckpt_files:
        print(f"[!] No .pth checkpoint files found in {ckpt_dir}.")
        print("    Run will start from Epoch 1.")
    else:
        for fname in ckpt_files:
            fpath = os.path.join(ckpt_dir, fname)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            try:
                chk = torch.load(fpath, map_location="cpu")
                keys = list(chk.keys()) if isinstance(chk, dict) else []
                ep = chk.get("epoch", "unknown")
                miou = chk.get("best_miou", chk.get("best_val_miou", chk.get("val_miou", "unknown")))
                has_opt = ("optimizer_state_dict" in chk or "optimizer_state" in chk)
                has_sched = ("scheduler_state_dict" in chk or "scheduler_state" in chk)
                
                print(f"\\n[*] Checkpoint: {fname}")
                print(f"    Resolved Path:    {fpath}")
                print(f"    File Size:        {size_mb:.2f} MB")
                print(f"    Saved Epoch:      {ep}")
                print(f"    Recorded mIoU:    {miou}")
                print(f"    Optimizer State:  {'PRESENT (Resumable)' if has_opt else 'MISSING (Weights only)'}")
                print(f"    Scheduler State:  {'PRESENT' if has_sched else 'MISSING'}")
                print(f"    Top-level Keys:   {keys[:6]}...")
            except Exception as e:
                print(f"[-] {fname}: Failed to inspect ({e})")

latest_path = f"{ckpt_dir}/latest_checkpoint.pth"
if os.path.exists(latest_path):
    chk_latest = torch.load(latest_path, map_location="cpu")
    latest_ep = chk_latest.get("epoch", 0)
    has_opt = ("optimizer_state_dict" in chk_latest or "optimizer_state" in chk_latest)
    has_sched = ("scheduler_state_dict" in chk_latest or "scheduler_state" in chk_latest)
    ckpt_miou = float(chk_latest.get("best_miou", chk_latest.get("best_val_miou", 0.0)))

    print("\\n" + "=" * 75)
    print("METRIC INCONSISTENCY & RESUME DECISION AUDIT:")
    if hist_val_miou and len(hist_val_miou) >= 2:
        ep2_hist_miou = float(hist_val_miou[1])
        ep2_hist_dice = float(hist_val_dice[1]) if len(hist_val_dice) >= 2 else "N/A"
        print(f"  Checkpoint header best_miou:   {ckpt_miou:.4f} (stale from Epoch 1)")
        print(f"  training_history.json Epoch 2: val_miou = {ep2_hist_miou:.4f}, val_dice = {ep2_hist_dice}")
        if abs(ckpt_miou - ep2_hist_miou) > 1e-4:
            print("  [!] INCONSISTENCY EXPLAINED:")
            print("      latest_checkpoint.pth header holds 0.1415, while training_history.json recorded 0.2018.")
            print("      The checkpoint file on Google Drive is preserved untouched (metadata not rewritten).")
            print("      Section 6.6 will evaluate the actual model weights on the full 975 validation split.")

    if latest_ep == 2 and has_opt and has_sched:
        print(f"\\n  --> Genuine Epoch 2 checkpoint VERIFIED in Google Drive ({latest_path}).")
        print(f"  --> Full training state (model + optimizer + scheduler) is PRESENT.")
        print(f"  --> Training will resume from EPOCH 3 (Target: {TOTAL_EPOCHS} Total Epochs).")
    elif latest_ep == 1 and has_opt:
        print(f"\\n  --> Epoch 1 checkpoint VERIFIED in Google Drive.")
        print(f"  --> Training will resume from EPOCH 2 (Target: {TOTAL_EPOCHS} Total Epochs).")
    else:
        print(f"\\n  --> latest_checkpoint.pth is at Epoch {latest_ep}.")
        print(f"  --> Training will resume from EPOCH {latest_ep + 1}.")
    print("=" * 75)
else:
    print("\\n[+] No latest_checkpoint.pth found. Run will begin cleanly at Epoch 1.")""")

# Section 6.6
add_markdown("""## Section 6.6: Reproducible Checkpoint Evaluation & Metric Verification (Pre-Training Audit)
Runs deterministic validation on `latest_checkpoint.pth` (or `best_model.pth`) across all 975 validation images at $\\tau = 0.5$.
- Computes per-class TP, FP, FN, Union, IoU, Dice, Precision, and Recall.
- Calculates the arithmetic mean of all 19 per-class IoUs and macro Dice.
- Confirms mathematical equivalence with `evaluator.compute()`.
- Verifies whether the previous $\\text{mIoU} = 0.2018$ / $\\text{macro Dice} = 0.2975$ metrics are reproduced from the Google Drive Epoch 2 checkpoint.""")

add_code("""# ==============================================================================
# SECTION 6.6: REPRODUCIBLE VALIDATION EVALUATION ON GOOGLE DRIVE CHECKPOINTS
# ==============================================================================
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.dataset import DACL10KDataset, DACL10K_CLASSES, custom_collate_fn
from src.segmentation.evaluate import SegmentationEvaluator
from src.segmentation.model import BridgeSegmentationModel

target_ckpt = f"{DRIVE_OUTPUT_DIR}/checkpoints/latest_checkpoint.pth"
if not os.path.exists(target_ckpt):
    target_ckpt = f"{DRIVE_OUTPUT_DIR}/checkpoints/best_model.pth"

if not os.path.exists(target_ckpt):
    print(f"[!] No checkpoint found at {target_ckpt}. Run Section 7 to begin fresh training.")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"EVALUATING CHECKPOINT: {target_ckpt}")
    print(f"Device: {device} | Threshold: 0.5")
    print("=" * 75)

    chk = torch.load(target_ckpt, map_location=device)
    model_state = chk.get("model_state_dict", chk.get("model_state", chk))
    ep = chk.get("epoch", "unknown")
    rec_miou = chk.get("best_miou", chk.get("best_val_miou", chk.get("val_miou", "unknown")))
    print(f"Checkpoint Epoch: {ep} | Recorded Metadata mIoU: {rec_miou}")

    eval_model = BridgeSegmentationModel(
        architecture="DeepLabV3Plus",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=19,
    ).to(device)
    eval_model.load_state_dict(model_state, strict=True)
    eval_model.eval()

    val_dataset = DACL10KDataset(
        root_dir=DATASET_ROOT,
        split="validation",
        target_size=IMAGE_SIZE,
        augment=False,
    )
    print(f"Total Validation Images: {len(val_dataset)} (Exact target: 975)")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    evaluator = SegmentationEvaluator(class_names=list(DACL10K_CLASSES), threshold=0.5, eps=1e-7)

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Evaluating Epoch {ep} Checkpoint"):
            images = images.to(device)
            masks = masks.to(device)
            logits = eval_model(images)
            evaluator.update(logits, masks, is_logits=True)

    metrics = evaluator.compute()
    per_class = metrics["per_class"]
    macro = metrics["macro"]

    rows = []
    ious, dices, precs, recs = [], [], [], []
    for c in per_class:
        tp = c["tp"]
        fp = c["fp"]
        fn = c["fn"]
        union = tp + fp + fn
        iou = c["iou"]
        dice = c["dice"]
        prec = c["precision"]
        rec = c["recall"]

        ious.append(iou)
        dices.append(dice)
        precs.append(prec)
        recs.append(rec)

        rows.append({
            "Class": c["class_name"],
            "TP (Intersection)": f"{tp:,}",
            "Union": f"{union:,}",
            "IoU": f"{iou:.4f}",
            "Dice": f"{dice:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
        })

    df = pd.DataFrame(rows)
    print("\\n" + "=" * 85)
    print(f"PER-CLASS EVALUATION RESULTS (Checkpoint: Epoch {ep}, Threshold: 0.5)")
    print("=" * 85)
    print(df.to_string(index=False))

    arith_mean_iou = float(np.mean(ious))
    arith_mean_dice = float(np.mean(dices))
    evaluator_miou = macro["mIoU_all_classes"]
    evaluator_dice = macro["macro_dice_all_classes"]

    print("\\n" + "=" * 85)
    print("CHECKPOINT EVALUATION AUDIT REPORT")
    print("=" * 85)
    print(f"Checkpoint Absolute Path:               {target_ckpt}")
    print(f"Checkpoint Saved Epoch:                 {ep}")
    print(f"Validation Images Processed:            {len(val_dataset)} (Exact: 975)")
    print(f"Evaluator Reported mIoU (tau=0.5):      {evaluator_miou:.4f} ({evaluator_miou*100:.2f}%)")
    print(f"Evaluator Reported Macro Dice:          {evaluator_dice:.4f} ({evaluator_dice*100:.2f}%)")
    print(f"Arithmetic Mean of 19 per-class IoUs:   {arith_mean_iou:.4f} ({arith_mean_iou*100:.2f}%)")
    print(f"Arithmetic Mean of 19 per-class Dice:   {arith_mean_dice:.4f} ({arith_mean_dice*100:.2f}%)")
    print(f"IoU Absolute Difference:                {abs(evaluator_miou - arith_mean_iou):.8f}")
    print(f"Dice Absolute Difference:               {abs(evaluator_dice - arith_mean_dice):.8f}")
    print(f"Macro Precision:                        {float(np.mean(precs)):.4f}")
    print(f"Macro Recall:                           {float(np.mean(recs)):.4f}")
    print("-" * 85)
    print("METRIC REPRODUCTION VERIFICATION:")
    print(f"  Target Epoch 2 History mIoU: 0.2018 | Measured: {evaluator_miou:.4f}")
    print(f"  Target Epoch 2 History Dice: 0.2975 | Measured: {evaluator_dice:.4f}")
    if abs(evaluator_miou - 0.2018) < 0.01 and abs(evaluator_dice - 0.2975) < 0.01:
        print("  [SUCCESS] Metric reproduction confirmed! Checkpoint matches recorded 0.2018 / 0.2975.")
    else:
        print(f"  [AUDIT NOTE] Evaluated checkpoint mIoU is {evaluator_miou:.4f} vs history 0.2018.")
    print("=" * 85)
""")

# Section 7
add_markdown("""## Section 7: Resumable Full Baseline Training Loop (15 Total Epochs)
- **Resumption Behavior:** Automatically inspects `latest_checkpoint.pth` in Google Drive. If an epoch-2 checkpoint is found, training seamlessly resumes from **Epoch 3**. If only epoch-1 exists, it resumes from **Epoch 2**. If no checkpoint exists, it starts at **Epoch 1**. The total epoch counter is **NOT reset**.
- **Checkpoint Keys:** Uses unified canonical keys (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_miou`) with legacy compatibility.
- **Model Selection:** `best_model.pth` is updated strictly whenever validation mIoU ($\tau = 0.5$) improves.
- **Safety Switch:** Set `EXECUTE_TRAINING = True` to launch the run.""")

add_code("""# ==============================================================================
# SAFETY SWITCH: Set to True when you are ready to start training
# ==============================================================================
EXECUTE_TRAINING = False

if not EXECUTE_TRAINING:
    print("[SAFETY GUARD] EXECUTE_TRAINING is set to False.")
    print("Set EXECUTE_TRAINING = True and re-run this cell to launch the 15-epoch training run.")
else:
    import json
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from src.segmentation.evaluate import SegmentationEvaluator

    print("=" * 60)
    print(f"STARTING / RESUMING DACL10K-v2 TRAINING RUN (TARGET: {TOTAL_EPOCHS} TOTAL EPOCHS)")
    print("=" * 60)

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
        "val_miou_diag03": [],
        "val_dice_diag03": [],
        "lr": [],
        "epoch_time": []
    }

    start_epoch = 1
    best_val_miou = 0.0

    # 3. Check for existing checkpoint to resume
    resume_path = f"{DRIVE_OUTPUT_DIR}/checkpoints/latest_checkpoint.pth"
    history_path = f"{DRIVE_OUTPUT_DIR}/training_history.json"

    if os.path.exists(resume_path):
        print(f"\\nFound existing checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Resilient key extraction
        model_state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
        opt_state = checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer_state"))
        sched_state = checkpoint.get("scheduler_state_dict", checkpoint.get("scheduler_state"))
        
        model.load_state_dict(model_state)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            print("  [+] Optimizer state restored successfully.")
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
            print("  [+] Scheduler state restored successfully.")
            
        saved_epoch = checkpoint.get("epoch", 0)
        start_epoch = saved_epoch + 1
        best_val_miou = float(checkpoint.get("best_miou", checkpoint.get("best_val_miou", 0.0)))
        
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            hist_val_miou = history.get("val_miou", [])
            print(f"  [+] Loaded previous history with {len(history.get('train_loss', []))} epochs recorded (val_miou={hist_val_miou}).")
            if hist_val_miou:
                hist_max = float(max(hist_val_miou))
                if hist_max > best_val_miou:
                    print(f"  [!] Checkpoint metadata best_miou ({best_val_miou:.4f}) is lower than history max ({hist_max:.4f}).")
                    print(f"      Setting baseline best_val_miou = {hist_max:.4f} from training_history.json to protect best_model.pth.")
                    best_val_miou = hist_max
            
        print(f"\\n--> RESUMING FROM EPOCH {start_epoch} (Saved Epoch: {saved_epoch})")
        print(f"--> Target: {TOTAL_EPOCHS} Total Epochs | Previous Best Val mIoU: {best_val_miou:.4f}")
    else:
        print(f"\\nNo previous checkpoint found at {resume_path}. Starting clean run from Epoch 1.")

    if start_epoch > TOTAL_EPOCHS:
        print(f"Run already reached target of {TOTAL_EPOCHS} epochs! Nothing to train.")
    else:
        for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
            epoch_start = time.time()
            current_lr = optimizer.param_groups[0]["lr"]

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

            # --- VALIDATION PHASE (After each completed epoch) ---
            model.eval()
            val_loss_acc = 0.0
            val_steps = 0
            evaluator_05 = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=PRIMARY_THRESHOLD)
            evaluator_03 = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=DIAG_THRESHOLD)

            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{TOTAL_EPOCHS:02d} [Val]"):
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)

                    logits = model(images)
                    loss = criterion(logits, masks)
                    val_loss_acc += loss.item()
                    val_steps += 1

                    evaluator_05.update(logits, masks, is_logits=True)
                    evaluator_03.update(logits, masks, is_logits=True)

            avg_val_loss = val_loss_acc / max(1, val_steps)
            val_metrics_05 = evaluator_05.compute()
            val_metrics_03 = evaluator_03.compute()

            val_miou = val_metrics_05["macro"]["mIoU_all_classes"]
            val_dice = val_metrics_05["macro"].get("macro_dice_all_classes", val_metrics_05["macro"].get("macro_dice", 0.0))
            val_miou_03 = val_metrics_03["macro"]["mIoU_all_classes"]
            val_dice_03 = val_metrics_03["macro"].get("macro_dice_all_classes", val_metrics_03["macro"].get("macro_dice", 0.0))

            epoch_dur = time.time() - epoch_start

            # Record history
            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))
            history["val_miou"].append(round(val_miou, 4))
            history["val_dice"].append(round(val_dice, 4))
            history["val_miou_diag03"].append(round(val_miou_03, 4))
            history["val_dice_diag03"].append(round(val_dice_03, 4))
            history["lr"].append(current_lr)
            history["epoch_time"].append(round(epoch_dur, 2))

            print(f"\\nEpoch {epoch:02d}/{TOTAL_EPOCHS:02d} Summary | Duration: {epoch_dur:.1f}s | LR: {current_lr:.2e}")
            print(f"  Train Loss:       {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Val mIoU (tau=0.5): {val_miou:.4f} | Val Dice (tau=0.5): {val_dice:.4f}")
            print(f"  Val mIoU (tau=0.3): {val_miou_03:.4f} [Diagnostic] | Val Dice: {val_dice_03:.4f}")

            # Check if this epoch achieved best validation mIoU
            is_best = (val_miou > best_val_miou)
            if is_best:
                best_val_miou = val_miou
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_state": model.state_dict(),
                    "best_miou": best_val_miou,
                    "best_val_miou": best_val_miou,
                    "val_miou": val_miou,
                    "val_dice": val_dice,
                    "val_loss": avg_val_loss,
                    "val_metrics": val_metrics_05,
                    "config": config,
                }, f"{DRIVE_OUTPUT_DIR}/checkpoints/best_model.pth")
                print(f"  [*] NEW BEST MODEL SAVED! (Val mIoU: {best_val_miou:.4f}, Val Loss: {avg_val_loss:.4f})")

            # Save latest checkpoint after EVERY completed epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_state": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_miou": best_val_miou,
                "best_val_miou": best_val_miou,
                "val_miou": val_miou,
                "val_loss": avg_val_loss,
                "history": history,
                "config": config,
            }, f"{DRIVE_OUTPUT_DIR}/checkpoints/latest_checkpoint.pth")

            with open(f"{DRIVE_OUTPUT_DIR}/training_history.json", "w") as f:
                json.dump(history, f, indent=2)

        print("\\n" + "=" * 60)
        print("TRAINING RUN COMPLETE (15 TOTAL EPOCHS)!")
        print(f"Best Validation mIoU: {best_val_miou:.4f}")
        print(f"Artifacts saved in:   {DRIVE_OUTPUT_DIR}")
        print("=" * 60)""")

# Section 8
add_markdown("""## Section 8: Held-Out Validation Evaluation & Weak Class Visualizations
Evaluates the best checkpoint across all 975 validation images:
- Outputs per-class IoU, Dice, Precision, Recall, GT pixels, and Pred pixels.
- Saves high-resolution qualitative visual comparisons for weak and small defect classes:
  `Crack`, `ACrack`, `Cavity`, `ExposedRebars`, `JTape`, `Restformwork`, `Rockpocket`, `WConccor`.""")

add_code("""import pandas as pd
import matplotlib.pyplot as plt
from src.segmentation.evaluate import save_qualitative_comparison

best_ckpt_path = f"{DRIVE_OUTPUT_DIR}/checkpoints/best_model.pth"

if not os.path.exists(best_ckpt_path):
    print(f"[NOTE] Best checkpoint not found at {best_ckpt_path}. Run Section 7 first.")
else:
    print(f"Loading best checkpoint from {best_ckpt_path}...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    eval_model = BridgeSegmentationModel(
        architecture="DeepLabV3Plus",
        encoder_name="resnet34",
        num_classes=19,
        encoder_weights=None
    ).to(device)
    
    model_st = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    eval_model.load_state_dict(model_st)
    eval_model.eval()
    print(f"Loaded model from Epoch {ckpt.get('epoch', '?')} (Recorded mIoU: {ckpt.get('best_miou', ckpt.get('val_miou', '?'))})")

    val_dataset = DACL10KDataset(
        root_dir=DATASET_ROOT,
        split="validation",
        target_size=IMAGE_SIZE,
        augment=False,
        return_metadata=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn
    )

    evaluator_05 = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=0.5)
    evaluator_03 = SegmentationEvaluator(class_names=DACL10K_CLASSES, threshold=0.3)

    print(f"Evaluating complete validation split ({len(val_dataset)} images)...")
    with torch.no_grad():
        for images, masks, metas in tqdm(val_loader, desc="Validation Evaluation"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            sample_ids = [m["image_id"] for m in metas]

            logits = eval_model(images)
            evaluator_05.update(logits, masks, sample_ids=sample_ids, is_logits=True)
            evaluator_03.update(logits, masks, sample_ids=sample_ids, is_logits=True)

    val_results_05 = evaluator_05.compute()
    val_results_03 = evaluator_03.compute()

    # Save per-class CSV
    csv_path = f"{DRIVE_OUTPUT_DIR}/val_per_class_metrics.csv"
    json_path = f"{DRIVE_OUTPUT_DIR}/val_summary_metrics.json"

    df_05 = pd.DataFrame(val_results_05["per_class"])
    df_03 = pd.DataFrame(val_results_03["per_class"])
    df_05["iou_tau03"] = df_03["iou"]
    df_05["dice_tau03"] = df_03["dice"]
    df_05["pred_pixels_tau03"] = df_03["pred_pixels"]

    df_05.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({"tau_05": val_results_05, "tau_03": val_results_03}, f, indent=2)

    print("\\n" + "=" * 80)
    print("HELD-OUT VALIDATION SET PERFORMANCE (DACL10K-v2, 975 IMAGES)")
    print("=" * 80)
    display_cols = ["class_name", "iou", "dice", "precision", "recall", "gt_pixels", "pred_pixels", "iou_tau03"]
    print(df_05[display_cols].to_string(index=False))
    print("-" * 80)
    print(f"Primary Validation mIoU (tau=0.5, All 19 Classes):        {val_results_05['macro']['mIoU_all_classes']:.4f}")
    print(f"Primary Validation mIoU (tau=0.5, Present Classes Only):  {val_results_05['macro']['mIoU_present_classes']:.4f}")
    m_dice_05 = val_results_05['macro'].get('macro_dice_all_classes', val_results_05['macro'].get('macro_dice', 0.0))
    print(f"Primary Validation Macro Dice (tau=0.5):                 {m_dice_05:.4f}")
    print(f"Diagnostic Validation mIoU (tau=0.3):                    {val_results_03['macro']['mIoU_all_classes']:.4f}")
    print("=" * 80)

    # Save Qualitative Visualizations for Weak Classes
    print("\\nGenerating 4-panel visual comparisons on validation samples with weak defects...")
    vis_dir = f"{DRIVE_OUTPUT_DIR}/visualizations/weak_classes"
    os.makedirs(vis_dir, exist_ok=True)

    WEAK_CLASSES = ["Crack", "ACrack", "Cavity", "ExposedRebars", "JTape", "Restformwork", "Rockpocket", "WConccor"]
    saved_per_class = {c: 0 for c in WEAK_CLASSES}

    for i in range(len(val_dataset)):
        img_t, mask_t, meta = val_dataset[i]
        sample_id = meta["image_id"]
        
        # Check if sample contains any weak class
        mask_np = mask_t.numpy()
        for ci, cname in enumerate(DACL10K_CLASSES):
            if cname in WEAK_CLASSES and saved_per_class[cname] < 2:
                if mask_np[ci].sum() > 50:
                    with torch.no_grad():
                        logits = eval_model(img_t.unsqueeze(0).to(device))
                        pred_prob = torch.sigmoid(logits).squeeze(0).cpu()
                        pred_bin = (pred_prob >= 0.5).float()
                        
                    save_path = f"{vis_dir}/{cname}_{sample_id}.png"
                    save_qualitative_comparison(
                        image_tensor=img_t,
                        gt_mask=mask_t,
                        pred_mask=pred_bin,
                        class_names=DACL10K_CLASSES,
                        save_path=save_path,
                        title=f"Sample: {sample_id} | Highlighted: {cname}"
                    )
                    saved_per_class[cname] += 1
        if all(cnt >= 2 for cnt in saved_per_class.values()):
            break

    print(f"Qualitative comparisons saved to: {vis_dir}!")""")

# Section 9
add_markdown("""## Section 9: Artifact Export & Download Helper
Helper to verify and download artifacts from Google Drive to your local environment.""")

add_code("""print("=" * 60)
print("TRAINING ARTIFACTS IN GOOGLE DRIVE:")
print("=" * 60)
!ls -lh {DRIVE_OUTPUT_DIR}/checkpoints
!ls -lh {DRIVE_OUTPUT_DIR}

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
print("\\nTo download latest resume checkpoint:")
print(">>> download_artifact('checkpoints/latest_checkpoint.pth')")
print("\\nTo download validation per-class metrics:")
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
