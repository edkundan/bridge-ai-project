# DACL10K-v2 Colab Runtime Recovery & Quick Resume Guide

**Project:** DACL10K-v2 Bridge Defect Multi-Label Semantic Segmentation  
**Notebook:** [`notebooks/DACL10K_Colab_Training.ipynb`](file:///Users/kundankumar/Documents/untitled%20folder%202/bridge-ai-project/notebooks/DACL10K_Colab_Training.ipynb)  
**Target:** 15 Total Epochs (Resumable baseline)  
**Output Directory:** Google Drive `/content/drive/MyDrive/bridge_ai_experiments/segmentation_baseline`  

---

## 1. Quick Decision Tree: Browser Refresh vs. Runtime Reset

| Event | Colab State | In-Memory Variables & Packages | Action Required |
|:---|:---|:---|:---|
| **A. Browser Tab Refresh / Reconnect** | VM is **still running** in the cloud | **Intact** | **Do NOT run setup cells.** Reconnect to the active session. Check cell execution indicators to see ongoing training progress. |
| **B. Runtime Disconnect / Reset / New Machine** | VM was **terminated or restarted** | **Wiped (clean environment)** | **Run Cell R1 (`RUNTIME RECOVERY`) once.** It restores Google Drive, GPU checks, missing packages, repository code, datasets, configuration variables, and audits Drive checkpoints in a single idempotent pass.<br>Then use **Cell R2 (`QUICK RESUME`)** to resume training. |

> ⚠️ **Golden Rule:** **Never assume an in-memory variable or installed package survived a runtime reset.** If the kernel was restarted or a new VM was assigned, execute **Cell R1 (Runtime Recovery)** first before running any other cell!

---

## 2. Cell R1: Idempotent Runtime Recovery (What It Does)

Located right after the header in the notebook, **Cell R1** executes a deterministic 7-step restoration:

1. **Mount Google Drive:** Mounts `/content/drive` if not already mounted (`force_remount=False`).
2. **GPU Verification:** Checks `torch.cuda.is_available()` and logs GPU device name and VRAM (T4/A100).
3. **Idempotent Dependency Installation:** Uses `importlib.util.find_spec` to check whether packages are already present. Only runs `pip install` for genuinely missing packages (`segmentation-models-pytorch`, `albumentations`, etc.).
4. **Non-Destructive Code Restoration:** Locates `/content/bridge-ai-project`. If already present, it preserves existing user edits. If absent, restores from Drive zip/folder or GitHub clone, and adds the directory to `sys.path`.
5. **Dataset Verification:** Discovers and validates DACL10K-v2 dataset roots. Verifies actual file counts: exactly 6,935 train images and 975 validation images.
6. **Configuration & Class Restoration:** Restores all training hyperparameters (`BATCH_SIZE=8`, `IMAGE_SIZE=512x512`, `TOTAL_EPOCHS=15`, `PRIMARY_THRESHOLD=0.5`), classes (`DACL10K_CLASSES`), and loss definitions (`CombinedBCEDiceLoss`).
7. **Google Drive Checkpoint Audit:**
   - Scans `/content/drive/MyDrive/bridge_ai_experiments/segmentation_baseline/checkpoints/`
   - Inspects `latest_checkpoint.pth` and `best_model.pth`
   - Logs epoch, `best_miou`, and verifies `optimizer_state_dict` and `scheduler_state_dict`
   - Calculates `RECOVERY_START_EPOCH = saved_epoch + 1`
   - If no valid checkpoint exists, explains options instead of silently training.
   - **Does NOT launch training automatically.**

---

## 3. Cell R2: Quick Resume (How to Launch Training)

Located immediately below Cell R1, **Cell R2** executes the baseline training continuation:

- **Safety Switch:** Initialized with `EXECUTE_QUICK_RESUME = False`.
- **Pre-execution Verification:** Confirms all recovery variables and dataset loaders exist.
- **Resumption Behavior:**
  - Loads model weights, optimizer momentum, and scheduler steps from `RECOVERY_CHECKPOINT`.
  - Resumes strictly from `RECOVERY_START_EPOCH` (epoch counter is **not** reset).
  - Evaluates all 975 validation images after every epoch at $\tau=0.5$ (benchmark) and $\tau=0.3$ (diagnostic).
  - Saves `latest_checkpoint.pth` after every completed epoch.
  - Updates `best_model.pth` strictly when validation mIoU at $\tau=0.5$ improves.
  - Updates `training_history.json` on Google Drive.

---

## 4. Exact User Steps in Google Colab

### When you connect after a break or runtime reset:
1. Open [`notebooks/DACL10K_Colab_Training.ipynb`](file:///Users/kundankumar/Documents/untitled%20folder%202/bridge-ai-project/notebooks/DACL10K_Colab_Training.ipynb) in Google Colab.
2. Ensure GPU runtime is connected (**Runtime -> Change runtime type -> T4 GPU**).
3. **Execute Cell R1 (RUNTIME RECOVERY):**
   - Wait for it to print:
     ```text
     [RECOVERY STATUS: FULL-STATE RESUME VERIFIED]
       Valid Checkpoint:     .../checkpoints/latest_checkpoint.pth
       Completed Epoch:      2 of 15
       Next Epoch to Train:  3
       Best Val mIoU So Far: 0.2018
     ```
4. *(Optional)* If you want to audit the checkpoint metrics first, run **Section 6.6** to evaluate the 975 validation images.
5. **Execute Cell R2 (QUICK RESUME):**
   - Set `EXECUTE_QUICK_RESUME = True`.
   - Run the cell. Training will seamlessly resume from Epoch 3 through Epoch 15!
