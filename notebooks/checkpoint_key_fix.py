"""
Fix: Corrected checkpoint saving and loading code for the Colab notebook.
This file contains the corrected Python snippets that replace the inline
saving/loading cells in DACL10K_Colab_Training.ipynb.

KEY FIX: All checkpoint keys unified to match train.py format:
  - 'model_state_dict'   (was 'model_state' in old notebook cells)
  - 'optimizer_state_dict'  (was 'optimizer_state')
  - 'scheduler_state_dict'  (was 'scheduler_state')
  - 'best_miou'          (added, was missing)

The actual best_model.pth on disk was saved by train.py using 'model_state_dict'.
All load/save paths must use this key consistently.

USAGE: Copy-paste these corrected snippets into the corresponding cells in Colab.
"""

# === CORRECTED SNIPPET A: Save latest checkpoint ===
SAVE_LATEST_CHECKPOINT_SNIPPET = """
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_miou": best_val_miou,
    "config": config,
}, f"{DRIVE_OUTPUT_DIR}/checkpoints/latest_checkpoint.pth")
"""

# === CORRECTED SNIPPET B: Save best model ===
SAVE_BEST_MODEL_SNIPPET = """
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "best_miou": val_miou,
    "val_miou": val_miou,
    "val_dice": val_dice,
    "val_metrics": val_metrics,
    "config": config,
}, f"{DRIVE_OUTPUT_DIR}/checkpoints/best_model.pth")
"""

# === CORRECTED SNIPPET C: Resume from checkpoint ===
RESUME_FROM_CHECKPOINT_SNIPPET = """
if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_PATH):
    print(f"Resuming from: {RESUME_PATH}")
    checkpoint = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_miou = checkpoint.get("best_val_miou", checkpoint.get("best_miou", 0.0))
    print(f"Resumed at epoch {start_epoch}, best mIoU so far: {best_val_miou:.4f}")
"""

# === CORRECTED SNIPPET D: Load best model for evaluation ===
LOAD_BEST_MODEL_SNIPPET = """
ckpt = torch.load(f"{DRIVE_OUTPUT_DIR}/checkpoints/best_model.pth", map_location=device)
eval_model = BridgeSegmentationModel(
    architecture='DeepLabV3Plus',
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    num_classes=19,
).to(device)
eval_model.load_state_dict(ckpt["model_state_dict"])
eval_model.eval()
print(f"Loaded best model from epoch {ckpt.get('epoch', '?')}, val mIoU: {ckpt.get('best_miou', ckpt.get('val_miou', '?'))}")
"""

if __name__ == "__main__":
    print("Checkpoint key correction snippets loaded.")
    print("Snippets A-D above should replace the corresponding notebook cells.")
