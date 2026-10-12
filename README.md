# Bridge AI Project

## Project Objective
This project aims to develop a multi-label semantic segmentation model for bridge damage detection using the DACL10K-v2 dataset. 

## Current Status
- **Stage 0:** Complete (Dataset Audit and Verification)
- **Stages 1-4:** Not yet implemented (e.g., Segmentation Training, Severity Scoring, etc.)

## Dataset Information
- **Name:** DACL10K-v2 (dacl10k_v2_devphase)
- **Version:** v2 (20230811)
- **Location:** `/Users/kundankumar/Downloads/dacl10k_v2_devphase`
- **License:** CC BY-NC 4.0

### Directory Structure
```
dacl10k_v2_devphase/
├── README.md
├── LICENSE.md
├── annotations/
│   ├── train/  (6935 .json files)
│   └── validation/  (975 .json files)
└── images/
    ├── train/  (6935 .jpg files)
    ├── validation/  (975 .jpg files)
    └── testdev/  (1012 .jpg files)
```

### Audit Findings Summary
- **Total Images:** 8,922 (Train: 6935, Val: 975, Testdev: 1012 without annotations)
- **Total Classes:** 19 (13 damage classes, 6 object classes)
- **Key Issues:** 
  - Significant class imbalance (Max ratio 35.73x)
  - 4 rare classes below 1% frequency
  - 71 empty-annotation images
  - Highly variable image resolutions
  - No severity/risk labels available in the dataset

## How to Run

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dataset Audit:**
   ```bash
   python src/data/dataset_audit.py --config configs/dataset.yaml
   ```

3. **Run Dataset Visualization:**
   ```bash
   python src/data/visualize_dataset.py --config configs/dataset.yaml --samples 10
   ```

## Reproducibility
- **Random Seed:** 42
- **Dataset Version:** v2 (20230811)

## License
The DACL10K-v2 dataset is licensed under CC BY-NC 4.0.
