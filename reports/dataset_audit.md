# Dataset Audit Report

### 1. Dataset Identification
- Name: DACL10K-v2 (dacl10k_v2_devphase)
- Version: v2 (20230811)
- License: CC BY-NC 4.0
- Citation: Flotzinger et al., "dacl10k: Benchmark for Semantic Bridge Damage Segmentation", WACV 2024, arXiv:2309.00460
- Task: Multi-label semantic segmentation
- Location: /Users/kundankumar/Downloads/dacl10k_v2_devphase

### 2. Directory Structure
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

### 3. Dataset Statistics
- Total images: 8,922
- Total annotation files: 7,910
- Total polygon annotations: 62,325

### 4. Train/Validation/Test Split
| Split | Images | Annotations |
|---|---|---|
| train | 6,935 | 6,935 |
| validation | 975 | 975 |
| testdev | 1,012 | 0 (no annotations — competition held-out set) |

### 5. Class List (19 classes)
13 damage classes: Crack, ACrack (Alligator Crack), Wetspot, Efflorescence, Rust, Rockpocket, Hollowareas, Cavity, Spalling, Graffiti, Weathering, Restformwork, ExposedRebars
6 object classes: Bearing, EJoint (Expansion Joint), Drainage, PEquipment (Protective Equipment), JTape (Joint Tape), WConccor (Washouts/Concrete corrosion)

### 6. Class Distribution (from train+validation combined)
| Class | Count | % | Imbalance Ratio |
|---|---|---|---|
| Rust | 14,720 | 23.62% | 1.00 |
| Spalling | 9,884 | 15.86% | 1.49 |
| Cavity | 9,271 | 14.88% | 1.59 |
| Weathering | 4,634 | 7.44% | 3.18 |
| Efflorescence | 3,965 | 6.36% | 3.71 |
| Crack | 3,626 | 5.82% | 4.06 |
| Graffiti | 2,200 | 3.53% | 6.69 |
| ExposedRebars | 2,000 | 3.21% | 7.36 |
| PEquipment | 1,888 | 3.03% | 7.80 |
| Wetspot | 1,661 | 2.67% | 8.86 |
| Drainage | 1,652 | 2.65% | 8.91 |
| Hollowareas | 1,557 | 2.50% | 9.45 |
| Bearing | 1,208 | 1.94% | 12.19 |
| JTape | 1,078 | 1.73% | 13.65 |
| Restformwork | 1,022 | 1.64% | 14.40 |
| Rockpocket | 617 | 0.99% | 23.86 |
| EJoint | 506 | 0.81% | 29.09 |
| ACrack | 424 | 0.68% | 34.72 |
| WConccor | 412 | 0.66% | 35.73 |

### 7. Class Imbalance
- Most frequent: Rust (14,720 polygons, 23.62%)
- Least frequent: WConccor (412 polygons, 0.66%)
- Maximum imbalance ratio: 35.73× (WConccor vs Rust)
- 4 rare classes below 1%: Rockpocket (0.99%), EJoint (0.81%), ACrack (0.68%), WConccor (0.66%)
- Significant class imbalance exists — will require weighted loss or oversampling strategies during training.

### 8. Image Dimension Statistics
- 339 unique resolutions
- Width: Min=336, Max=6,000, Mean=1,965.7, Std=1,138.7
- Height: Min=245, Max=5,152, Mean=1,596.7, Std=986.4
- Top resolutions: 1280×960 (1,148 images), 1024×768 (936), 1920×1440 (695), 4000×3000 (532)
- Images have highly variable resolutions — resizing/normalization is mandatory before model training.

### 9. Annotation Format
- Format: Modified labelme-style JSON (one JSON per image)
- Top-level keys: dacl10k_version, had_WConccor, imageHeight, imageName, imagePath, imageWidth, shapes, split
- Shape keys: description (always empty), label, points, shape_type (always 'polygon')
- Polygons defined as ordered vertex coordinate lists [[x1,y1], [x2,y2], ...]
- Multi-label: each pixel can belong to multiple classes simultaneously
- The dataset explicitly states: overlapping polygons of the SAME class are merged into one mask

### 10. Annotation Quality Checks
- Malformed JSONs: 0
- Invalid polygons (<3 points): 0
- Empty annotations (0 shapes): 71 images
- Shapes per image: Min=0, Max=158, Mean=7.88, Median=5.0, Std=8.12
- All 'description' fields are empty strings
- had_WConccor field: Present in 1,219 files (True), absent in 6,691 files

### 11. Missing/Corrupt Files
- Missing images (with annotation but no image): 0
- Missing annotations (with image but no annotation): 0 (train & validation)
- testdev split: 1,012 images with NO annotations (by design — competition held-out set)
- Corrupted images: 0 (all 8,922 images verified via PIL)
- Duplicate filenames: 0

### 12. Overlapping Labels
- Multi-label images (>1 class per image): 6,620 / 7,910 = 83.7%
- This confirms DACL10K-v2 is a MULTI-LABEL segmentation dataset
- Multiple defect types frequently co-occur on the same bridge surface
- Training must use multi-label loss (e.g., per-class BCE) not single-label softmax cross-entropy

### 13. Severity-Label Availability
**DACL10K-v2 does NOT contain severity labels.**
Exhaustive scan of all 7,910 JSON annotation files found zero occurrences of: severity, risk_level, damage_grade, severity_score, urgency, condition_rating, or any other severity/risk-related key.
The dataset provides ONLY: defect class labels and polygon boundaries (spatial localization).
Implication: Severity assessment must be derived downstream from computed defect metrics (area, extent, location, type criticality) — NOT from ground-truth labels in this dataset.

### 14. Problems Discovered
1. 71 images have empty annotations (0 shapes) — these are effectively background-only images
2. Significant class imbalance (35.73× between most and least frequent class)
3. 4 rare classes below 1% frequency: Rockpocket, EJoint, ACrack, WConccor
4. Highly variable image resolutions (339 unique sizes) — requires careful resize strategy
5. testdev split has no ground-truth annotations (competition format)
6. No severity/risk labels available in the dataset
7. had_WConccor metadata field only present in subset of files (provenance flag)

### 15. Recommendations
1. Keep the 71 empty-annotation images as valid background/negative samples during training
2. Apply class-aware weighted loss functions (e.g., inverse frequency weighting) or focal loss to handle class imbalance
3. Consider class-aware data augmentation or oversampling for the 4 rare classes
4. Standardize image resolution before training (e.g., resize to 512×512 or 1024×1024)
5. Use the validation split (975 images) for hyperparameter tuning; consider k-fold cross-validation if needed
6. Design a defect severity scoring module that computes severity from measurable defect properties (area ratio, crack width approximation, defect type criticality) rather than assuming ground-truth severity labels
7. Document that DACL10K-v2 annotation format requires conversion to binary mask arrays for standard segmentation training pipelines

### 16. Final Dataset Readiness Decision
**STATUS: READY FOR STAGE 1 SEGMENTATION TRAINING**
- All 7,910 annotated images are verified intact with matching annotations
- Class distribution and imbalance are documented and will inform loss function design
- Annotation format is well-structured and parseable
- The 71 empty images and 4 rare classes are known issues with documented mitigation strategies
- Multi-label nature is confirmed — training pipeline must support multi-label masks
