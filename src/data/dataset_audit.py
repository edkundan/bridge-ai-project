import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SEVERITY_KEYS = {'severity', 'risk', 'grade', 'urgency', 'priority', 'damage_level', 'condition_rating'}

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class DatasetAuditor:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'images'
        self.anno_dir = self.root_dir / 'annotations'
        
        self.stats = {
            'images_per_split': defaultdict(int),
            'annotations_per_split': defaultdict(int),
            'missing_annotations': 0,
            'missing_images': 0,
            'corrupted_images': 0,
            'malformed_jsons': 0,
            'duplicate_filenames': 0,
            'invalid_polygons': 0,
            'empty_annotations': 0,
            'class_frequencies': defaultdict(int),
            'image_dimensions': [],
            'shapes_per_image': [],
            'severity_keys_found': set(),
            'had_WConccor_values': defaultdict(int),
            'description_values': defaultdict(int),
            'total_polygons': 0,
        }
        self.all_filenames = set()

    def audit(self):
        logger.info(f"Starting audit for dataset at {self.root_dir}")
        if not self.root_dir.exists():
            logger.error("Dataset root directory does not exist.")
            return

        image_splits = [d.name for d in self.image_dir.iterdir() if d.is_dir()] if self.image_dir.exists() else []
        anno_splits = [d.name for d in self.anno_dir.iterdir() if d.is_dir()] if self.anno_dir.exists() else []

        logger.info(f"Found image splits: {image_splits}")
        logger.info(f"Found annotation splits: {anno_splits}")

        for split in image_splits:
            split_dir = self.image_dir / split
            images = list(split_dir.glob('*.jpg'))
            self.stats['images_per_split'][split] = len(images)
            for img_path in tqdm(images, desc=f"Checking images in {split}"):
                if img_path.name in self.all_filenames:
                    self.stats['duplicate_filenames'] += 1
                self.all_filenames.add(img_path.name)
                
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    self.stats['corrupted_images'] += 1

                if split in anno_splits:
                    anno_path = self.anno_dir / split / (img_path.stem + '.json')
                    if not anno_path.exists():
                        self.stats['missing_annotations'] += 1

        for split in anno_splits:
            split_dir = self.anno_dir / split
            annos = list(split_dir.glob('*.json'))
            self.stats['annotations_per_split'][split] = len(annos)
            for anno_path in tqdm(annos, desc=f"Checking annotations in {split}"):
                img_path = self.image_dir / split / (anno_path.stem + '.jpg')
                if not img_path.exists():
                    self.stats['missing_images'] += 1

                try:
                    with open(anno_path, 'r') as f:
                        data = json.load(f)
                except Exception:
                    self.stats['malformed_jsons'] += 1
                    continue

                if 'imageHeight' in data and 'imageWidth' in data:
                    self.stats['image_dimensions'].append((data['imageWidth'], data['imageHeight']))
                
                if 'had_WConccor' in data:
                    self.stats['had_WConccor_values'][str(data['had_WConccor'])] += 1
                
                for key in data.keys():
                    if key.lower() in SEVERITY_KEYS:
                        self.stats['severity_keys_found'].add(key)
                
                shapes = data.get('shapes', [])
                if not shapes:
                    self.stats['empty_annotations'] += 1
                
                self.stats['shapes_per_image'].append(len(shapes))
                
                for shape in shapes:
                    for key in shape.keys():
                        if key.lower() in SEVERITY_KEYS:
                            self.stats['severity_keys_found'].add(key)
                    
                    desc = shape.get('description', '')
                    self.stats['description_values'][str(desc)] += 1
                    
                    label = shape.get('label', 'unknown')
                    self.stats['class_frequencies'][label] += 1
                    self.stats['total_polygons'] += 1
                    
                    points = shape.get('points', [])
                    if len(points) < 3:
                        self.stats['invalid_polygons'] += 1
        
        self._calculate_summaries()
        self._save_report()
        self._print_summary()

    def _calculate_summaries(self):
        dims = np.array(self.stats['image_dimensions'])
        if len(dims) > 0:
            self.stats['width_mean'] = float(np.mean(dims[:, 0]))
            self.stats['height_mean'] = float(np.mean(dims[:, 1]))
        else:
            self.stats['width_mean'] = self.stats['height_mean'] = 0.0
            
        shapes_arr = np.array(self.stats['shapes_per_image'])
        if len(shapes_arr) > 0:
            self.stats['shapes_min'] = int(np.min(shapes_arr))
            self.stats['shapes_max'] = int(np.max(shapes_arr))
            self.stats['shapes_mean'] = float(np.mean(shapes_arr))
            self.stats['shapes_median'] = float(np.median(shapes_arr))
            self.stats['shapes_std'] = float(np.std(shapes_arr))
            
        classes = self.stats['class_frequencies']
        if classes:
            max_count = max(classes.values())
            self.stats['imbalance_ratios'] = {k: max_count / v for k, v in classes.items()}
            total_polygons = sum(classes.values())
            self.stats['rare_classes'] = [k for k, v in classes.items() if (v / total_polygons) < 0.01]

    def _save_report(self):
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / 'dataset_audit_summary.json'
        
        # Convert sets and defaultdicts for JSON
        json_stats = dict(self.stats)
        json_stats['severity_keys_found'] = list(json_stats['severity_keys_found'])
        for k in ['images_per_split', 'annotations_per_split', 'class_frequencies', 'had_WConccor_values', 'description_values']:
            json_stats[k] = dict(json_stats[k])
        
        json_stats.pop('image_dimensions', None)
        json_stats.pop('shapes_per_image', None)
        
        with open(report_path, 'w') as f:
            json.dump(json_stats, f, indent=4)
        logger.info(f"Report saved to {report_path}")

    def _print_summary(self):
        print("\n" + "="*50)
        print("DATASET AUDIT SUMMARY")
        print("="*50)
        
        print("\nImages per split:")
        for k, v in self.stats['images_per_split'].items():
            print(f"  {k}: {v}")
            
        print("\nAnnotations per split:")
        for k, v in self.stats['annotations_per_split'].items():
            print(f"  {k}: {v}")
            
        print(f"\nMissing annotations for images: {self.stats['missing_annotations']}")
        print(f"Missing images for annotations: {self.stats['missing_images']}")
        print(f"Corrupted images: {self.stats['corrupted_images']}")
        print(f"Malformed JSONs: {self.stats['malformed_jsons']}")
        print(f"Duplicate filenames: {self.stats['duplicate_filenames']}")
        print(f"Invalid polygons (<3 points): {self.stats['invalid_polygons']}")
        print(f"Empty annotations: {self.stats['empty_annotations']}")
        
        print("\nImage Dimensions (Mean):")
        print(f"  Width: {self.stats.get('width_mean', 0):.2f}")
        print(f"  Height: {self.stats.get('height_mean', 0):.2f}")
        
        print("\nShapes per Image:")
        print(f"  Min: {self.stats.get('shapes_min', 0)}")
        print(f"  Max: {self.stats.get('shapes_max', 0)}")
        print(f"  Mean: {self.stats.get('shapes_mean', 0):.2f}")
        print(f"  Median: {self.stats.get('shapes_median', 0)}")
        print(f"  Std: {self.stats.get('shapes_std', 0):.2f}")
        
        print(f"\nSeverity keys found: {self.stats['severity_keys_found']}")
        print(f"had_WConccor values: {dict(self.stats['had_WConccor_values'])}")
        print(f"description values: {dict(self.stats['description_values'])}")
        
        print("\nClass Frequencies:")
        for k, v in sorted(self.stats['class_frequencies'].items(), key=lambda item: item[1], reverse=True):
            print(f"  {k}: {v}")
            
        print("\nRare Classes (< 1%):")
        for c in self.stats.get('rare_classes', []):
            print(f"  {c}")
            
        print("\nImbalance Ratios (Max Count / Class Count):")
        if 'imbalance_ratios' in self.stats:
            for k, v in sorted(self.stats['imbalance_ratios'].items(), key=lambda item: item[1]):
                print(f"  {k}: {v:.2f}")

        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Dataset Audit Script")
    parser.add_argument('--config', type=str, default='configs/dataset.yaml', help='Path to dataset config file')
    parser.add_argument('--dataset-root', type=str, help='Override dataset root path')
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
    
    root_dir = args.dataset_root
    if not root_dir and 'dataset' in config and 'root_dir' in config['dataset']:
        root_dir = config['dataset']['root_dir']
        
    if not root_dir:
        logger.error("Dataset root directory not specified in config or arguments.")
        return

    auditor = DatasetAuditor(root_dir)
    auditor.audit()

if __name__ == '__main__':
    main()
