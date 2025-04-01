"""
Base data module for object detection tasks (face and person detection).
"""
import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from PIL import Image


class LimitedDataset(Dataset):
    """Wrapper dataset that limits the number of samples per epoch."""
    def __init__(self, dataset: Dataset, max_samples: int):
        self.dataset = dataset
        self.max_samples = max_samples
        self.indices = list(range(len(dataset)))
        self.shuffle_indices()
        
    def shuffle_indices(self):
        """Shuffle indices at the start of each epoch"""
        random.shuffle(self.indices)
        self.indices = self.indices[:self.max_samples]
    
    def __len__(self):
        return min(self.max_samples, len(self.dataset))
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        return self.dataset[self.indices[idx]]

class DetectionDataset(Dataset):
    """Base dataset for object detection tasks."""
    
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        image_size: int = 640,
        augment: bool = True,
        cache_images: bool = False
    ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Find all image files
        self.img_files = sorted([f for f in self.img_dir.glob("*") 
                               if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
        
        # Cache labels
        self.labels = {}
        for img_file in self.img_files:
            label_file = self.label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file) as f:
                    labels = []
                    for line in f:
                        try:
                            values = line.strip().split()
                            if len(values) == 5:  # class, x, y, w, h
                                labels.append([float(x) for x in values])
                        except ValueError:
                            continue
                    self.labels[img_file] = np.array(labels, dtype=np.float32)
            else:
                self.labels[img_file] = np.zeros((0, 5), dtype=np.float32)

        # Setup minimal augmentations - just resize and basic transforms
        self.transform = A.Compose([
            # Simple resize to target size
            A.Resize(
                height=image_size,
                width=image_size,
                # always_apply=True
            ),
            # Only horizontal flip during training
            A.HorizontalFlip(p=0.5) if augment else A.NoOp(),
        ], 
        bbox_params=A.BboxParams(
            format='yolo',  # YOLO format: [x_center, y_center, width, height]
            label_fields=['class_labels']
        ))

        # Simple YOLO-style normalization (just divide by 255)
        self.normalize = A.Compose([
            A.Normalize(
                mean=[0, 0, 0],
                std=[255, 255, 255],
                max_pixel_value=255.0
            ),
        ])

    def __len__(self) -> int:
        return len(self.img_files)

    @staticmethod
    def clip_boxes(boxes: np.ndarray) -> np.ndarray:
        """Clip box coordinates to [0, 1] range with a small epsilon to avoid edge cases"""
        epsilon = 1e-6
        return np.clip(boxes, epsilon, 1.0 - epsilon)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Load image
        img_file = self.img_files[idx]
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get labels
        labels = self.labels[img_file].copy()  # Make a copy to avoid modifying original data
        
        # Prepare boxes and classes separately for albumentations
        if len(labels) > 0:
            classes = labels[:, 0]
            boxes = labels[:, 1:5]  # Get just the bbox coordinates
            
            # Strict validation of box coordinates - ensure all are in [0,1] range
            valid_indices = []
            for i, box in enumerate(boxes):
                # Check if all values are within valid range and not NaN/Inf
                if (np.all(box >= 0) and np.all(box <= 1) and 
                    not np.any(np.isnan(box)) and not np.any(np.isinf(box))):
                    valid_indices.append(i)
            
            # Filter to only valid boxes
            if len(valid_indices) < len(boxes):
                print(f"Warning: Filtered out {len(boxes) - len(valid_indices)} invalid boxes for {img_file.name}")
                boxes = boxes[valid_indices]
                classes = classes[valid_indices]
            
            # Additional safety - clip values to ensure they're in the valid range
            boxes = np.clip(boxes, 0.001, 0.999)
            
            # Ensure width and height are not zero
            boxes[:, 2] = np.maximum(boxes[:, 2], 0.01)  # width
            boxes[:, 3] = np.maximum(boxes[:, 3], 0.01)  # height
            
            # Make sure center + half width/height doesn't exceed boundaries
            boxes[:, 0] = np.minimum(boxes[:, 0], 1.0 - boxes[:, 2]/2)  # x center
            boxes[:, 1] = np.minimum(boxes[:, 1], 1.0 - boxes[:, 3]/2)  # y center
            boxes[:, 0] = np.maximum(boxes[:, 0], boxes[:, 2]/2)  # x center
            boxes[:, 1] = np.maximum(boxes[:, 1], boxes[:, 3]/2)  # y center
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros(0, dtype=np.float32)
        
        # Apply augmentations
        try:
            transformed = self.transform(
                image=img,
                bboxes=boxes if len(boxes) else [],
                class_labels=classes if len(classes) else []
            )
            
            img = transformed['image']
            
            # Create target dict
            if len(transformed['bboxes']):
                boxes = np.array(transformed['bboxes'])  # [N, 4]
                classes = np.array(transformed['class_labels'])  # [N]
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros(0, dtype=np.float32)
        except Exception as e:
            print(f"Error in transform for {img_file}: {e}")
            print(f"Original boxes: {boxes}")
            # Fallback to empty boxes
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros(0, dtype=np.float32)
        
        # Apply normalization
        img = self.normalize(image=img)['image']
        
        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        boxes = torch.from_numpy(boxes).float()
        classes = torch.from_numpy(classes).long()

        targets = {
            'boxes': boxes,
            'labels': classes,
            'image_id': torch.tensor([idx])
        }

        return img, targets

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for detection data."""
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
            
        images = torch.stack(images)
        
        # Combine targets
        batch_targets = {
            'boxes': [],
            'labels': [],
            'image_id': []
        }
        
        for idx, target in enumerate(targets):
            batch_targets['boxes'].append(target['boxes'])
            batch_targets['labels'].append(target['labels'])
            batch_targets['image_id'].append(target['image_id'])
            
        batch_targets['boxes'] = torch.cat(batch_targets['boxes'])
        batch_targets['labels'] = torch.cat(batch_targets['labels'])
        batch_targets['image_id'] = torch.cat(batch_targets['image_id'])
        batch_targets['batch_idx'] = torch.cat([
            torch.full_like(target['labels'], i)
            for i, target in enumerate(targets)
        ])
        
        return images, batch_targets

class BaseDetectionDataModule(pl.LightningDataModule):
    """Base Lightning Data Module for object detection."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        image_size: int = 640,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_samples_per_epoch_train: int = 1000,
        max_samples_per_epoch_val: int = 200
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_samples_per_epoch_train = max_samples_per_epoch_train
        self.max_samples_per_epoch_val = max_samples_per_epoch_val

    def _get_data_dirs(self, split: str) -> Tuple[str, str]:
        """Get image and label directories for a given split."""
        img_dir = self.data_dir / 'images' / split
        label_dir = self.data_dir / 'labels' / split
        return str(img_dir), str(label_dir)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            train_img_dir, train_label_dir = self._get_data_dirs('train')
            val_img_dir, val_label_dir = self._get_data_dirs('val')

            train_dataset_full = DetectionDataset(
                train_img_dir,
                train_label_dir,
                self.image_size,
                augment=True
            )
            val_dataset_full = DetectionDataset(
                val_img_dir,
                val_label_dir,
                self.image_size,
                augment=False
            )
            
            # Wrap datasets with LimitedDataset
            self.train_dataset = LimitedDataset(train_dataset_full, self.max_samples_per_epoch_train)
            self.val_dataset = LimitedDataset(val_dataset_full, self.max_samples_per_epoch_val)
            
            print(f"\nDataset sizes after limiting:")
            print(f"Training samples per epoch: {len(self.train_dataset)}")
            print(f"Validation samples per epoch: {len(self.val_dataset)}")
            print(f"Expected training batches: {len(self.train_dataset) // self.batch_size}")
            print(f"Expected validation batches: {len(self.val_dataset) // self.batch_size}\n")

            print("train_img_dir", train_img_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=DetectionDataset.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=DetectionDataset.collate_fn
        )
        
    def on_epoch_end(self):
        """Called at the end of every epoch to reshuffle indices"""
        if hasattr(self, 'train_dataset'):
            self.train_dataset.shuffle_indices()
        if hasattr(self, 'val_dataset'):
            self.val_dataset.shuffle_indices()