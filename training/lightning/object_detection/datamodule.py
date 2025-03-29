"""
Base data module for object detection tasks (face and person detection).
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from PIL import Image

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

        # Setup augmentations
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ),
            A.OneOf([
                A.RandomResizedCrop(
                    height=image_size,
                    width=image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.8, 1.2),
                ),
                A.Resize(height=image_size, width=image_size),
            ], p=1.0),
        ] + ([  # Additional augmentations only during training
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RGBShift(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
        ] if augment else []), 
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

        # Normalization as a separate transform (always applied)
        self.normalize = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Load image
        img_file = self.img_files[idx]
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get labels
        labels = self.labels[img_file]
        
        # Apply augmentations
        transformed = self.transform(
            image=img,
            bboxes=labels[:, 1:] if len(labels) else [],
            class_labels=labels[:, 0] if len(labels) else []
        )
        
        img = transformed['image']
        
        # Create target dict
        if len(transformed['bboxes']):
            boxes = np.array(transformed['bboxes'])  # [N, 4]
            classes = np.array(transformed['class_labels'])  # [N]
        else:
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
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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

            self.train_dataset = DetectionDataset(
                train_img_dir,
                train_label_dir,
                self.image_size,
                augment=True
            )
            self.val_dataset = DetectionDataset(
                val_img_dir,
                val_label_dir,
                self.image_size,
                augment=False
            )

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