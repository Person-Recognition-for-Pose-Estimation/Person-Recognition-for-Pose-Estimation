"""
COCO dataset module specifically for person detection, using direct COCO API without FiftyOne.
"""
import os
from pathlib import Path
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import pytorch_lightning as pl
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Dict, List

class COCOPersonDataset(Dataset):
    """Dataset class specifically for COCO person detection."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        img_size: int = 640,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize COCO person dataset.
        
        Args:
            data_dir: Root directory of COCO dataset
            split: Dataset split ('train' or 'val')
            transform: Albumentations transformations
            img_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.max_samples = max_samples
        
        # Setup paths
        self.img_dir = self.data_dir / 'images' / f'{split}2017'
        ann_file = self.data_dir / 'annotations' / f'person_instances_{split}2017.json'
        
        # Load COCO annotations
        self.coco = COCO(str(ann_file))
        
        # Get person category ID
        cat_ids = self.coco.getCatIds(catNms=['person'])
        self.person_cat_id = cat_ids[0]
        
        # Get all image IDs that have person annotations
        self.img_ids = self.coco.getImgIds(catIds=cat_ids)
        
        # Randomly subsample if max_samples is specified
        if max_samples is not None and max_samples < len(self.img_ids):
            self.img_ids = np.random.choice(self.img_ids, size=max_samples, replace=False)
        
        # Setup transform
        self.transform = transform or A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
            ),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dict containing:
                - image: [3, H, W] tensor
                - boxes: [N, 4] tensor of bbox coordinates (x1, y1, x2, y2)
                - labels: [N] tensor of class labels (all 0 for person)
                - image_id: COCO image ID
        """
        # Load image
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self.img_dir / img_info['file_name']
        img = Image.open(str(img_path)).convert('RGB')
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[self.person_cat_id])
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract boxes
        boxes = []
        for ann in anns:
            # Skip crowd annotations
            if ann.get('iscrowd', 0):
                continue
            
            # Get bbox coordinates
            x, y, w, h = ann['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_info['width'], x + w)
            y2 = min(img_info['height'], y + h)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            boxes.append([x1, y1, x2, y2])
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.zeros(len(boxes), dtype=np.int64)  # All 0 for person class
        
        # Apply transforms
        transformed = self.transform(
            image=np.array(img),
            bboxes=boxes if len(boxes) > 0 else np.zeros((0, 4)),
            labels=labels
        )
        
        # Convert to tensors
        img = transformed['image']  # Already a tensor from ToTensorV2
        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        labels = torch.tensor(transformed['labels'], dtype=torch.long)
        
        return {
            'image': img,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id
        }

class PersonDetectionDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for COCO person detection."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        img_size: int = 640,
        max_samples_per_epoch_train: Optional[int] = None,
        max_samples_per_epoch_val: Optional[int] = None,
    ):
        """
        Initialize COCO person data module.
        
        Args:
            data_dir: Root directory of COCO dataset
            batch_size: Batch size
            num_workers: Number of workers for data loading
            img_size: Target image size
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.max_samples_per_epoch_train = max_samples_per_epoch_train
        self.max_samples_per_epoch_val = max_samples_per_epoch_val
        
        # Define transforms
        self.train_transform = A.Compose([
            A.RandomResizedCrop(
                size=(img_size, img_size),  # (height, width)
                scale=(0.8, 1.0),
                ratio=(0.8, 1.2),
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
        
        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
            ),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
    
    def setup(self, stage: Optional[str] = None):
        """Create train/val datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = COCOPersonDataset(
                data_dir=self.data_dir,
                split='train',
                transform=self.train_transform,
                img_size=self.img_size,
                max_samples=self.max_samples_per_epoch_train
            )
            
            self.val_dataset = COCOPersonDataset(
                data_dir=self.data_dir,
                split='val',
                transform=self.val_transform,
                img_size=self.img_size,
                max_samples=self.max_samples_per_epoch_val
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function to handle variable size boxes and labels"""
        images = torch.stack([item['image'] for item in batch])
        image_ids = [item['image_id'] for item in batch]
        
        # Pad boxes and labels to same size
        max_boxes = max(item['boxes'].shape[0] for item in batch)
        
        batch_boxes = []
        batch_labels = []
        
        for item in batch:
            num_boxes = item['boxes'].shape[0]
            if num_boxes == 0:
                # Handle empty annotations
                boxes = torch.zeros((max_boxes, 4))
                labels = torch.zeros(max_boxes, dtype=torch.long)
            else:
                # Pad with zeros
                boxes = torch.zeros((max_boxes, 4))
                boxes[:num_boxes] = item['boxes']
                
                labels = torch.zeros(max_boxes, dtype=torch.long)
                labels[:num_boxes] = item['labels']
            
            batch_boxes.append(boxes)
            batch_labels.append(labels)
        
        boxes = torch.stack(batch_boxes)
        labels = torch.stack(batch_labels)
        
        return {
            'images': images,
            'boxes': boxes,
            'labels': labels,
            'image_ids': image_ids
        }