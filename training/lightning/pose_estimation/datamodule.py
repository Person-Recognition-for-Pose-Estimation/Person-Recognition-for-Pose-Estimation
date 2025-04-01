"""
COCO keypoint dataset module using direct COCO API without FiftyOne dependency.
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
from typing import Optional, Dict, List, Tuple

class COCOKeypointDataset(Dataset):
    """Dataset class for COCO keypoint data using direct COCO API."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        img_size: int = 640,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize COCO keypoint dataset.
        
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
        self.img_dir = self.data_dir / 'images' / f'{split}'
        ann_file = self.data_dir / 'annotations' / f'person_keypoints_{split}2017.json'
        
        # Load COCO annotations
        self.coco = COCO(str(ann_file))
        
        # Get person category ID
        cat_ids = self.coco.getCatIds(catNms=['person'])
        self.person_cat_id = cat_ids[0]
        
        # Get all image IDs that have keypoint annotations
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
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        ))
        
        # Print dataset statistics
        print(f"\nLoaded {split} dataset:")
        print(f"Total images: {len(self.img_ids)}")
        ann_ids = self.coco.getAnnIds(catIds=cat_ids)
        print(f"Total keypoint annotations: {len(ann_ids)}")
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dict containing:
                - image: [3, H, W] tensor
                - keypoints: [N, K, 3] tensor of keypoint coordinates and visibility
                - boxes: [N, 4] tensor of person bounding boxes
                - masks: [N] boolean tensor indicating valid instances
                - image_id: COCO image ID
        """
        # Load image
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self.img_dir / img_info['file_name']
        img = Image.open(str(img_path)).convert('RGB')
        img = np.array(img)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[self.person_cat_id])
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract keypoints and boxes
        keypoints_list = []
        boxes_list = []
        
        for ann in anns:
            # Skip annotations without keypoints
            if 'keypoints' not in ann or len(ann['keypoints']) == 0:
                continue
                
            # Get keypoints [K, 3] array where K is number of keypoints (17 for COCO)
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            
            # Get bounding box [x, y, w, h]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2] format
            box = np.array([x, y, x + w, y + h])
            
            keypoints_list.append(keypoints)
            boxes_list.append(box)
        
        # Convert to numpy arrays with padding if needed
        if len(keypoints_list) == 0:
            # No annotations - create empty arrays
            keypoints = np.zeros((1, 17, 3), dtype=np.float32)
            boxes = np.zeros((1, 4), dtype=np.float32)
            masks = np.zeros(1, dtype=bool)
        else:
            keypoints = np.stack(keypoints_list)
            boxes = np.stack(boxes_list)
            masks = np.ones(len(keypoints_list), dtype=bool)
        
        # Apply transformations
        if self.transform:
            # Transform image and keypoints
            transformed = self.transform(
                image=img,
                keypoints=keypoints[..., :2].reshape(-1, 2),  # [N*K, 2]
                bboxes=boxes
            )
            
            img = transformed['image']  # Already a tensor from ToTensorV2
            
            # Reshape keypoints back to [N, K, 2]
            transformed_keypoints = np.array(transformed['keypoints']).reshape(keypoints.shape[0], -1, 2)
            
            # Combine with visibility from original keypoints
            keypoints = np.dstack([transformed_keypoints, keypoints[..., 2]])
            boxes = np.array(transformed['bboxes'])
        
        return {
            'image': img,
            'keypoints': torch.from_numpy(keypoints).float(),
            'boxes': torch.from_numpy(boxes).float(),
            'masks': torch.from_numpy(masks),
            'image_id': img_id
        }

class PoseEstimationDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for COCO keypoints."""
    
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
        Initialize COCO keypoint data module.
        
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
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        ))
        
        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        ))
    
    def setup(self, stage: Optional[str] = None):
        """Create train/val datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = COCOKeypointDataset(
                data_dir=self.data_dir,
                split='train',
                transform=self.train_transform,
                img_size=self.img_size,
                max_samples=self.max_samples_per_epoch_train
            )
            
            self.val_dataset = COCOKeypointDataset(
                data_dir=self.data_dir,
                split='val',
                transform=self.val_transform,
                img_size=self.img_size,
                max_samples=self.max_samples_per_epoch_val
            )
            
            # Store validation annotations path for evaluation
            self.val_annotations_path = str(Path(self.data_dir) / 'annotations' / 'person_keypoints_val2017.json')
    
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
        """Custom collate function to handle variable size keypoint annotations"""
        images = torch.stack([item['image'] for item in batch])
        image_ids = [item['image_id'] for item in batch]
        
        # Pad keypoints and boxes to same size
        max_instances = max(item['keypoints'].shape[0] for item in batch)
        
        batch_keypoints = []
        batch_boxes = []
        batch_masks = []  # To track valid instances
        
        for item in batch:
            num_instances = item['keypoints'].shape[0]
            
            # Pad keypoints
            keypoints = torch.zeros((max_instances, 17, 3))
            keypoints[:num_instances] = item['keypoints']
            
            # Pad boxes
            boxes = torch.zeros((max_instances, 4))
            boxes[:num_instances] = item['boxes']
            
            # Create mask for valid instances
            mask = torch.zeros(max_instances, dtype=torch.bool)
            mask[:num_instances] = item['masks']
            
            batch_keypoints.append(keypoints)
            batch_boxes.append(boxes)
            batch_masks.append(mask)
        
        keypoints = torch.stack(batch_keypoints)
        boxes = torch.stack(batch_boxes)
        masks = torch.stack(batch_masks)
        
        return {
            'images': images,
            'keypoints': keypoints,
            'boxes': boxes,
            'masks': masks,
            'image_ids': image_ids
        }