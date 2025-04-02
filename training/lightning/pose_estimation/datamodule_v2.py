"""
Enhanced pose estimation datamodule using GluonCV components.
"""
import os
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from gluoncv.data.transforms.pose import (
    transform_preds,
    get_affine_transform,
    affine_transform
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

class COCOKeypointsDataset(Dataset):
    """COCO keypoint dataset using GluonCV preprocessing."""
    
    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        image_size: List[int],
        heatmap_size: List[int],
        sigma: float = 2,
        use_udp: bool = True,
        is_train: bool = True,
        min_keypoints: int = 1
    ):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.use_udp = use_udp
        self.is_train = is_train
        self.min_keypoints = min_keypoints
        
        # Initialize COCO api
        self.coco = COCO(ann_file)
        
        # Get all images containing people
        self.img_ids = self.coco.getImgIds(catIds=self.coco.getCatIds(['person']))
        
        # Load dataset
        self.data_info = self._load_annotations()
        
        # Set up transforms
        self.transform = self._build_transforms()
        
    def _build_transforms(self) -> A.Compose:
        """Build albumentations transforms."""
        if self.is_train:
            transform = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.Blur(blur_limit=3, p=0.3),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=30,
                    border_mode=0,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=False,
                label_fields=['keypoint_labels']
            ))
        else:
            transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=False,
                label_fields=['keypoint_labels']
            ))
            
        return transform
        
    def _load_annotations(self) -> List[dict]:
        """Load annotations from COCO format."""
        data_list = []
        
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.coco.getCatIds(['person']))
            anns = self.coco.loadAnns(ann_ids)
            
            if len(anns) == 0:
                continue
                
            # Load image info
            img_info = self.coco.loadImgs([img_id])[0]
            
            for ann in anns:
                # Skip annotations with too few keypoints
                if ann['num_keypoints'] < self.min_keypoints:
                    continue
                    
                # Get keypoint info
                keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                
                # Get bbox and center
                bbox = ann['bbox']  # [x, y, w, h]
                center = np.array([bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5])
                scale = np.array([bbox[2], bbox[3]])
                
                # Adjust center/scale for better cropping
                aspect_ratio = self.image_size[0] / self.image_size[1]
                if aspect_ratio > 1:
                    center[0] = center[0] + (bbox[2] * 0.5 * (aspect_ratio - 1))
                else:
                    center[1] = center[1] + (bbox[3] * 0.5 * (1 / aspect_ratio - 1))
                    
                # Create instance
                instance = {
                    'image_file': os.path.join(self.img_prefix, img_info['file_name']),
                    'center': center,
                    'scale': scale,
                    'bbox': bbox,
                    'keypoints': keypoints[:, :2],  # [x, y]
                    'visibility': keypoints[:, 2],   # visibility
                    'area': ann['area'],
                    'image_id': img_id,
                    'bbox_id': len(data_list)
                }
                
                data_list.append(instance)
                
        return data_list
        
    def _generate_target(
        self,
        joints: np.ndarray,
        joints_vis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate target heatmap using GluonCV's approach."""
        target = np.zeros((len(joints), self.heatmap_size[1], self.heatmap_size[0]), 
                         dtype=np.float32)
        target_weight = joints_vis[:, 0]

        tmp_size = self.sigma * 3

        for joint_id in range(len(joints)):
            feat_stride = np.array(self.image_size) / np.array(self.heatmap_size)
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            
            # Check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            
            if (ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or
                br[0] < 0 or br[1] < 0):
                # If not, just return the current target
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            if target_weight[joint_id] > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
        
    def __len__(self) -> int:
        return len(self.data_info)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get training/testing sample."""
        data_info = self.data_info[idx].copy()
        
        # Load image
        image = np.array(Image.open(data_info['image_file']).convert('RGB'))
        
        # Get keypoints and visibility
        keypoints = data_info['keypoints']  # [N, 2]
        visibility = data_info['visibility']  # [N]
        
        # Get center and scale
        c = data_info['center']
        s = data_info['scale']
        
        # Get affine transform
        trans = get_affine_transform(c, s, 0, self.image_size)
        
        # Apply affine transform to image
        input_size = self.image_size
        inp = cv2.warpAffine(
            image,
            trans,
            (int(input_size[0]), int(input_size[1])),
            flags=cv2.INTER_LINEAR
        )
        
        # Transform keypoints
        for i in range(len(keypoints)):
            if visibility[i] > 0.0:
                keypoints[i] = affine_transform(keypoints[i], trans)
        
        # Apply transforms
        transformed = self.transform(
            image=inp,
            keypoints=keypoints,
            keypoint_labels=list(range(len(keypoints)))
        )
        
        image = transformed['image']  # Now a tensor
        keypoints = np.array(transformed['keypoints'])
        
        # Generate target heatmaps
        target, target_weight = self._generate_target(keypoints, visibility[:, np.newaxis])
        
        # Convert to tensors
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        keypoints = torch.from_numpy(keypoints)
        visibility = torch.from_numpy(visibility)
        center = torch.from_numpy(c)
        scale = torch.from_numpy(s)
        
        return {
            'image': image,
            'target': target,
            'target_weight': target_weight,
            'keypoints': keypoints,
            'visibility': visibility,
            'center': center,
            'scale': scale,
            'image_id': data_info['image_id'],
            'bbox': torch.tensor(data_info['bbox']),
            'area': data_info['area']
        }

class PoseEstimationDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for pose estimation using GluonCV components."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: List[int] = [256, 256],
        heatmap_size: List[int] = [64, 64],
        sigma: float = 2,
        use_udp: bool = True,
        max_samples_train: Optional[int] = None,
        max_samples_val: Optional[int] = None,
        min_keypoints: int = 1
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.use_udp = use_udp
        self.max_samples_train = max_samples_train
        self.max_samples_val = max_samples_val
        self.min_keypoints = min_keypoints
        
    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = COCOKeypointsDataset(
                ann_file=str(self.data_dir / 'annotations' / 'person_keypoints_train2017.json'),
                img_prefix=str(self.data_dir / 'train2017'),
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                sigma=self.sigma,
                use_udp=self.use_udp,
                is_train=True,
                min_keypoints=self.min_keypoints
            )
            
            self.val_dataset = COCOKeypointsDataset(
                ann_file=str(self.data_dir / 'annotations' / 'person_keypoints_val2017.json'),
                img_prefix=str(self.data_dir / 'val2017'),
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                sigma=self.sigma,
                use_udp=self.use_udp,
                is_train=False,
                min_keypoints=self.min_keypoints
            )
            
            # Optionally limit dataset size
            if self.max_samples_train is not None:
                self.train_dataset.data_info = self.train_dataset.data_info[:self.max_samples_train]
            if self.max_samples_val is not None:
                self.val_dataset.data_info = self.val_dataset.data_info[:self.max_samples_val]
                
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )