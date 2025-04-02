"""
COCO keypoint dataset module using direct COCO API without FiftyOne dependency.
Implements COCO-standard keypoint handling and evaluation.
"""
import os
from pathlib import Path
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import pytorch_lightning as pl
from pycocotools.coco import COCO  # type: ignore
import numpy as np
from PIL import Image
import albumentations as A  # type: ignore
from albumentations.pytorch import ToTensorV2  # type: ignore
from typing import Optional, Dict, List, Tuple, NamedTuple

# COCO Keypoint Constants
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_FLIP_PAIRS = [
    (1, 2),   # left_eye, right_eye
    (3, 4),   # left_ear, right_ear
    (5, 6),   # left_shoulder, right_shoulder
    (7, 8),   # left_elbow, right_elbow
    (9, 10),  # left_wrist, right_wrist
    (11, 12), # left_hip, right_hip
    (13, 14), # left_knee, right_knee
    (15, 16)  # left_ankle, right_ankle
]

# COCO OKS sigmas - Used for keypoint similarity calculation
COCO_SIGMAS = np.array([
    .026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062,
    .107, .107, .087, .087, .089, .089
], dtype=np.float32)

class PersonInstance(NamedTuple):
    """Data structure for a single person instance with keypoints."""
    keypoints: np.ndarray  # [K, 3] - (x, y, v) for each keypoint
    bbox: np.ndarray      # [4] - (x1, y1, x2, y2)
    area: float          # Original annotation area
    crowd: bool         # Whether this is a crowd annotation
    num_keypoints: int  # Number of labeled keypoints

class COCOKeypointDataset(Dataset):
    """Dataset class for COCO keypoint data using direct COCO API."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        img_size: int = 640,
        max_samples: Optional[int] = None,
        min_keypoints: int = 1,  # Minimum number of keypoints per person
        filter_crowd: bool = True,  # Whether to filter crowd annotations
    ):
        """
        Initialize COCO keypoint dataset.
        
        Args:
            data_dir: Root directory of COCO dataset
            split: Dataset split ('train' or 'val')
            transform: Albumentations transformations
            img_size: Target image size
            max_samples: Maximum number of samples to use
            min_keypoints: Minimum number of keypoints required per person
            filter_crowd: Whether to filter out crowd annotations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.max_samples = max_samples
        self.min_keypoints = min_keypoints
        self.filter_crowd = filter_crowd
        
        # Setup paths
        self.img_dir = self.data_dir / 'images' / f'{split}'
        ann_file = self.data_dir / 'annotations' / f'person_keypoints_{split}2017.json'
        
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        # Load COCO annotations
        self.coco = COCO(str(ann_file))
        
        # Get person category ID
        cat_ids = self.coco.getCatIds(catNms=['person'])
        if not cat_ids:
            raise ValueError("No 'person' category found in dataset")
        self.person_cat_id = cat_ids[0]
        
        # Get valid image IDs (containing person instances with sufficient keypoints)
        self.img_ids = []
        self.ann_ids_by_image = {}  # Cache annotations by image
        
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            
            # Filter annotations
            valid_anns = []
            for ann in anns:
                if (not self.filter_crowd or ann.get('iscrowd', 0) == 0) and \
                   ann.get('num_keypoints', 0) >= self.min_keypoints:
                    valid_anns.append(ann)
            
            if valid_anns:
                self.img_ids.append(img_id)
                self.ann_ids_by_image[img_id] = [ann['id'] for ann in valid_anns]
        
        # Randomly subsample if max_samples is specified
        if max_samples is not None and max_samples < len(self.img_ids):
            selected_indices = np.random.choice(len(self.img_ids), size=max_samples, replace=False)
            self.img_ids = [self.img_ids[i] for i in selected_indices]
        
        # Setup transform with COCO keypoint format and fixed size output
        self.transform = transform or A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                value=0
            ),
            # Add resize to ensure fixed size
            A.Resize(
                height=img_size,
                width=img_size,
                always_apply=True
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            label_fields=['keypoint_labels']
        ))
        
        # Print dataset statistics
        print(f"\nLoaded {split} dataset:")
        print(f"Total images with valid annotations: {len(self.img_ids)}")
        total_instances = sum(len(ids) for ids in self.ann_ids_by_image.values())
        print(f"Total person instances: {total_instances}")
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def _process_annotation(self, ann: Dict) -> PersonInstance:
        """Process a single COCO annotation into our PersonInstance format."""
        # Get keypoints [K, 3] array where K is number of keypoints (17 for COCO)
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        
        # Get bounding box [x, y, w, h] and convert to [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
        
        return PersonInstance(
            keypoints=keypoints.astype(np.float32),
            bbox=bbox,
            area=float(ann['area']),
            crowd=bool(ann.get('iscrowd', 0)),
            num_keypoints=int(ann['num_keypoints'])
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dict containing:
                - image: [3, H, W] tensor
                - keypoints: [N, K, 3] tensor of keypoint coordinates and visibility
                - boxes: [N, 4] tensor of person bounding boxes [x1, y1, x2, y2]
                - areas: [N] tensor of instance areas
                - masks: [N] boolean tensor indicating valid instances
                - image_id: COCO image ID
                - is_crowd: [N] boolean tensor indicating crowd annotations
        """
        # Load image
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self.img_dir / img_info['file_name']
        img = Image.open(str(img_path)).convert('RGB')
        img = np.array(img)
        
        # Get annotations
        ann_ids = self.ann_ids_by_image[img_id]
        instances = [self._process_annotation(self.coco.loadAnns([ann_id])[0]) 
                    for ann_id in ann_ids]
        
        if not instances:
            # Create dummy instance if no valid annotations
            instances = [PersonInstance(
                keypoints=np.zeros((17, 3), dtype=np.float32),
                bbox=np.zeros(4, dtype=np.float32),
                area=0.0,
                crowd=False,
                num_keypoints=0
            )]
        
        # Handle empty instances
        if not instances:
            # Create a single dummy instance with zeros
            keypoints = np.zeros((1, 17, 3), dtype=np.float32)
            boxes = np.zeros((1, 4), dtype=np.float32)
            areas = np.zeros(1, dtype=np.float32)
            is_crowd = np.zeros(1, dtype=bool)
            masks = np.zeros(1, dtype=bool)
            valid_instances = 0
        else:
            # Stack instance data
            keypoints = np.stack([inst.keypoints for inst in instances])
            boxes = np.stack([inst.bbox for inst in instances])
            areas = np.array([inst.area for inst in instances], dtype=np.float32)
            is_crowd = np.array([inst.crowd for inst in instances], dtype=bool)
            masks = np.array([inst.num_keypoints >= self.min_keypoints for inst in instances])
            valid_instances = len(instances)
        
        # Prepare keypoint labels for albumentations
        keypoint_labels = list(range(17)) * max(1, valid_instances)
        
        # Apply transformations
        if self.transform:
            # Flatten keypoints for transformation
            flat_keypoints = keypoints[..., :2].reshape(-1, 2)
            
            transformed = self.transform(
                image=img,
                keypoints=flat_keypoints,
                bboxes=boxes.tolist(),  # Convert to list for albumentations
                keypoint_labels=keypoint_labels
            )
            
            img = transformed['image']  # Already a tensor
            
            # Handle transformed keypoints and boxes
            if transformed['keypoints'] and transformed['bboxes']:
                # Reshape keypoints back to [N, K, 2]
                transformed_keypoints = np.array(transformed['keypoints']).reshape(keypoints.shape[0], -1, 2)
                transformed_boxes = np.array(transformed['bboxes'])
            else:
                # If transformations removed all keypoints/boxes, create dummy data
                transformed_keypoints = np.zeros_like(keypoints[..., :2])
                transformed_boxes = np.zeros_like(boxes)
                masks = np.zeros_like(masks)
            
            # Combine with original visibility
            keypoints = np.dstack([transformed_keypoints, keypoints[..., 2]])
            boxes = transformed_boxes
        
        # Convert to tensors with proper shapes
        keypoints_tensor = torch.from_numpy(keypoints).float()
        boxes_tensor = torch.from_numpy(boxes).float()
        areas_tensor = torch.from_numpy(areas).float()
        masks_tensor = torch.from_numpy(masks)
        is_crowd_tensor = torch.from_numpy(is_crowd)
        
        # Ensure all tensors have at least one instance dimension
        if keypoints_tensor.shape[0] == 0:
            keypoints_tensor = torch.zeros((1, 17, 3), dtype=torch.float32)
            boxes_tensor = torch.zeros((1, 4), dtype=torch.float32)
            areas_tensor = torch.zeros(1, dtype=torch.float32)
            masks_tensor = torch.zeros(1, dtype=torch.bool)
            is_crowd_tensor = torch.zeros(1, dtype=torch.bool)
        
        return {
            'image': img,
            'keypoints': keypoints_tensor,
            'boxes': boxes_tensor,
            'areas': areas_tensor,
            'masks': masks_tensor,
            'is_crowd': is_crowd_tensor,
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
        min_keypoints: int = 1,
        filter_crowd: bool = True,
        debug_mode: bool = False,
    ):
        """
        Initialize COCO keypoint data module.
        
        Args:
            data_dir: Root directory of COCO dataset
            batch_size: Batch size
            num_workers: Number of workers for data loading
            img_size: Target image size
            max_samples_per_epoch_train: Maximum training samples per epoch
            max_samples_per_epoch_val: Maximum validation samples per epoch
            min_keypoints: Minimum number of keypoints required per person
            filter_crowd: Whether to filter out crowd annotations
            debug_mode: If True, enables additional validation and logging
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.max_samples_per_epoch_train = max_samples_per_epoch_train
        self.max_samples_per_epoch_val = max_samples_per_epoch_val
        self.min_keypoints = min_keypoints
        self.filter_crowd = filter_crowd
        
        # Define transforms with COCO keypoint handling
        self.train_transform = A.Compose([
            # Initial resize to maintain aspect ratio
            A.LongestMaxSize(max_size=int(img_size * 1.2)),  # Slightly larger for augmentation
            A.PadIfNeeded(
                min_height=int(img_size * 1.2),
                min_width=int(img_size * 1.2),
                border_mode=0,
                value=0
            ),
            # Data augmentation
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.5),
            # Scale and rotation augmentation
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    border_mode=0,
                    p=1.0
                ),
                A.RandomRotate90(p=1.0),
            ], p=0.3),
            # Final resize to target size
            A.Resize(
                height=img_size,
                width=img_size,
                always_apply=True
            ),
            # Normalize and convert to tensor
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            label_fields=['keypoint_labels']
        ))
        
        self.val_transform = A.Compose([
            # Consistent resizing for validation
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                value=0
            ),
            # Ensure fixed size
            A.Resize(
                height=img_size,
                width=img_size,
                always_apply=True
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            label_fields=['keypoint_labels']
        ))
    
    def setup(self, stage: Optional[str] = None):
        """Create train/val datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = COCOKeypointDataset(
                data_dir=self.data_dir,
                split='train',
                transform=self.train_transform,
                img_size=self.img_size,
                max_samples=self.max_samples_per_epoch_train,
                min_keypoints=self.min_keypoints,
                filter_crowd=self.filter_crowd
            )
            
            self.val_dataset = COCOKeypointDataset(
                data_dir=self.data_dir,
                split='val',
                transform=self.val_transform,
                img_size=self.img_size,
                max_samples=self.max_samples_per_epoch_val,
                min_keypoints=self.min_keypoints,
                filter_crowd=self.filter_crowd
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
            collate_fn=self.collate_fn,
            drop_last=True  # Drop incomplete batches for stable training
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
        # Get device from input tensors (should all be on same device)
        device = batch[0]['image'].device
        
        # Stack images and collect IDs
        images = torch.stack([item['image'] for item in batch])
        image_ids = [item['image_id'] for item in batch]
        
        # Find maximum number of instances
        max_instances = max(max(item['keypoints'].shape[0] for item in batch), 1)
        
        # Initialize tensors for batch (on same device as input)
        batch_size = len(batch)
        keypoints = torch.zeros(batch_size, max_instances, 17, 3, device=device)
        boxes = torch.zeros(batch_size, max_instances, 4, device=device)
        areas = torch.zeros(batch_size, max_instances, device=device)
        masks = torch.zeros(batch_size, max_instances, dtype=torch.bool, device=device)
        is_crowd = torch.zeros(batch_size, max_instances, dtype=torch.bool, device=device)
        
        # Fill tensors
        for i, item in enumerate(batch):
            # Get number of valid instances, ensuring at least 1
            num_instances = max(item['keypoints'].shape[0], 1)
            
            # Handle potentially empty tensors
            if item['keypoints'].shape[0] > 0:
                keypoints[i, :num_instances] = item['keypoints']
                boxes[i, :num_instances] = item['boxes']
                areas[i, :num_instances] = item['areas']
                masks[i, :num_instances] = item['masks']
                is_crowd[i, :num_instances] = item['is_crowd']
            else:
                # Use dummy values for empty instances
                keypoints[i, 0] = torch.zeros(17, 3, device=device)
                boxes[i, 0] = torch.zeros(4, device=device)
                areas[i, 0] = torch.zeros(1, device=device)
                masks[i, 0] = torch.zeros(1, dtype=torch.bool, device=device)
                is_crowd[i, 0] = torch.zeros(1, dtype=torch.bool, device=device)
        
        return {
            'images': images,
            'keypoints': keypoints,
            'boxes': boxes,
            'areas': areas,
            'masks': masks,
            'is_crowd': is_crowd,
            'image_ids': image_ids
        }