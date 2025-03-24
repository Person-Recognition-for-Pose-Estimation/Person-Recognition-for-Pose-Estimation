"""
COCO keypoint dataset module using FiftyOne for efficient data management.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import fiftyone as fo
import fiftyone.zoo as foz
from typing import Optional, Dict, List, Tuple
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FiftyOneCOCOKeypointDataset(Dataset):
    """
    Dataset class that uses FiftyOne to manage COCO keypoint data.
    """
    def __init__(
        self,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        max_samples: int = None,
        shuffle: bool = True,
        seed: int = None,
        img_size: int = 640
    ):
        """
        Initialize FiftyOne COCO keypoint dataset.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            transform: Albumentations transformations
            max_samples: Maximum number of samples to load
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            img_size: Target image size
        """
        self.transform = transform
        self.img_size = img_size
        # seed = 1
        
        # Load dataset through FiftyOne
        # Create unique dataset name
        dataset_name = f"coco-2017-keypoints-{split}-{max_samples}-{seed if seed else 'noseed'}"

        dataset_dir = "~/fiftyone/coco-2017/validation"
        labels_path = "~/fiftyone/coco-2017/raw/person_keypoints_val2017.json"
        
        # Try to load dataset with this name
        try:
            self.dataset = foz.load_dataset(dataset_name)
            print(f"Loading existing dataset '{dataset_name}'")
        except:
            print(f"Creating new dataset '{dataset_name}'")
            self.dataset = foz.load_zoo_dataset(
                "coco-2017",
                # dataset_type = fo.types.COCODetectionDataset,
                split=split,
                label_types=["detections", "keypoints"],  # Request keypoint annotations
                classes=["person"],  # Only load person class
                max_samples=max_samples,
                shuffle=shuffle,
                seed=seed,
                only_matching=True,  # Only load samples with keypoint annotations
                dataset_name=dataset_name,
                # dataset_dir = dataset_dir,
                labels_path = labels_path
            )
        
            # self.dataset = fo.Dataset.from_dir(
            #     dataset_type = fo.types.COCODetectionDataset,
            #     # "coco-2017",
            #     # split=split,
            #     label_types=["detections", "segmentations", "keypoints"],  # Request keypoint annotations
            #     # classes=["person"],  # Only load person class
            #     # max_samples=max_samples,
            #     # shuffle=shuffle,
            #     # seed=seed,
            #     # only_matching=True,  # Only load samples with keypoint annotations
            #     # dataset_name=dataset_name,
            #     dataset_dir = dataset_dir,
            #     labels_path = labels_path
            # )
        
        # Convert to PyTorch format
        self.samples = list(self.dataset.iter_samples())
        
        # Print dataset statistics
        print(f"\nLoaded {split} dataset:")
        print(f"Total samples: {len(self.samples)}")
        keypoint_counts = [len(kp.keypoints) for sample in self.samples 
                         for kp in sample.ground_truth.keypoints]
        print(f"Total keypoint annotations: {sum(keypoint_counts)}")
        print(f"Average keypoints per person: {np.mean(keypoint_counts):.1f}")
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dict containing:
                - image: [C, H, W] tensor
                - keypoints: [N, K, 3] tensor of keypoint coordinates and visibility
                - boxes: [N, 4] tensor of person bounding boxes
                - image_id: COCO image ID
        """
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample.filepath).convert('RGB')
        img = np.array(img)
        
        # Get keypoint annotations
        keypoints_list = []
        boxes_list = []
        
        for kp_ann in sample.ground_truth.keypoints:
            # Get keypoints [K, 3] array where K is number of keypoints (17 for COCO)
            # Each keypoint is [x, y, visibility]
            keypoints = np.array(kp_ann.keypoints).reshape(-1, 3)
            
            # Get bounding box [x, y, w, h]
            bbox = kp_ann.bounding_box
            x, y, w, h = bbox
            # Convert to [x1, y1, x2, y2] format
            box = np.array([x, y, x + w, y + h])
            
            keypoints_list.append(keypoints)
            boxes_list.append(box)
            
        # Convert to numpy arrays with padding if needed
        if len(keypoints_list) == 0:
            # No annotations - create empty arrays
            keypoints = np.zeros((1, 17, 3), dtype=np.float32)
            boxes = np.zeros((1, 4), dtype=np.float32)
        else:
            keypoints = np.stack(keypoints_list)
            boxes = np.stack(boxes_list)
            
        # Apply transformations if provided
        if self.transform:
            # Create additional keypoint format for albumentations
            transformed = self.transform(
                image=img,
                keypoints=keypoints[..., :2].reshape(-1, 2),  # [N*K, 2]
                bboxes=boxes
            )
            img = transformed['image']
            
            # Reshape keypoints back to [N, K, 2]
            transformed_keypoints = np.array(transformed['keypoints']).reshape(keypoints.shape[0], -1, 2)
            
            # Combine with visibility from original keypoints
            keypoints = np.dstack([transformed_keypoints, keypoints[..., 2]])
            boxes = np.array(transformed['bboxes'])
        else:
            # Just resize and normalize image
            img = Image.fromarray(img).resize((self.img_size, self.img_size), Image.BILINEAR)
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            
            # Normalize coordinates to [0, 1]
            h, w = img.shape[1:]
            keypoints[..., 0] /= w
            keypoints[..., 1] /= h
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h
        
        return {
            'image': img,
            'keypoints': torch.from_numpy(keypoints).float(),
            'boxes': torch.from_numpy(boxes).float(),
            'image_id': sample.id
        }

class FiftyOneCOCOKeypointDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for COCO keypoints using FiftyOne.
    """
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        train_samples: int = 1000,
        val_samples: int = 200,
        img_size: int = 640,
    ):
        """
        Initialize COCO keypoint data module.
        
        Args:
            batch_size: Batch size
            num_workers: Number of workers for data loading
            train_samples: Number of training samples to use
            val_samples: Number of validation samples to use
            img_size: Target image size
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.img_size = img_size
        
        # Define transforms
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        self.val_transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
    def setup(self, stage: Optional[str] = None):
        """Create train/val datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = FiftyOneCOCOKeypointDataset(
                split='train',
                transform=self.train_transform,
                max_samples=self.train_samples,
                shuffle=True,
                img_size=self.img_size
            )
            
            self.val_dataset = FiftyOneCOCOKeypointDataset(
                split='validation',
                transform=self.val_transform,
                max_samples=self.val_samples,
                shuffle=True,
                img_size=self.img_size
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
            mask[:num_instances] = True
            
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
            'masks': masks,  # To identify valid instances
            'image_ids': image_ids
        }