"""
COCO dataset module using FiftyOne for efficient data management.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import fiftyone as fo
import fiftyone.zoo as foz
from typing import Optional, Dict, List
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FiftyOneCOCODataset(Dataset):
    """
    Dataset class that uses FiftyOne to manage COCO data.
    """
    def __init__(
        self,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        max_samples: int = None,
        classes: Optional[List[str]] = None,
        shuffle: bool = True,
        seed: int = None
    ):
        """
        Initialize FiftyOne COCO dataset.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            transform: Albumentations transformations
            max_samples: Maximum number of samples to load
            classes: List of required classes (e.g., ['person'])
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
        """
        self.transform = transform
        
        # Load dataset through FiftyOne
        # Create unique dataset name
        dataset_name = f"coco-2017-person-{split}-{max_samples}-{seed if seed else 'noseed'}"
        
        # Try to load dataset with this name
        try:
            self.dataset = foz.load_dataset(dataset_name)
            print(f"Loading existing dataset '{dataset_name}'")
        except:
            print(f"Creating new dataset '{dataset_name}'")
            self.dataset = foz.load_zoo_dataset(
                "coco-2017",
                split=split,
                label_types=["detections"],
                classes=["person"],  # Explicitly request person class
                max_samples=max_samples,
                shuffle=shuffle,
                seed=seed,
                only_matching=True,  # Only load samples that contain persons
                dataset_name=dataset_name
            )
        
        # Convert to PyTorch format
        self.samples = list(self.dataset.iter_samples())
        
        # Print dataset statistics
        print(f"\nLoaded {split} dataset:")
        print(f"Total samples: {len(self.samples)}")
        person_counts = [len([det for det in sample.ground_truth.detections if det.label == 'person']) 
                        for sample in self.samples]
        print(f"Total person detections: {sum(person_counts)}")
        print(f"Average persons per image: {sum(person_counts)/len(self.samples):.1f}")
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dict containing:
                - image: [C, H, W] tensor
                - boxes: [N, 4] tensor of bbox coordinates (x1, y1, x2, y2)
                - labels: [N] tensor of class labels
                - image_id: COCO image ID
        """
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample.filepath).convert('RGB')
        
        # Get detections
        boxes = []
        labels = []
        
        # Filter for person detections
        person_detections = [det for det in sample.ground_truth.detections if det.label == 'person']
        
        for det in person_detections:
            # FiftyOne stores boxes as [x, y, w, h]
            # Convert to [x1, y1, x2, y2]
            x, y, w, h = det.bounding_box
            boxes.append([x, y, x + w, y + h])
            labels.append(0)  # 0 for person class
            
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=np.array(img),
                bboxes=boxes,
                labels=labels
            )
            img = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
        
        # Convert to tensors
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        boxes = torch.from_numpy(boxes)
        labels = torch.from_numpy(labels)
        
        return {
            'image': img,
            'boxes': boxes,
            'labels': labels,
            'image_id': sample.id
        }

class FiftyOneCOCODataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for COCO using FiftyOne.
    """
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        train_samples: int = 1000,
        val_samples: int = 200,
        classes: Optional[List[str]] = None,
        img_size: int = 640,
    ):
        """
        Initialize COCO data module.
        
        Args:
            batch_size: Batch size
            num_workers: Number of workers for data loading
            train_samples: Number of training samples to use
            val_samples: Number of validation samples to use
            classes: List of required classes (e.g., ['person'])
            img_size: Target image size
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.classes = classes or ['person']  # Default to person detection
        self.img_size = img_size
        
        # Define transforms
        self.train_transform = A.Compose([
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
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
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
        
    def setup(self, stage: Optional[str] = None):
        """Create train/val datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = FiftyOneCOCODataset(
                split='train',
                transform=self.train_transform,
                max_samples=self.train_samples,
                classes=self.classes,
                shuffle=True
            )
            
            self.val_dataset = FiftyOneCOCODataset(
                split='validation',
                transform=self.val_transform,
                max_samples=self.val_samples,
                classes=self.classes,
                shuffle=True
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