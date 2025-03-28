import os
from pathlib import Path
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import pytorch_lightning as pl
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Dict, List
import numpy as np

class ImageFolderDataset(Dataset):
    """Dataset for face recognition using standard image folders."""
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        image_size: int = 112,  # AdaFace default
    ):
        """
        Initialize face recognition dataset.
        
        Args:
            root_dir: Root directory containing identity folders
            transform: Albumentations transformations
            image_size: Target image size (default 112 for AdaFace)
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        
        # Get all identity folders
        self.identity_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        # Create identity to index mapping
        self.identity_to_idx = {folder.name: idx for idx, folder in enumerate(self.identity_folders)}
        
        # Get all image paths and labels
        self.samples = []
        for folder in self.identity_folders:
            identity_idx = self.identity_to_idx[folder.name]
            for img_path in folder.glob("*.jpg"):  # Assuming jpg format
                self.samples.append((str(img_path), identity_idx))
        
        # Setup transform
        self.transform = transform or A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dict containing:
                - image: [3, H, W] tensor
                - label: class index
        """
        img_path, label = self.samples[idx]
        
        # Load and convert image
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Apply transforms
        transformed = self.transform(image=img)
        img = transformed['image']  # Already a tensor from ToTensorV2
        
        return {
            'image': img,
            'label': label
        }
    
    @property
    def num_classes(self) -> int:
        """Return number of identity classes."""
        return len(self.identity_folders)

class FaceRecognitionDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for face recognition."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 112,
        val_split: float = 0.1,
        **kwargs
    ):
        """
        Initialize Face Recognition DataModule
        
        Args:
            data_dir: Path to dataset directory containing identity folders
            batch_size: Number of samples per batch
            num_workers: Number of workers for data loading
            image_size: Target image size
            val_split: Fraction of data to use for validation
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        
        # Define transforms
        self.train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightness(limit=0.1, p=1),
                A.RandomContrast(limit=0.1, p=1),
            ], p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
    
    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets"""
        if stage == 'fit' or stage is None:
            # Create full dataset
            full_dataset = ImageFolderDataset(
                self.data_dir,
                transform=self.train_transform,
                image_size=self.image_size
            )
            
            # Split into train/val
            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Override val dataset transform
            self.val_dataset.dataset.transform = self.val_transform
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    @property
    def num_classes(self) -> int:
        """Return number of classes"""
        if not hasattr(self, 'train_dataset'):
            return None
        return self.train_dataset.dataset.num_classes