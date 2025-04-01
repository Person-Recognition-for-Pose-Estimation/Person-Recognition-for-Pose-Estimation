import os
from pathlib import Path
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import pytorch_lightning as pl
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Dict, List
import random
import numpy as np

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
        print(f"Found {len(self.identity_folders)} identity folders in {self.root_dir}")
        
        # Create identity to index mapping
        self.identity_to_idx = {folder.name: idx for idx, folder in enumerate(self.identity_folders)}
        
        if len(self.identity_folders) == 0:
            raise ValueError(f"No identity folders found in {self.root_dir}. Expected structure: identity_folder/image.jpg")
        
        # Get all image paths and labels
        self.samples = []
        total_images = 0
        for folder in self.identity_folders:
            identity_idx = self.identity_to_idx[folder.name]
            folder_images = list(folder.glob("*.jpg"))  # Assuming jpg format
            total_images += len(folder_images)
            for img_path in folder_images:
                self.samples.append((str(img_path), identity_idx))
        
        print(f"Total identities: {len(self.identity_folders)}")
        print(f"Total images: {total_images}")
        print(f"Average images per identity: {total_images / len(self.identity_folders):.1f}")
        
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
        
        # Convert BGR to RGB if needed (AdaFace expects BGR)
        img = img[..., ::-1]  # RGB to BGR
        
        # Apply transforms
        transformed = self.transform(image=img)
        img = transformed['image']  # Already a tensor from ToTensorV2
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
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
        max_samples_per_epoch_train: int = 1000,
        max_samples_per_epoch_val: int = 200,
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
        self.max_samples_per_epoch_train = max_samples_per_epoch_train
        self.max_samples_per_epoch_val = max_samples_per_epoch_val
        
        # Define transforms
        self.train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            # A.OneOf([
            #     A.RandomBrightness(limit=0.1, p=1),
            #     A.RandomContrast(limit=0.1, p=1),
            # ], p=0.3),
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
            # Create full dataset - point to the imgs subdirectory
            imgs_dir = os.path.join(self.data_dir, 'imgs')
            if not os.path.exists(imgs_dir):
                raise ValueError(f"Images directory not found at {imgs_dir}")
            
            print(f"Loading face recognition dataset from: {imgs_dir}")
            full_dataset = ImageFolderDataset(
                imgs_dir,
                transform=self.train_transform,
                image_size=self.image_size
            )
            
            # Split into train/val
            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size
            
            # Create train/val splits
            train_dataset_full, val_dataset_full = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Override val dataset transform
            val_dataset_full.dataset.transform = self.val_transform
            
            # Wrap datasets with LimitedDataset
            self.train_dataset = LimitedDataset(train_dataset_full, self.max_samples_per_epoch_train)
            self.val_dataset = LimitedDataset(val_dataset_full, self.max_samples_per_epoch_val)
            
            print(f"\nDataset sizes after limiting:")
            print(f"Training samples per epoch: {len(self.train_dataset)}")
            print(f"Validation samples per epoch: {len(self.val_dataset)}")
            print(f"Expected training batches: {len(self.train_dataset) // self.batch_size}")
            print(f"Expected validation batches: {len(self.val_dataset) // self.batch_size}\n")
    
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
    
    def on_epoch_end(self):
        """Called at the end of every epoch to reshuffle indices"""
        if hasattr(self, 'train_dataset'):
            self.train_dataset.shuffle_indices()
        if hasattr(self, 'val_dataset'):
            self.val_dataset.shuffle_indices()

    @property
    def num_classes(self) -> int:
        """Return number of classes"""
        if not hasattr(self, 'train_dataset'):
            return None
        return self.train_dataset.dataset.dataset.num_classes  # Access through LimitedDataset wrapper