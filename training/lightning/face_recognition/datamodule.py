"""
DataModule for AdaFace face recognition training.
"""
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict
import mxnet as mx
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.data import Dataset

class MXFaceDataset(Dataset):
    def __init__(self, root_dir: str, rec_path: str, idx_path: str, transform=None):
        """
        Initialize MXNet face dataset reader
        
        Args:
            root_dir: Root directory containing the dataset
            rec_path: Path to .rec file
            idx_path: Path to .idx file
            transform: Optional transforms to apply
        """
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        
        # Load MXNet record file
        self.record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
        
        # Read header
        self.header0 = mx.recordio.unpack(self.record.read_idx(0))
        self.imgidx = np.array(range(1, int(self.header0.label[0])))
        
        # Create label mapping
        self.idx2name = {}
        self.name2idx = {}
        self.classes = []
        
        # Parse header to get label information
        for idx in self.imgidx:
            header = mx.recordio.unpack(self.record.read_idx(idx))
            label = int(header.label)
            if label not in self.idx2name:
                self.idx2name[label] = len(self.classes)
                self.name2idx[len(self.classes)] = label
                self.classes.append(label)
                
    def __getitem__(self, index):
        idx = self.imgidx[index]
        record = mx.recordio.unpack(self.record.read_idx(idx))
        header, img = record.header, record.img
        
        # Convert label to consecutive index
        label = int(header.label)
        label = self.idx2name[label]
        
        # Convert image from string to numpy array
        img = mx.image.imdecode(img).asnumpy()
        
        # Convert to RGB if needed
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Convert to PIL Image for transforms
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
        
    def __len__(self):
        return len(self.imgidx)
    
    @property
    def num_classes(self):
        return len(self.classes)

class FaceRecognitionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/Person-Recognition-for-Pose-Estimation/dataset_folders/ada_face",
        train_rec: str = "train.rec",
        train_idx: str = "train.idx",
        val_rec: str = "val.rec",
        val_idx: str = "val.idx",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        Initialize Face Recognition DataModule
        
        Args:
            data_dir: Path to dataset directory
            train_rec: Name of training .rec file
            train_idx: Name of training .idx file
            val_rec: Name of validation .rec file
            val_idx: Name of validation .idx file
            batch_size: Number of samples per batch
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory in data loading
        """
        super().__init__()
        self.data_dir = data_dir
        self.train_rec = os.path.join(data_dir, train_rec)
        self.train_idx = os.path.join(data_dir, train_idx)
        self.val_rec = os.path.join(data_dir, val_rec)
        self.val_idx = os.path.join(data_dir, val_idx)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = MXFaceDataset(
                self.data_dir,
                self.train_rec,
                self.train_idx,
                transform=self.transform
            )
            
            self.val_dataset = MXFaceDataset(
                self.data_dir,
                self.val_rec,
                self.val_idx,
                transform=self.val_transform
            )
            
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    @property
    def num_classes(self):
        """Return number of classes"""
        if self.train_dataset is None:
            return None
        return self.train_dataset.num_classes