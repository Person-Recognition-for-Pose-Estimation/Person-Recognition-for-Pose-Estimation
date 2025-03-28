"""
DataModule for YOLO face detection training.
"""
import os
from pathlib import Path
import pathlib
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader  # type: ignore
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.torch_utils import torch_distributed_zero_first

class FaceDetectionDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize YOLO Face Detection DataModule
        
        Args:
            data_dir: Path to dataset directory
            batch_size: Number of samples per batch
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory in data loading
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        """
        Setup datasets for training and validation
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        with torch_distributed_zero_first(self.trainer.local_rank):
            # Initialize YOLO datasets
            # Load data.yaml configuration
            import yaml
            with open(f"{self.data_dir}/data.yaml", 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Initialize datasets with paths from data.yaml
            self.train_dataset = YOLODataset(
                img_path=f"{self.data_dir}/{data_config['train']}",
                data=data_config,
                batch_size=self.batch_size,
            )
            
            self.val_dataset = YOLODataset(
                img_path=f"{self.data_dir}/{data_config['val']}",
                data=data_config,
                batch_size=self.batch_size,
            )
            
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.train_dataset.collate_fn
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.val_dataset.collate_fn
        )