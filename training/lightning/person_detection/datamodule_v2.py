"""
Person detection data module using the new implementation.
"""
from ..object_detection.datamodule import BaseDetectionDataModule
from pathlib import Path

class PersonDetectionDataModule(BaseDetectionDataModule):
    """Data module specifically for person detection."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        image_size: int = 640,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_samples_per_epoch_train: int = None,
        max_samples_per_epoch_val: int = None
    ):
        """
        Initialize person detection data module.
        
        Args:
            data_dir: Path to COCO dataset directory
            batch_size: Batch size for training
            image_size: Input image size
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory in data loading
            max_samples_per_epoch_train: Maximum number of training samples per epoch
            max_samples_per_epoch_val: Maximum number of validation samples per epoch
        """
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.max_samples_train = max_samples_per_epoch_train
        self.max_samples_val = max_samples_per_epoch_val
        
    def _get_data_dirs(self, split: str):
        """
        Get image and label directories for COCO person detection.
        Assumes COCO format with train2017/val2017 structure.
        """
        base_dir = Path(self.data_dir)
        split_name = 'train2017' if split == 'train' else 'val2017'
        img_dir = base_dir / 'images' / split_name
        label_dir = base_dir / 'labels' / split_name
        return str(img_dir), str(label_dir)
        
    def train_dataloader(self):
        """Return training dataloader with optional sample limit."""
        loader = super().train_dataloader()
        if self.max_samples_train is not None:
            # Limit dataset size
            loader.dataset.img_files = loader.dataset.img_files[:self.max_samples_train]
        return loader
        
    def val_dataloader(self):
        """Return validation dataloader with optional sample limit."""
        loader = super().val_dataloader()
        if self.max_samples_val is not None:
            # Limit dataset size
            loader.dataset.img_files = loader.dataset.img_files[:self.max_samples_val]
        return loader