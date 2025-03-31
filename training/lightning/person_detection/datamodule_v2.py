"""
Person detection data module using the new implementation.
"""
from ..object_detection.datamodule import BaseDetectionDataModule
from pathlib import Path

class PersonDetectionDataModule(BaseDetectionDataModule):
    """Data module specifically for Person detection."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        image_size: int = 640,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize Person detection data module.
        
        Args:
            data_dir: Path to Person detection dataset directory
            batch_size: Batch size for training
            image_size: Input image size
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory in data loading
        """
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
    def _get_data_dirs(self, split: str):
        """
        Get image and label directories for Person detection dataset.
        Assumes WIDER Person format with 'images' and 'labels' subdirectories.
        """
        base_dir = Path(self.data_dir)
        img_dir = base_dir / 'images' / split
        label_dir = base_dir / 'labels' / split
        print("img_dir", img_dir)
        return str(img_dir), str(label_dir)