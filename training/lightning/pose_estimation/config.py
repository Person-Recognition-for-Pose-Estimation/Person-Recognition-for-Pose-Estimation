"""
Configuration for pose estimation branch.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class PoseEstimationConfig:
    # Model configuration
    backbone: str = 'hrnet-w32'  # or 'hrnet-w48', 'resnet-50', etc.
    pretrained: bool = True
    deconv_layers: int = 3
    num_joints: int = 17  # COCO keypoints
    
    # Training configuration
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 100
    
    # Data configuration
    image_size: List[int] = (256, 256)
    heatmap_size: List[int] = (64, 64)
    sigma: float = 2.0
    use_udp: bool = True  # Unbiased Data Processing
    
    # Loss configuration
    use_ohkm: bool = True  # Online Hard Keypoint Mining
    topk: int = 8
    loss_weight: float = 1.0
    
    # Augmentation configuration
    flip_prob: float = 0.5
    rot_factor: int = 40
    scale_factor: float = 0.5
    
    # Testing configuration
    flip_test: bool = True
    shift_heatmap: bool = True
    
    # Dataset configuration
    max_samples_train: Optional[int] = None
    max_samples_val: Optional[int] = None
    min_keypoints: int = 1
    
    # Evaluation configuration
    pck_thr: float = 0.05
    nms_thr: float = 0.05
    soft_nms: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        assert len(self.image_size) == 2, "image_size must be a tuple of (height, width)"
        assert len(self.heatmap_size) == 2, "heatmap_size must be a tuple of (height, width)"
        assert self.sigma > 0, "sigma must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.topk > 0, "topk must be positive"
        assert 0 <= self.flip_prob <= 1, "flip_prob must be between 0 and 1"
        assert self.rot_factor >= 0, "rot_factor must be non-negative"
        assert self.scale_factor > 0, "scale_factor must be positive"
        assert self.min_keypoints > 0, "min_keypoints must be positive"
        assert 0 <= self.pck_thr <= 1, "pck_thr must be between 0 and 1"
        assert 0 <= self.nms_thr <= 1, "nms_thr must be between 0 and 1"