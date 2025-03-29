"""
Lightning module for YOLO person detection using custom implementation.
"""
import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from ..utils import compute_iou
from ..yolopt.util import non_max_suppression
from ..face_detection.module_v2 import DetectionMetrics, FaceDetectionModule

class PersonDetectionModule(FaceDetectionModule):
    """
    Person detection module using the same architecture as face detection.
    Inherits from FaceDetectionModule since the logic is identical,
    only the task name and data are different.
    """
    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        """
        Initialize YOLO Lightning Module for person detection
        
        Args:
            model: Combined multi-task model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super().__init__(model, learning_rate, weight_decay)
        self.model.set_task('person_detection')  # Override task name
        
    def configure_optimizers(self):
        """Configure optimizers"""
        # Optimize person detection branch parameters
        params = list(self.model.yolo_person.parameters())
                
        optimizer = Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer