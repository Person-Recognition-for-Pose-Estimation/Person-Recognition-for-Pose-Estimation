"""
Lightning module for YOLO COCO object detection training.
"""
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from ultralytics.utils.metrics import DetMetrics
from ..ultralytics_wrapper import UltralyticsTrainerWrapper
from ultralytics.cfg import get_cfg
from ..utils import compute_iou as box_iou
import numpy as np
from typing import Dict, List, Optional, Tuple

class COCOYOLOModule(pl.LightningModule):
    def __init__(
        self,
        model,
        data_cfg: str,
        num_classes: int,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        **ultralytics_args
    ):
        """
        Initialize COCO YOLO Lightning Module
        
        Args:
            model: Combined multi-task model
            data_cfg: Path to YOLO data config file
            num_classes: Number of COCO classes
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            **ultralytics_args: Additional arguments for Ultralytics trainer
        """
        super().__init__()
        self.model = model
        self.model.set_task('object_detection')  # Switch to object detection task
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        
        # Initialize metrics for each class
        self.metrics = DetMetrics(
            save_dir=None,
            plot=False,
            names={i: f'class_{i}' for i in range(num_classes)}
        )
        
        # Configure Ultralytics settings
        self.ultralytics_cfg = get_cfg()
        self.ultralytics_cfg.data = data_cfg
        self.ultralytics_cfg.lr0 = learning_rate
        self.ultralytics_cfg.weight_decay = weight_decay
        self.ultralytics_cfg.nc = num_classes  # Set number of classes
        
        for k, v in ultralytics_args.items():
            setattr(self.ultralytics_cfg, k, v)
            
        self.trainer_wrapper = UltralyticsTrainerWrapper(self.model)
        
        # Track best metrics
        self.best_map = 0.0
        self.best_map_epoch = 0
        
    def setup(self, stage: Optional[str] = None):
        """Setup trainer on fit start"""
        if stage == 'fit':
            self.trainer_wrapper.setup_trainer(self.ultralytics_cfg)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step using Ultralytics trainer
        
        Args:
            batch: Dictionary containing:
                - images: [B, C, H, W]
                - boxes: [B, N, 4]
                - labels: [B, N]
            batch_idx: Index of current batch
        """
        loss_dict = self.trainer_wrapper.train_step(batch)
        
        # Log individual loss components
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=True, sync_dist=True)
            
        return loss_dict['loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step
        
        Args:
            batch: Dictionary containing:
                - images: [B, C, H, W]
                - boxes: [B, N, 4]
                - labels: [B, N]
            batch_idx: Index of current batch
        """
        preds, targets, loss_dict = self.trainer_wrapper.validation_step(batch)
        if preds is None:
            return
            
        # Log validation loss components
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, prog_bar=True, sync_dist=True)
            
        # Process predictions for metrics
        if isinstance(preds, (list, tuple)) and len(preds) > 0:
            # Get the first output tensor which contains detections
            detections = preds[0]  # [batch_size, num_anchors, num_classes + 5]
            
            # Extract boxes, scores, and class predictions
            boxes = detections[..., :4]  # [batch_size, num_anchors, 4]
            scores = detections[..., 4]  # [batch_size, num_anchors]
            class_scores = detections[..., 5:]  # [batch_size, num_anchors, num_classes]
            
            # Get predicted classes
            pred_cls = torch.argmax(class_scores, dim=-1)  # [batch_size, num_anchors]
            
            # Get target information
            target_boxes = targets['boxes'].to(boxes.device)
            target_cls = targets['labels'].to(boxes.device)
            
            # Calculate IoU between predictions and targets
            boxes_flat = boxes.reshape(-1, 4)
            target_boxes_flat = target_boxes.reshape(-1, 4)
            iou = box_iou(boxes_flat, target_boxes_flat)
            max_iou, _ = iou.max(dim=1)
            
            # True positives are predictions with IoU > 0.5 and correct class
            tp = ((max_iou > 0.5) & (pred_cls.reshape(-1) == target_cls.reshape(-1))).float()
            tp = tp.view_as(scores)
            
            # Update metrics
            self.metrics.process(tp, scores, pred_cls, target_cls)
    
    def on_validation_epoch_end(self):
        """Called at the end of validation"""
        results = self.metrics.results_dict
        maps = self.metrics.maps
        fitness = self.metrics.fitness
        
        # Log overall metrics
        self.log("val/fitness", fitness, prog_bar=True)
        
        # Log mAP for different IoU thresholds
        for k, v in maps.items():
            self.log(f"val/mAP{k}", v, prog_bar=True)
            
            # Track best mAP
            if k == '50-95' and v > self.best_map:
                self.best_map = v
                self.best_map_epoch = self.current_epoch
                self.log("val/best_mAP", self.best_map)
                self.log("val/best_mAP_epoch", self.best_map_epoch)
        
        # Log detailed metrics
        for k, v in results.items():
            self.log(f"val/{k}", v, prog_bar=True)
        
        # Reset metrics for next epoch
        self.metrics = DetMetrics(
            save_dir=None,
            plot=False,
            names={i: f'class_{i}' for i in range(self.num_classes)}
        )
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Get trainable parameters
        params = list(self.model.yolo.adapter.parameters()) + \
                list(self.model.yolo.yolo.parameters())
                
        # Create optimizer
        optimizer = Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.trainer.train_dataloader),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }