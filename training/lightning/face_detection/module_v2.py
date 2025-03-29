"""
Lightning module for YOLO face detection training using custom implementation.
"""
import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from ..utils import compute_iou
from ..yolopt.util import non_max_suppression

class DetectionMetrics:
    """Simple detection metrics calculator."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.correct = 0
        self.total_pred = 0
        self.total_gt = 0
        self.total_tp = 0
        self.total_fp = 0
        self.ap_scores = []
        
    def update(self, pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes):
        if len(pred_boxes) == 0:
            self.total_gt += len(gt_boxes)
            return
            
        if len(gt_boxes) == 0:
            self.total_fp += len(pred_boxes)
            return
            
        # Calculate IoU between predictions and ground truth
        ious = compute_iou(pred_boxes, gt_boxes)  # [num_pred, num_gt]
        max_ious, max_idx = ious.max(dim=1)
        
        # True positives: IoU > 0.5 and correct class
        correct_class = pred_classes == gt_classes[max_idx]
        tp = (max_ious > 0.5) & correct_class
        
        self.total_tp += tp.sum().item()
        self.total_fp += (~tp).sum().item()
        self.total_gt += len(gt_boxes)
        
        # Store scores for AP calculation
        self.ap_scores.extend([
            (score.item(), tp_i.item())
            for score, tp_i in zip(pred_scores, tp)
        ])
        
    def compute(self):
        # Compute precision and recall
        precision = self.total_tp / (self.total_tp + self.total_fp + 1e-6)
        recall = self.total_tp / (self.total_gt + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        # Compute AP
        if not self.ap_scores:
            ap = 0.0
        else:
            # Sort by confidence
            scores = sorted(self.ap_scores, key=lambda x: x[0], reverse=True)
            tp = torch.tensor([x[1] for x in scores])
            fp = ~tp
            
            # Cumulative sum
            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)
            recalls = tp_cumsum / (self.total_gt + 1e-6)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            # Append start and end points
            recalls = torch.cat((torch.tensor([0]), recalls, torch.tensor([1])))
            precisions = torch.cat((torch.tensor([1]), precisions, torch.tensor([0])))
            
            # Compute mean AP
            ap = torch.trapz(precisions, recalls).item()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP': ap
        }

class FaceDetectionModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        """
        Initialize YOLO Lightning Module
        
        Args:
            model: Combined multi-task model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super().__init__()
        self.model = model
        self.model.set_task('face_detection')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize metrics
        self.train_metrics = DetectionMetrics()
        self.val_metrics = DetectionMetrics()
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def compute_loss(self, pred_boxes, pred_scores, targets):
        """
        Compute detection losses
        
        Args:
            pred_boxes: Predicted boxes [B, 4, HW]
            pred_scores: Predicted class scores [B, C, HW]
            targets: Dictionary with 'boxes', 'labels', and 'batch_idx'
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # Get target boxes and classes
        gt_boxes = targets['boxes']  # [N, 4]
        gt_classes = targets['labels']  # [N]
        batch_idx = targets['batch_idx']  # [N]
        
        B = pred_boxes.shape[0]
        total_loss = 0
        num_targets = len(gt_boxes)
        
        if num_targets == 0:
            # No targets, only background
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_scores.flatten(),
                torch.zeros_like(pred_scores.flatten()),
                reduction='mean'
            )
            return cls_loss, {'cls_loss': cls_loss.item()}
            
        losses = {}
        
        # For each batch item
        for b in range(B):
            # Get predictions and targets for this batch
            batch_mask = batch_idx == b
            if not batch_mask.any():
                continue
                
            b_gt_boxes = gt_boxes[batch_mask]
            b_gt_classes = gt_classes[batch_mask]
            
            b_pred_boxes = pred_boxes[b]  # [4, HW]
            b_pred_scores = pred_scores[b]  # [C, HW]
            
            # Calculate IoU between predictions and targets
            ious = compute_iou(
                b_pred_boxes.transpose(0, 1),  # [HW, 4]
                b_gt_boxes
            )  # [HW, num_targets]
            
            # For each prediction, get the best matching target
            max_ious, max_idx = ious.max(dim=1)  # [HW]
            
            # Positive samples: IoU > 0.5
            pos_mask = max_ious > 0.5
            
            if pos_mask.any():
                # Box regression loss (CIoU)
                box_loss = -compute_iou(
                    b_pred_boxes.transpose(0, 1)[pos_mask],
                    b_gt_boxes[max_idx[pos_mask]],
                    CIoU=True
                ).mean()
                
                # Classification loss
                cls_loss = F.cross_entropy(
                    b_pred_scores[:, pos_mask],
                    b_gt_classes[max_idx[pos_mask]]
                )
            else:
                box_loss = torch.tensor(0.0, device=pred_boxes.device)
                cls_loss = torch.tensor(0.0, device=pred_boxes.device)
            
            # Background loss
            bg_loss = F.binary_cross_entropy_with_logits(
                pred_scores[b].flatten()[~pos_mask],
                torch.zeros_like(pred_scores[b].flatten()[~pos_mask])
            )
            
            batch_loss = box_loss + cls_loss + 0.5 * bg_loss
            total_loss += batch_loss
            
            losses.update({
                f'box_loss_{b}': box_loss.item(),
                f'cls_loss_{b}': cls_loss.item(),
                f'bg_loss_{b}': bg_loss.item()
            })
            
        avg_loss = total_loss / B
        losses['total_loss'] = avg_loss.item()
        
        return avg_loss, losses
    
    def training_step(self, batch, batch_idx):
        """Single training step"""
        images, targets = batch
        
        # Forward pass
        pred_boxes, pred_scores = self.model.yolo_face(images)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(pred_boxes, pred_scores, targets)
        
        # Log metrics
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()})
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        images, targets = batch
        
        # Forward pass
        pred_boxes, pred_scores = self.model.yolo_face(images)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(pred_boxes, pred_scores, targets)
        
        # Log metrics
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()})
        
        # Apply NMS
        pred_boxes = pred_boxes.transpose(1, 2)  # [B, HW, 4]
        pred_scores = pred_scores.transpose(1, 2)  # [B, HW, C]
        predictions = torch.cat([pred_boxes, pred_scores], dim=-1)
        nms_predictions = non_max_suppression(predictions)
        
        # Update metrics
        for i, pred in enumerate(nms_predictions):
            if len(pred) == 0:
                continue
                
            # Get predictions for this batch item
            batch_mask = targets['batch_idx'] == i
            gt_boxes = targets['boxes'][batch_mask]
            gt_classes = targets['labels'][batch_mask]
            
            if len(gt_boxes) == 0:
                continue
                
            # Update metrics
            self.val_metrics.update(
                pred[:, :4],
                pred[:, 4],
                pred[:, 5].long(),
                gt_boxes,
                gt_classes
            )
    
    def on_validation_epoch_end(self):
        """Called at the end of validation"""
        metrics = self.val_metrics.compute()
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})
        self.val_metrics.reset()
    
    def configure_optimizers(self):
        """Configure optimizers"""
        # Optimize both adapter and YOLO model parameters
        params = list(self.model.yolo_face.parameters())
                
        optimizer = Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer