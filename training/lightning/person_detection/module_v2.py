"""
Lightning module for YOLO person detection training using custom implementation.
"""
import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from ..utils import compute_iou
from yolopt.util import non_max_suppression

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

class PersonDetectionModule(pl.LightningModule):
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
        self.model.set_task('person_detection')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize metrics
        self.train_metrics = DetectionMetrics()
        self.val_metrics = DetectionMetrics()
        
        # Ensure model parameters are float32
        self.model = self.model.float()
        
        # Configure manual optimization
        self.automatic_optimization = False
        
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # # Configure the GradScaler for mixed precision training
        # self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        return optimizer
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def compute_loss(self, pred_boxes, pred_scores, targets):
        """
        Compute detection losses with improved efficiency
        
        Args:
            pred_boxes: Predicted boxes [B, N, 4] or [B, 4, N]
            pred_scores: Predicted class scores [B, N, C] or [B, C, N]
            targets: Dictionary with 'boxes', 'labels', and 'batch_idx'
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        print("\n=== Computing Detection Loss ===")
        print(f"Initial pred_boxes shape: {pred_boxes.shape}")
        print(f"Initial pred_scores shape: {pred_scores.shape}")
        
        # Get target boxes and classes
        gt_boxes = targets['boxes']  # [N, 4]
        gt_classes = targets['labels']  # [N]
        batch_idx = targets['batch_idx']  # [N]
        
        print(f"Ground truth shapes:")
        print(f"- boxes: {gt_boxes.shape}")
        print(f"- classes: {gt_classes.shape}")
        print(f"- batch_idx: {batch_idx.shape}")
        print(f"Unique batch indices: {torch.unique(batch_idx)}")
        
        B = pred_boxes.shape[0]
        total_loss = 0
        num_targets = len(gt_boxes)
        losses = {}
        
        print(f"\nBatch size: {B}, Total targets: {num_targets}")
        
        # Ensure boxes are in [B, N, 4] format
        if pred_boxes.shape[1] == 4:
            print("Transposing predictions to [B, N, 4] format")
            pred_boxes = pred_boxes.transpose(1, 2)  # [B, N, 4]
            pred_scores = pred_scores.transpose(1, 2)  # [B, N, C]
            print(f"After transpose - boxes: {pred_boxes.shape}, scores: {pred_scores.shape}")
        
        # Filter predictions by confidence to reduce computation
        score_threshold = 0.01  # Adjust as needed
        max_scores, _ = pred_scores.max(dim=-1)  # [B, N]
        print(f"\nMax scores shape: {max_scores.shape}")
        print(f"Score range: min={max_scores.min().item():.4f}, max={max_scores.max().item():.4f}")
        
        for b in range(B):
            # print(f"\n--- Processing batch {b} ---")
            # Get batch-specific predictions and targets
            batch_mask = batch_idx == b
            # print(f"Number of targets in batch: {batch_mask.sum().item()}")
            
            if not batch_mask.any():
                print("No targets in this batch, skipping...")
                continue
                
            b_gt_boxes = gt_boxes[batch_mask]
            b_gt_classes = gt_classes[batch_mask]
            # print(f"Ground truth - boxes: {b_gt_boxes.shape}, classes: {b_gt_classes.shape}")
            
            # Filter predictions by confidence
            conf_mask = max_scores[b] > score_threshold
            # print(f"Predictions above threshold: {conf_mask.sum().item()} / {conf_mask.numel()}")
            
            b_pred_boxes = pred_boxes[b, conf_mask]  # [M, 4]
            b_pred_scores = pred_scores[b, conf_mask]  # [M, C]
            # print(f"Filtered predictions - boxes: {b_pred_boxes.shape}, scores: {b_pred_scores.shape}")
            
            if len(b_pred_boxes) == 0 or len(b_gt_boxes) == 0:
                # print("Empty predictions or targets")
                # Handle empty case
                if len(b_pred_boxes) > 0:
                    # print(f"No targets but have {len(b_pred_boxes)} predictions - computing background loss")
                    # No targets but have predictions - all background
                    bg_loss = F.binary_cross_entropy_with_logits(
                        b_pred_scores.max(dim=-1)[0],
                        torch.zeros_like(b_pred_scores.max(dim=-1)[0]),
                        reduction='mean'
                    )
                    total_loss += bg_loss
                    losses[f'bg_loss_{b}'] = bg_loss.item()
                    # print(f"Background loss: {bg_loss.item():.4f}")
                continue
            
            # print("\nComputing IoU matrix...")
            # print(f"Pred boxes shape: {b_pred_boxes.shape}, GT boxes shape: {b_gt_boxes.shape}")
            try:
                # Compute IoU matrix efficiently
                ious = compute_iou(b_pred_boxes, b_gt_boxes)  # [M, K]
                # print(f"IoU matrix shape: {ious.shape}")
                # print(f"IoU range: min={ious.min().item():.4f}, max={ious.max().item():.4f}")
                
                max_ious, max_idx = ious.max(dim=1)  # [M]
                # print(f"Max IoUs shape: {max_ious.shape}")
                # print(f"Max IoU values range: min={max_ious.min().item():.4f}, max={max_ious.max().item():.4f}")
                
                # Positive samples: IoU > 0.5
                pos_mask = max_ious > 0.5
                num_positives = pos_mask.sum().item()
                # print(f"Number of positive matches (IoU > 0.5): {num_positives}")
                
                if pos_mask.any():
                    # print("\nComputing losses for positive matches...")
                    # Box regression loss (CIoU)
                    box_loss = -compute_iou(
                        b_pred_boxes[pos_mask],
                        b_gt_boxes[max_idx[pos_mask]],
                        CIoU=True
                    ).mean()
                    
                    # Classification loss for positive samples
                    cls_loss = F.cross_entropy(
                        b_pred_scores[pos_mask],
                        b_gt_classes[max_idx[pos_mask]]
                    )
                    
                    # Background loss for negative samples
                    bg_loss = F.binary_cross_entropy_with_logits(
                        b_pred_scores[~pos_mask].max(dim=-1)[0],
                        torch.zeros_like(b_pred_scores[~pos_mask].max(dim=-1)[0]),
                        reduction='mean'
                    )
                    
                    batch_loss = box_loss + cls_loss + 0.5 * bg_loss
                    total_loss += batch_loss
                    
                    losses.update({
                        f'box_loss_{b}': box_loss.item(),
                        f'cls_loss_{b}': cls_loss.item(),
                        f'bg_loss_{b}': bg_loss.item()
                    })
                    # print(f"Losses - box: {box_loss.item():.4f}, cls: {cls_loss.item():.4f}, bg: {bg_loss.item():.4f}")
                else:
                    # print("No positive matches, computing background loss only")
                    # No positive samples - treat all as background
                    bg_loss = F.binary_cross_entropy_with_logits(
                        b_pred_scores.max(dim=-1)[0],
                        torch.zeros_like(b_pred_scores.max(dim=-1)[0]),
                        reduction='mean'
                    )
                    total_loss += bg_loss
                    losses[f'bg_loss_{b}'] = bg_loss.item()
                    print(f"Background loss: {bg_loss.item():.4f}")
            except Exception as e:
                # print(f"Error during IoU computation: {str(e)}")
                # print(f"Pred boxes stats - min: {b_pred_boxes.min().item():.4f}, max: {b_pred_boxes.max().item():.4f}")
                # print(f"GT boxes stats - min: {b_gt_boxes.min().item():.4f}, max: {b_gt_boxes.max().item():.4f}")
                raise
        
        avg_loss = total_loss / B
        losses['total_loss'] = avg_loss.item()
        print(f"\nFinal average loss: {avg_loss.item():.4f}")
        
        return avg_loss, losses
    
    def convert_boxes_cxcywh_to_xyxy(self, boxes):
        """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format"""
        cx, cy, w, h = boxes.chunk(4, dim=-1)
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return torch.cat([x1, y1, x2, y2], dim=-1)

    def process_yolo_output(self, predictions, is_training=True):
        """Process YOLO outputs for both training and validation"""
        from yolopt.util import make_anchors, wh2xy
        
        if is_training:
            # During training, YOLO returns a list of feature maps
            # Each feature map has shape [B, 65, H, W] where 65 = 4 (box) + 1 (obj) + 60 (unused)
            all_boxes = []
            all_scores = []
            
            # Calculate strides for each feature map
            input_size = 640  # YOLO default input size
            strides = []
            for feat_map in predictions:
                _, _, H, W = feat_map.shape
                stride = input_size / H
                strides.append(stride)
            
            # Generate anchors
            anchors, strides_tensor = make_anchors(predictions, strides)
            
            # print(f"\nProcessing {len(predictions)} feature maps with strides {strides}")
            # print(f"Generated {len(anchors)} anchors")
            
            for i, feat_map in enumerate(predictions):
                B, _, H, W = feat_map.shape
                # print(f"\nFeature map {i}: shape={feat_map.shape}, stride={strides[i]}")
                
                # Extract predictions
                feat_map = feat_map.permute(0, 2, 3, 1)  # [B, H, W, C]
                feat_map = feat_map.reshape(B, -1, feat_map.shape[-1])  # [B, HW, C]
                
                # Split predictions
                box_preds = feat_map[..., :4]  # [B, HW, 4]
                score_preds = feat_map[..., 4:5]  # [B, HW, 1]
                
                # Get anchors for this feature level
                start_idx = sum(H * W for feat in predictions[:i])
                end_idx = start_idx + H * W
                level_anchors = anchors[start_idx:end_idx]
                level_strides = strides_tensor[start_idx:end_idx]
                
                # Decode boxes
                boxes = torch.cat((
                    level_anchors + box_preds[..., :2] * level_strides,  # xy = anchor + pred * stride
                    torch.exp(box_preds[..., 2:]) * level_strides  # wh = exp(pred) * stride
                ), dim=-1)
                
                # Convert to xyxy format
                boxes = wh2xy(boxes)  # [B, HW, 4]
                scores = torch.sigmoid(score_preds)  # [B, HW, 1]
                
                # Reshape to [B, 4, HW] and [B, 1, HW]
                boxes = boxes.permute(0, 2, 1)
                scores = scores.permute(0, 2, 1)
                
                all_boxes.append(boxes)
                all_scores.append(scores)
            
            # Concatenate predictions from all feature levels
            pred_boxes = torch.cat(all_boxes, dim=2)  # [B, 4, total_grid_points]
            pred_scores = torch.cat(all_scores, dim=2)  # [B, 1, total_grid_points]
            
            print("\nFinal predictions:")
            print(f"Boxes shape: {pred_boxes.shape}")
            print(f"Scores shape: {pred_scores.shape}")
            print(f"Box range: min={pred_boxes.min().item():.4f}, max={pred_boxes.max().item():.4f}")
            print(f"Score range: min={pred_scores.min().item():.4f}, max={pred_scores.max().item():.4f}")
            
            return pred_boxes, pred_scores
        else:
            # During inference, predictions are already processed
            boxes = predictions[:, :4]  # [B, 4, N]
            scores = predictions[:, 4:5]  # [B, 1, N]
            return boxes, scores

    def training_step(self, batch, batch_idx):
        """Single training step"""
        images, targets = batch
        optimizer = self.optimizers()
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast():
            predictions = self.model(images)
            pred_boxes, pred_scores = self.process_yolo_output(predictions, is_training=True)
            
            # Scale boxes to absolute coordinates
            img_size = images.shape[-1]  # Assuming square images
            pred_boxes = pred_boxes * img_size
            targets['boxes'] = targets['boxes'] * img_size
            
            # Debug box coordinates
            print("\nBox coordinate ranges:")
            print(f"Pred boxes - x1: [{pred_boxes[:,:,0].min():.1f}, {pred_boxes[:,:,0].max():.1f}], "
                  f"y1: [{pred_boxes[:,:,1].min():.1f}, {pred_boxes[:,:,1].max():.1f}], "
                  f"x2: [{pred_boxes[:,:,2].min():.1f}, {pred_boxes[:,:,2].max():.1f}], "
                  f"y2: [{pred_boxes[:,:,3].min():.1f}, {pred_boxes[:,:,3].max():.1f}]")
            print(f"GT boxes - x1: [{targets['boxes'][:,0].min():.1f}, {targets['boxes'][:,0].max():.1f}], "
                  f"y1: [{targets['boxes'][:,1].min():.1f}, {targets['boxes'][:,1].max():.1f}], "
                  f"x2: [{targets['boxes'][:,2].min():.1f}, {targets['boxes'][:,2].max():.1f}], "
                  f"y2: [{targets['boxes'][:,3].min():.1f}, {targets['boxes'][:,3].max():.1f}]")
            
            loss, loss_dict = self.compute_loss(pred_boxes, pred_scores, targets)
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(optimizer)
        # self.scaler.update()
        
        # Log metrics
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()})
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        images, targets = batch
        
        # Forward pass with no gradients
        with torch.no_grad():
            predictions = self.model(images)
            pred_boxes, pred_scores = self.process_yolo_output(predictions, is_training=False)
        
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
        params = list(self.model.yolo_person.parameters())
                
        optimizer = Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer