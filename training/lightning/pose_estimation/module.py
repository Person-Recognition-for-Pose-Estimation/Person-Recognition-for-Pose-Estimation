"""
Lightning module for pose estimation using ViTPose.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Optional, List
import torchmetrics
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.
    
    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """
    def __init__(self, use_target_weight=True, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
    
    def forward(self, output, target, target_weight):
        """Forward function.
        
        Args:
            output: Predicted heatmaps [B, K, H, W]
            target: Target heatmaps [B, K, H, W]
            target_weight: Target weights [B, K]
        """
        batch_size = output.size(0)
        num_joints = output.size(1)
        
        # Reshape to [B, K, -1] and split along joints dimension
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        
        loss = 0.
        
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred * target_weight[:, idx].unsqueeze(-1),
                    heatmap_gt * target_weight[:, idx].unsqueeze(-1)
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
        
        return loss / num_joints * self.loss_weight


class ViTPoseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        heatmap_size: tuple = (64, 48),  # From ViTPose config
        sigma: float = 2.0,  # Gaussian sigma for heatmap generation
        keypoint_thresh: float = 0.3,  # Threshold for keypoint visibility
    ):
        """
        Initialize ViTPose Lightning Module
        
        Args:
            model: Combined multi-task model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            heatmap_size: Size of output heatmaps (H, W)
            sigma: Gaussian sigma for heatmap generation
            keypoint_thresh: Confidence threshold for keypoint visibility
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Model and parameters
        self.model = model
        self.model.set_task('pose_estimation')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.keypoint_thresh = keypoint_thresh
        
        # Loss and evaluation
        self.criterion = JointsMSELoss(use_target_weight=True)
        self.eval_predictions = []  # Store predictions during validation
        
    def setup(self, stage: Optional[str] = None):
        """Validate the datamodule has required attributes."""
        if stage == 'fit' or stage == 'validate':
            if not hasattr(self.trainer.datamodule, 'val_annotations_path'):
                raise AttributeError(
                    "DataModule must provide 'val_annotations_path' for COCO evaluation. "
                    "This should point to the COCO keypoint annotations JSON file."
                )
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
        
    def _get_keypoints_from_heatmaps(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Extract keypoint coordinates from heatmaps.
        
        Args:
            heatmaps: [B, K, H, W] Predicted heatmaps
            
        Returns:
            coords: [B, K, 2] Keypoint coordinates in [x, y] format
            scores: [B, K] Confidence scores
        """
        B, K, H, W = heatmaps.shape
        
        # Reshape and find max values
        heatmaps_reshaped = heatmaps.reshape(B, K, -1)
        max_vals, max_idx = heatmaps_reshaped.max(dim=2)
        
        # Convert indices to x,y coordinates in pixel space
        max_idx = max_idx.float()
        x = (max_idx % W)
        y = (max_idx // W)
        
        # Normalize coordinates to [0,1] range
        x = x / W
        y = y / H
        
        # Stack coordinates
        coords = torch.stack([x, y], dim=-1)  # [B, K, 2]
        
        return coords, max_vals
    
    def _generate_target_heatmap(self, keypoints, visibility, size):
        """Generate target heatmap from keypoint coordinates.
        
        Args:
            keypoints: [B, N, K, 2] tensor of keypoint coordinates (normalized)
            visibility: [B, N, K] tensor of keypoint visibility
            size: (H, W) tuple of heatmap size
            
        Returns:
            heatmaps: [B, K, H, W] tensor of target heatmaps
            weights: [B, K] tensor of target weights
        """
        B, N, K, _ = keypoints.shape
        H, W = size
        device = keypoints.device
        
        # Initialize heatmaps and weights
        heatmaps = torch.zeros((B, K, H, W), device=device)
        weights = torch.zeros((B, K), device=device)
        
        # Convert normalized coordinates to pixel coordinates
        keypoints = keypoints.clone()
        keypoints[..., 0] *= W
        keypoints[..., 1] *= H
        
        # Create coordinate grid
        x = torch.arange(W, device=device)
        y = torch.arange(H, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Generate heatmaps for each batch and instance
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    if visibility[b, n, k] > 0:  # Keypoint is labeled
                        mu_x, mu_y = keypoints[b, n, k]
                        
                        # Compute Gaussian
                        gaussian = torch.exp(
                            -((xx - mu_x)**2 + (yy - mu_y)**2) / (2 * self.sigma**2)
                        )
                        
                        # Add to heatmap (take maximum in case of multiple instances)
                        heatmaps[b, k] = torch.maximum(heatmaps[b, k], gaussian)
                        
                        # Update weight (1 for visible, 0.5 for occluded)
                        weights[b, k] = max(weights[b, k], 
                                         1.0 if visibility[b, n, k] == 2 else 0.5)
        
        return heatmaps, weights
    
    def training_step(self, batch, batch_idx):
        """Single training step"""
        # Get inputs and targets
        images = batch['images']
        keypoints = batch['keypoints']  # [B, N, K, 3] - N instances, K keypoints, 3 = (x, y, visibility)
        masks = batch['masks']  # [B, N] boolean mask for valid instances
        
        # Split keypoints into coordinates and visibility
        coords = keypoints[..., :2]  # [B, N, K, 2]
        visibility = keypoints[..., 2]  # [B, N, K]
        
        # Generate target heatmaps and weights
        target_heatmaps, target_weights = self._generate_target_heatmap(
            coords, visibility, self.heatmap_size
        )
        
        # Forward pass through backbone and pose branch
        pred_heatmaps = self.model(images)
        
        # Compute loss
        loss = self.criterion(pred_heatmaps, target_heatmaps, target_weights)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_start(self):
        """Clear stored predictions at the start of validation.
        
        The COCO evaluation process:
        1. During validation_step, we store predictions in COCO format:
           - image_id: ID from COCO dataset
           - category_id: 1 for person
           - keypoints: [x1,y1,v1,x2,y2,v2,...] format where v is visibility
           - score: confidence score
           - bbox: person bounding box [x,y,w,h]
           
        2. At epoch end, we:
           - Load COCO ground truth annotations
           - Convert our predictions to COCO format
           - Run evaluation using pycocotools
           - Log various AP metrics:
             * AP: Average Precision at IoU=.50:.05:.95
             * AP50: AP at IoU=.50
             * AP75: AP at IoU=.75
             * APm: AP for medium objects
             * APl: AP for large objects
        """
        self.eval_predictions = []

    def validation_step(self, batch, batch_idx):
        """Single validation step.
        
        Args:
            batch: Dictionary containing:
                - images: [B, C, H, W] input images
                - keypoints: [B, N, K, 3] keypoint annotations (x, y, visibility)
                - masks: [B, N] boolean mask for valid instances
                - image_ids: [B] COCO image IDs
                - boxes: [B, N, 4] person bounding boxes [x, y, w, h]
            batch_idx: Index of this batch
            
        Returns:
            loss: Validation loss value
        """
        # Check required batch keys
        required_keys = ['images', 'keypoints', 'masks', 'image_ids', 'boxes']
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise KeyError(f"Batch is missing required keys: {missing_keys}")
            
        images = batch['images']
        keypoints = batch['keypoints']  # [B, N, K, 3] - N instances, K keypoints, 3 = (x, y, visibility)
        masks = batch['masks']  # [B, N] boolean mask for valid instances
        image_ids = batch['image_ids']  # COCO image IDs
        boxes = batch['boxes']  # [B, N, 4] person bounding boxes
        
        try:
            # Forward pass
            pred_heatmaps = self.model(images)
            
            # Get predicted keypoint coordinates and scores
            pred_coords, pred_scores = self._get_keypoints_from_heatmaps(pred_heatmaps)
            
            # Generate target heatmaps for loss computation
            target_heatmaps, target_weights = self._generate_target_heatmap(
                keypoints[..., :2], keypoints[..., 2], self.heatmap_size
            )
            
            # Compute loss
            loss = self.criterion(pred_heatmaps, target_heatmaps, target_weights)
            self.log('val_loss', loss, prog_bar=True)
            
            # Store predictions in COCO format for evaluation
            batch_size = len(image_ids)
            for b in range(batch_size):
                valid_instances = masks[b]  # [N] boolean mask
                num_instances = valid_instances.sum().item()
                
                for n in range(num_instances):
                    # Get instance keypoints and box
                    kpts = pred_coords[b, n]  # [K, 2]
                    scores = pred_scores[b, n]  # [K]
                    box = boxes[b, n]  # [4]
                    
                    # Convert to COCO format
                    # COCO keypoint format: [x1,y1,v1,x2,y2,v2,...]
                    keypoints_coco = []
                    for kpt, score in zip(kpts, scores):
                        x, y = kpt.cpu().numpy()
                        # Set visibility based on confidence score
                        v = 2 if score > 0.3 else 1  # 2: visible, 1: not visible
                        keypoints_coco.extend([x, y, v])
                    
                    # Create prediction entry
                    pred = {
                        'image_id': image_ids[b].item(),  # Ensure it's a Python number
                        'category_id': 1,  # person category
                        'keypoints': keypoints_coco,
                        'score': scores.mean().item(),  # keypoint score
                        'bbox': box.cpu().numpy().tolist()  # [x, y, w, h]
                    }
                    
                    self.eval_predictions.append(pred)
                    
            return loss
            
        except Exception as e:
            self.print(f"Error in validation step: {str(e)}")
            raise e
                
    def on_validation_epoch_end(self):
        """Compute COCO metrics at the end of validation."""
        try:
            # Skip if no predictions
            if not self.eval_predictions:
                self.print("No predictions to evaluate")
                return
                
            # Initialize COCO ground truth
            gt_path = self.trainer.datamodule.val_annotations_path
            if not hasattr(self.trainer.datamodule, 'val_annotations_path'):
                raise AttributeError("DataModule must provide 'val_annotations_path' for COCO evaluation")
                
            # Load ground truth annotations
            try:
                coco_gt = COCO(gt_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load COCO annotations from {gt_path}: {str(e)}")
            
            # Create COCO prediction object
            try:
                coco_dt = coco_gt.loadRes(self.eval_predictions)
            except Exception as e:
                self.print("Failed to load predictions into COCO format")
                self.print(f"First prediction: {self.eval_predictions[0]}")
                raise e
            
            # Create COCO evaluator
            coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Log metrics
            metrics = {
                'val/AP': coco_eval.stats[0],  # AP @ IoU=0.50:0.95
                'val/AP50': coco_eval.stats[1],  # AP @ IoU=0.50
                'val/AP75': coco_eval.stats[2],  # AP @ IoU=0.75
                'val/APm': coco_eval.stats[3],  # AP medium
                'val/APl': coco_eval.stats[4],  # AP large
            }
            
            for name, value in metrics.items():
                self.log(name, value, prog_bar=True)
                
        except Exception as e:
            self.print(f"Error in validation epoch end: {str(e)}")
            raise e
            
        finally:
            # Always reset predictions
            self.eval_predictions = []
    
    def configure_optimizers(self):
        """Configure optimizers"""
        # Optimize both adapter and ViTPose model parameters
        params = list(self.model.vit_pose.adapter.parameters()) + \
                list(self.model.vit_pose.vit_pose.parameters())
                
        optimizer = AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }