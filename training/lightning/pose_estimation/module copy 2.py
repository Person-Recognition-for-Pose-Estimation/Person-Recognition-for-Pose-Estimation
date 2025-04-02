"""
Lightning module for pose estimation using ViTPose with COCO-standard evaluation.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore
import json
from pathlib import Path

from transformers.models.vitpose.modeling_vitpose import VitPoseEstimatorOutput

# Import COCO constants from datamodule
from .datamodule import COCO_KEYPOINTS, COCO_FLIP_PAIRS, COCO_SIGMAS

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class JointsMSELoss(nn.Module):
    """Enhanced MSE loss for heatmaps with COCO-standard weighting.
    
    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
        loss_weight (float): Weight of the loss. Default: 1.0.
        use_ohkm (bool): Use Online Hard Keypoint Mining
        topk (int): Number of hardest keypoints to keep when using OHKM
    """
    def __init__(self, use_target_weight=True, loss_weight=1.0, use_ohkm=True, topk=8):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.use_ohkm = use_ohkm
        self.topk = topk
        
        # Initialize COCO keypoint weights based on OKS sigmas
        self.keypoint_weights = torch.from_numpy(1 / (COCO_SIGMAS + 1e-8))
        self.keypoint_weights = self.keypoint_weights / self.keypoint_weights.mean()
    
    def forward(self, output, target, target_weight, areas=None):
        """Forward function with OHKM support.
        
        Args:
            output: Predicted heatmaps [B, K, H, W]
            target: Target heatmaps [B, K, H, W]
            target_weight: Target weights [B, K]
            areas: Instance areas for adaptive weighting [B, N]
        """
        batch_size = output.size(0)
        num_joints = output.size(1)
        
        # Move keypoint weights to correct device
        if self.keypoint_weights.device != output.device:
            self.keypoint_weights = self.keypoint_weights.to(output.device)
        
        # Compute per-keypoint loss
        output = output.reshape((batch_size, num_joints, -1))
        target = target.reshape((batch_size, num_joints, -1))
        
        # Apply target weights and keypoint weights
        if self.use_target_weight:
            # Combine target weight with keypoint weights
            combined_weight = target_weight * self.keypoint_weights.view(1, -1)
            combined_weight = combined_weight.unsqueeze(-1)
            
            loss = self.criterion(output, target)
            loss = loss.mean(dim=2)  # Mean over spatial dimensions
            loss = loss * combined_weight.squeeze(-1)
        else:
            loss = self.criterion(output, target).mean(dim=2)
        
        if self.use_ohkm:
            # Online Hard Keypoint Mining
            with torch.no_grad():
                # Get topk hardest keypoints
                topk_values, topk_indices = torch.topk(loss, k=self.topk, dim=1)
                
                # Create mask for selected keypoints
                mask = torch.zeros_like(loss)
                for i in range(batch_size):
                    mask[i, topk_indices[i]] = 1
                
                loss = loss * mask
        
        # Compute final loss
        if self.use_ohkm:
            loss = loss.sum() / (batch_size * self.topk)
        else:
            loss = loss.mean()
        
        return loss * self.loss_weight


class OKSLoss(nn.Module):
    """Object Keypoint Similarity based loss."""
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.sigmas = torch.from_numpy(COCO_SIGMAS).float()
    
    def forward(self, pred_coords, target_coords, target_vis, areas):
        """
        Args:
            pred_coords: [B, K, 2] Predicted coordinates (normalized)
            target_coords: [B, K, 2] Target coordinates (normalized)
            target_vis: [B, K] Visibility flags
            areas: [B] Object areas
        """
        if self.sigmas.device != pred_coords.device:
            self.sigmas = self.sigmas.to(pred_coords.device)
        
        # Compute squared distances
        dx = pred_coords[..., 0] - target_coords[..., 0]
        dy = pred_coords[..., 1] - target_coords[..., 1]
        squared_dist = dx.pow(2) + dy.pow(2)
        
        # Compute OKS
        areas = areas.view(-1, 1)  # [B, 1]
        squared_sigma = 2 * self.sigmas.pow(2).view(1, -1)  # [1, K]
        oks = torch.exp(-squared_dist / (2 * areas * squared_sigma + 1e-8))
        
        # Only consider visible keypoints
        oks = oks * target_vis
        
        # Compute loss as negative log OKS
        loss = -torch.log(oks.clamp(min=1e-8))
        
        # Average over visible keypoints
        num_visible = target_vis.sum(dim=1, keepdim=True).clamp(min=1)
        loss = (loss * target_vis).sum(dim=1) / num_visible.squeeze(1)
        
        return loss.mean() * self.loss_weight


class PoseEstimationModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        heatmap_size: tuple = (64, 48),  # ViTPose output size (H=64, W=48)
        sigma: float = 2.0,  # Gaussian sigma for heatmap generation
        keypoint_thresh: float = 0.3,  # Threshold for keypoint visibility
        use_ohkm: bool = True,  # Use Online Hard Keypoint Mining
        ohkm_topk: int = 8,  # Number of hardest keypoints to keep
        use_oks_loss: bool = True,  # Use OKS-based loss
    ):
        """
        Initialize ViTPose Lightning Module with COCO-standard evaluation
        
        Args:
            model: Combined multi-task model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            heatmap_size: Size of output heatmaps (H, W)
            sigma: Gaussian sigma for heatmap generation
            keypoint_thresh: Confidence threshold for keypoint visibility
            use_ohkm: Whether to use Online Hard Keypoint Mining
            ohkm_topk: Number of hardest keypoints to keep when using OHKM
            use_oks_loss: Whether to use OKS-based loss
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
        
        # Loss functions
        self.heatmap_loss = JointsMSELoss(
            use_target_weight=True,
            loss_weight=1.0,
            use_ohkm=use_ohkm,
            topk=ohkm_topk
        )
        
        if use_oks_loss:
            self.oks_loss = OKSLoss(loss_weight=0.1)
        else:
            self.oks_loss = None
            
        # Metrics
        self.train_acc = AverageMeter()
        self.val_acc = AverageMeter()
        self.eval_predictions = []  # Store predictions during validation
        
        # Cache for evaluation
        self._eval_cache = {
            'coco_gt': None,
            'image_ids': set()
        }
        
    def setup(self, stage: Optional[str] = None):
        """Validate the datamodule has required attributes and load COCO GT."""
        if stage == 'fit' or stage == 'validate':
            if not hasattr(self.trainer.datamodule, 'val_annotations_path'):
                raise AttributeError(
                    "DataModule must provide 'val_annotations_path' for COCO evaluation. "
                    "This should point to the COCO keypoint annotations JSON file."
                )
            
            # Load COCO GT annotations if not already loaded
            if self._eval_cache['coco_gt'] is None:
                self._eval_cache['coco_gt'] = COCO(
                    self.trainer.datamodule.val_annotations_path
                )
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
        
    def _get_keypoints_from_heatmaps(
        self, 
        heatmaps: torch.Tensor,
        boxes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract keypoint coordinates from heatmaps using UDP.
        
        Args:
            heatmaps: [B, K, H, W] Predicted heatmaps
            boxes: [B, 4] Optional bounding boxes for scale-aware decoding
            
        Returns:
            coords: [B, K, 2] Keypoint coordinates in [x, y] format
            scores: [B, K] Confidence scores
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Reshape heatmaps for efficient computation
        heatmaps_flat = heatmaps.reshape(B, K, -1)
        
        # Apply softmax to get probability distributions
        heatmaps_prob = F.softmax(heatmaps_flat, dim=2).reshape(B, K, H, W)
        
        # Compute expected coordinates using vectorized operations
        x_exp = (heatmaps_prob * x_grid.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))
        y_exp = (heatmaps_prob * y_grid.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))
        
        # Get confidence scores as max probabilities
        scores = heatmaps_prob.reshape(B, K, -1).max(dim=2)[0]
        
        # Apply UDP offset
        x_exp = x_exp + 0.5
        y_exp = y_exp + 0.5
        
        # Stack coordinates
        coords = torch.stack([x_exp, y_exp], dim=-1)  # [B, K, 2]
        
        # Normalize coordinates to [0,1] range
        coords[..., 0] = coords[..., 0] / W
        coords[..., 1] = coords[..., 1] / H
        
        # Apply scale-aware refinement if boxes are provided
        if boxes is not None:
            # Compute scale from box area
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            scale = torch.sqrt(box_areas).view(-1, 1, 1)  # [B, 1, 1]
            
            # Adjust confidence based on scale
            scale_weight = torch.clamp(scale / 96.0, min=0.5, max=2.0)  # normalize to nominal size
            scores = scores * scale_weight.squeeze(-1)
        
        return coords, scores
        
    def _generate_target_heatmap(
        self,
        keypoints: torch.Tensor,
        visibility: torch.Tensor,
        areas: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate target heatmaps with adaptive sigma based on instance scale.
        
        Args:
            keypoints: [B, N, K, 2] Normalized keypoint coordinates
            visibility: [B, N, K] Keypoint visibility flags
            areas: [B, N] Optional instance areas for adaptive sigma
            
        Returns:
            heatmaps: [B, K, H, W] Target heatmaps
            weights: [B, K] Target weights
        """
        B, N, K, _ = keypoints.shape
        H, W = self.heatmap_size
        device = keypoints.device
        
        # Initialize outputs
        heatmaps = torch.zeros((B, K, H, W), device=device)
        weights = torch.zeros((B, K), device=device)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Convert normalized coordinates to pixel coordinates
        keypoints = keypoints.clone()
        keypoints[..., 0] = keypoints[..., 0] * W - 0.5
        keypoints[..., 1] = keypoints[..., 1] * H - 0.5
        
        # Compute adaptive sigma if areas are provided
        if areas is not None:
            # Scale sigma based on sqrt of area, normalized to nominal size
            base_sigma = self.sigma
            scale = torch.sqrt(areas)  # [B, N]
            adaptive_sigma = base_sigma * torch.clamp(scale / 96.0, min=0.5, max=2.0)
            adaptive_sigma = adaptive_sigma.unsqueeze(-1)  # [B, N, 1]
        else:
            adaptive_sigma = torch.full((B, N, 1), self.sigma, device=device)
        
        # Generate heatmaps efficiently using vectorized operations
        for b in range(B):
            for n in range(N):
                valid_mask = visibility[b, n] > 0
                if not valid_mask.any():
                    continue
                
                sigma = adaptive_sigma[b, n]
                
                # Compute Gaussian for all keypoints at once
                mu_x = keypoints[b, n, :, 0]  # [K]
                mu_y = keypoints[b, n, :, 1]  # [K]
                
                dx = x_grid.unsqueeze(0) - mu_x.unsqueeze(1).unsqueeze(2)  # [K, H, W]
                dy = y_grid.unsqueeze(0) - mu_y.unsqueeze(1).unsqueeze(2)  # [K, H, W]
                
                gaussian = torch.exp(
                    -(dx.pow(2) + dy.pow(2)) / (2 * sigma.pow(2))
                ) * valid_mask.unsqueeze(1).unsqueeze(2)
                
                # Update heatmaps (take maximum in case of multiple instances)
                heatmaps[b] = torch.maximum(heatmaps[b], gaussian)
                
                # Update weights based on visibility
                weights[b] = torch.maximum(
                    weights[b],
                    torch.where(visibility[b, n] == 2, torch.ones_like(weights[b]), 0.5 * torch.ones_like(weights[b]))
                )
        
        # Normalize heatmaps
        heatmaps = heatmaps / (heatmaps.sum(dim=(2, 3), keepdim=True) + 1e-8)
        
        # Apply threshold to reduce noise
        heatmaps = torch.where(heatmaps > 0.005, heatmaps, torch.zeros_like(heatmaps))
        
        return heatmaps, weights
    
    def training_step(self, batch, batch_idx):
        """Training step with combined heatmap and OKS losses."""
        # Get inputs
        images = batch['images']
        keypoints = batch['keypoints']  # [B, N, K, 3]
        boxes = batch['boxes']  # [B, N, 4]
        areas = batch['areas']  # [B, N]
        masks = batch['masks']  # [B, N]
        
        # Split keypoints into coordinates and visibility
        coords = keypoints[..., :2]  # [B, N, K, 2]
        visibility = keypoints[..., 2]  # [B, N, K]
        
        # Generate target heatmaps with adaptive sigma
        target_heatmaps, target_weights = self._generate_target_heatmap(
            coords, visibility, areas
        )
        
        # Forward pass
        vit_pose_output: VitPoseEstimatorOutput = self.model(images)
        pred_heatmaps = vit_pose_output.heatmaps
        
        # Compute heatmap loss
        heatmap_loss = self.heatmap_loss(
            pred_heatmaps,
            target_heatmaps,
            target_weights,
            areas=areas.mean(dim=1) if areas is not None else None
        )
        
        # Get predicted coordinates
        pred_coords, pred_scores = self._get_keypoints_from_heatmaps(
            pred_heatmaps,
            boxes=boxes[:, 0] if boxes is not None else None  # Use first instance's box
        )
        
        # Compute OKS loss if enabled
        total_loss = heatmap_loss
        if self.oks_loss is not None:
            # Use first instance's keypoints and area
            oks_loss = self.oks_loss(
                pred_coords,
                coords[:, 0],  # [B, K, 2]
                visibility[:, 0],  # [B, K]
                areas[:, 0]  # [B]
            )
            total_loss = total_loss + oks_loss
            self.log('train/oks_loss', oks_loss, prog_bar=False)
        
        # Compute PCK (percentage of correct keypoints)
        with torch.no_grad():
            # Consider a keypoint correct if within 0.2 * sqrt(area) pixels
            threshold = 0.2 * torch.sqrt(areas[:, 0:1])  # [B, 1]
            
            # Compute distances for visible keypoints
            vis_mask = visibility[:, 0] > 0  # [B, K]
            dists = torch.norm(pred_coords - coords[:, 0], dim=-1)  # [B, K]
            correct = (dists < threshold.unsqueeze(1)) & vis_mask
            
            pck = correct.float().mean()
            self.train_acc.update(pck.item())
        
        # Log metrics
        self.log('train/heatmap_loss', heatmap_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/pck', self.train_acc.avg, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with COCO evaluation metrics."""
        # Get inputs
        images = batch['images']
        keypoints = batch['keypoints']  # [B, N, K, 3]
        boxes = batch['boxes']  # [B, N, 4]
        areas = batch['areas']  # [B, N]
        masks = batch['masks']  # [B, N]
        is_crowd = batch['is_crowd']  # [B, N]
        image_ids = batch['image_ids']
        
        # Split keypoints into coordinates and visibility
        coords = keypoints[..., :2]  # [B, N, K, 2]
        visibility = keypoints[..., 2]  # [B, N, K]
        
        try:
            # Forward pass with flip test
            with torch.no_grad():
                # Original forward pass
                vit_pose_output: VitPoseEstimatorOutput = self.model(images)
                pred_heatmaps = vit_pose_output.heatmaps
                
                # Flipped forward pass
                flipped_images = torch.flip(images, dims=[-1])
                flipped_output: VitPoseEstimatorOutput = self.model(flipped_images)
                flipped_heatmaps = flipped_output.heatmaps
                
                # Process flipped heatmaps
                flipped_heatmaps = torch.flip(flipped_heatmaps, dims=[-1])
                for pair in COCO_FLIP_PAIRS:
                    flipped_heatmaps[:, pair] = flipped_heatmaps[:, pair].flip(0)
                
                # Average predictions
                pred_heatmaps = (pred_heatmaps + flipped_heatmaps) * 0.5
            
            # Generate target heatmaps and compute loss
            target_heatmaps, target_weights = self._generate_target_heatmap(
                coords, visibility, areas
            )
            
            val_loss = self.heatmap_loss(
                pred_heatmaps,
                target_heatmaps,
                target_weights,
                areas=areas.mean(dim=1) if areas is not None else None
            )
            
            # Get predicted coordinates
            pred_coords, pred_scores = self._get_keypoints_from_heatmaps(
                pred_heatmaps,
                boxes=boxes[:, 0] if boxes is not None else None
            )
            
            # Store predictions for COCO evaluation
            batch_size = len(image_ids)
            for b in range(batch_size):
                img_id = image_ids[b]
                
                # Skip if we've already processed this image
                if img_id in self._eval_cache['image_ids']:
                    continue
                    
                self._eval_cache['image_ids'].add(img_id)
                
                # Process each instance
                for n in range(masks[b].sum()):
                    if is_crowd[b, n]:
                        continue
                        
                    # Get instance predictions
                    kpts = pred_coords[b]  # [K, 2]
                    scores = pred_scores[b]  # [K]
                    box = boxes[b, n]  # [4]
                    area = areas[b, n].item()
                    
                    # Convert to COCO format [x, y, v] * K
                    keypoints_coco = []
                    
                    # Move tensors to CPU first
                    kpts_cpu = kpts.cpu()
                    scores_cpu = scores.cpu()
                    box_cpu = box.cpu()
                    
                    # Get box dimensions
                    box_width = box_cpu[2] - box_cpu[0]
                    box_height = box_cpu[3] - box_cpu[1]
                    box_x = box_cpu[0]
                    box_y = box_cpu[1]
                    
                    # Convert normalized coordinates to absolute coordinates
                    for kpt, score in zip(kpts_cpu, scores_cpu):
                        # Scale normalized coordinates by box dimensions
                        x = float(kpt[0].item() * box_width + box_x)
                        y = float(kpt[1].item() * box_height + box_y)
                        v = 2 if score.item() > self.keypoint_thresh else 1
                        keypoints_coco.extend([x, y, int(v)])
                    
                    # Compute instance score as mean of keypoint scores
                    instance_score = float(scores_cpu.mean().item())
                    
                    # Create COCO prediction
                    pred = {
                        'image_id': int(img_id),
                        'category_id': 1,  # person
                        'keypoints': keypoints_coco,
                        'score': instance_score,
                        'bbox': [float(x) for x in box_cpu.tolist()],
                        'area': float(area)
                    }
                    
                    self.eval_predictions.append(pred)
            
            # Log validation metrics
            self.log('val_loss', val_loss, prog_bar=True)
            
            return val_loss
            
        except Exception as e:
            self.print(f"Error in validation step: {str(e)}")
            raise e
    
    def on_validation_epoch_start(self):
        """Initialize validation state."""
        self.eval_predictions = []
        self.val_acc.reset()
        self._eval_cache['image_ids'] = set()
    
    def on_validation_epoch_end(self):
        """Compute COCO metrics at the end of validation."""
        try:
            # Skip if no predictions
            if not self.eval_predictions:
                self.print("No predictions to evaluate")
                return
            
            # Save predictions for debugging if needed
            pred_file = Path(self.trainer.default_root_dir) / f"predictions_epoch{self.current_epoch}.json"
            with open(pred_file, 'w') as f:
                json.dump(self.eval_predictions, f)
            
            # Create COCO prediction object
            try:
                coco_dt = self._eval_cache['coco_gt'].loadRes(self.eval_predictions)
            except Exception as e:
                self.print("Failed to load predictions into COCO format")
                self.print(f"First prediction: {self.eval_predictions[0]}")
                raise e
            
            # Create COCO evaluator
            coco_eval = COCOeval(
                self._eval_cache['coco_gt'],
                coco_dt,
                'keypoints'
            )
            
            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Log detailed metrics
            metrics = {
                'val/AP': coco_eval.stats[0],      # AP @ IoU=.50:.05:.95
                'val/AP50': coco_eval.stats[1],    # AP @ IoU=.50
                'val/AP75': coco_eval.stats[2],    # AP @ IoU=.75
                'val/APm': coco_eval.stats[3],     # AP for medium objects
                'val/APl': coco_eval.stats[4],     # AP for large objects
                'val/AR': coco_eval.stats[5],      # AR @ IoU=.50:.05:.95
                'val/AR50': coco_eval.stats[6],    # AR @ IoU=.50
                'val/AR75': coco_eval.stats[7],    # AR @ IoU=.75
                'val/ARm': coco_eval.stats[8],     # AR for medium objects
                'val/ARl': coco_eval.stats[9],     # AR for large objects
                'val_pck': self.val_acc.avg        # PCK for checkpoint monitoring
            }
            
            # Log each metric
            for name, value in metrics.items():
                self.log(name, value, prog_bar=True, sync_dist=True)
            
            # Print detailed evaluation
            self.print("\nDetailed Evaluation Results:")
            self.print(f"Average Precision (AP) @ IoU=.50:.05:.95: {metrics['val/AP']:.3f}")
            self.print(f"Average Precision (AP) @ IoU=.50: {metrics['val/AP50']:.3f}")
            self.print(f"Average Precision (AP) @ IoU=.75: {metrics['val/AP75']:.3f}")
            self.print(f"Average Precision (AP) for medium objects: {metrics['val/APm']:.3f}")
            self.print(f"Average Precision (AP) for large objects: {metrics['val/APl']:.3f}")
            self.print(f"Average Recall (AR) @ IoU=.50:.05:.95: {metrics['val/AR']:.3f}")
            self.print(f"PCK @ 0.2: {metrics['val_pck']:.3f}")
            
        except Exception as e:
            self.print(f"Error in validation epoch end: {str(e)}")
            raise e
            
        finally:
            # Reset validation state
            self.eval_predictions = []
            self._eval_cache['image_ids'] = set()
    
    def configure_optimizers(self):
        """Configure optimizers with cosine annealing and warmup."""
        # Collect all trainable parameters
        params = []
        
        # Adapter parameters (higher learning rate)
        adapter_params = list(self.model.vit_pose.adapter.parameters())
        if adapter_params:
            params.append({
                'params': adapter_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay
            })
        
        # ViTPose parameters (lower learning rate)
        vitpose_params = list(self.model.vit_pose.vit_pose.parameters())
        if vitpose_params:
            params.append({
                'params': vitpose_params,
                'lr': self.learning_rate * 0.1,  # Lower LR for pretrained model
                'weight_decay': self.weight_decay
            })
        
        # Create optimizer
        optimizer = AdamW(params)
        
        # Create scheduler with warmup
        num_epochs = self.trainer.max_epochs
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = min(1000, total_steps // 5)  # Warmup for 20% of training or 1000 steps
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[p['lr'] for p in params],
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                cycle_momentum=False,
                div_factor=25.0,  # LR starts at max_lr/25
                final_div_factor=1e4,  # Final LR is max_lr/10000
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]