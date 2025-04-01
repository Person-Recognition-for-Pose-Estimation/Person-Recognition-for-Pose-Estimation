"""
Lightning module for pose estimation using ViTPose.
"""
import pytorch_lightning as pl
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.optim import AdamW  # type: ignore
from typing import Dict, Optional, List, Tuple
import torchmetrics
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

from transformers.models.vitpose.modeling_vitpose import VitPoseEstimatorOutput


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


class PoseEstimationModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        heatmap_size: tuple = (64, 48),  # ViTPose output size (H=64, W=48)
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
        
    def _get_keypoints_from_heatmaps(self, heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract keypoint coordinates from heatmaps using UDP.
        
        Args:
            heatmaps: [B, K, H, W] Predicted heatmaps
            
        Returns:
            coords: [B, K, 2] Keypoint coordinates in [x, y] format
            scores: [B, K] Confidence scores
        """
        B, K, H, W = heatmaps.shape
        
        # Apply softmax to convert heatmap to probability distribution
        heatmaps_softmax = heatmaps.reshape(B, K, -1)
        heatmaps_softmax = torch.nn.functional.softmax(heatmaps_softmax, dim=2)
        heatmaps = heatmaps_softmax.reshape(B, K, H, W)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=heatmaps.dtype, device=heatmaps.device),
            torch.arange(W, dtype=heatmaps.dtype, device=heatmaps.device),
            indexing='ij'
        )
        
        # Compute expected coordinates (soft-argmax)
        x_coords = []
        y_coords = []
        scores = []
        
        for b in range(B):
            for k in range(K):
                heatmap = heatmaps[b, k]  # [H, W]
                
                # Get score as max probability
                score = heatmap.max()
                scores.append(score)
                
                # Calculate expected x coordinate
                x_exp = (heatmap * x_grid).sum() / (heatmap.sum() + 1e-6)
                # Calculate expected y coordinate
                y_exp = (heatmap * y_grid).sum() / (heatmap.sum() + 1e-6)
                
                # Apply UDP: unbiased decoding considering discretization error
                x_exp = x_exp + 0.5
                y_exp = y_exp + 0.5
                
                x_coords.append(x_exp)
                y_coords.append(y_exp)
        
        # Stack and reshape results
        x_coords = torch.stack(x_coords).reshape(B, K)
        y_coords = torch.stack(y_coords).reshape(B, K)
        scores = torch.stack(scores).reshape(B, K)
        
        # Normalize coordinates to [0,1] range
        x_coords = x_coords / W
        y_coords = y_coords / H
        
        # Stack coordinates
        coords = torch.stack([x_coords, y_coords], dim=-1)  # [B, K, 2]
        
        return coords, scores
    
    def _generate_target_heatmap(self, keypoints, visibility, size):
        """Generate target heatmap from keypoint coordinates using UDP.
        
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
        
        # Convert normalized coordinates to pixel coordinates with UDP offset
        keypoints = keypoints.clone()
        keypoints[..., 0] = keypoints[..., 0] * W - 0.5
        keypoints[..., 1] = keypoints[..., 1] * H - 0.5
        
        # Create coordinate grid
        x = torch.arange(W, device=device)
        y = torch.arange(H, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx = xx.float()
        yy = yy.float()
        
        # Adaptive sigma based on keypoint scale
        # This helps handle different scales of people in the image
        base_sigma = self.sigma
        
        # Generate heatmaps for each batch and instance
        for b in range(B):
            for n in range(N):
                # Calculate person scale from bounding box or keypoint spread
                kpts_visible = visibility[b, n] > 0
                if kpts_visible.any():
                    visible_kpts = keypoints[b, n][kpts_visible]
                    xmax, ymax = visible_kpts.max(dim=0)[0]
                    xmin, ymin = visible_kpts.min(dim=0)[0]
                    person_scale = ((xmax - xmin) * (ymax - ymin)).sqrt()
                    # Adjust sigma based on person scale
                    adaptive_sigma = base_sigma * max(1.0, person_scale / 200.0)
                else:
                    adaptive_sigma = base_sigma
                
                for k in range(K):
                    if visibility[b, n, k] > 0:  # Keypoint is labeled
                        mu_x, mu_y = keypoints[b, n, k]
                        
                        # Compute Gaussian with UDP
                        gaussian = torch.exp(
                            -((xx - mu_x)**2 + (yy - mu_y)**2) / (2 * adaptive_sigma**2)
                        )
                        
                        # Add to heatmap (take maximum in case of multiple instances)
                        heatmaps[b, k] = torch.maximum(heatmaps[b, k], gaussian)
                        
                        # Update weight (1 for visible, 0.5 for occluded)
                        # Also consider the keypoint type importance
                        base_weight = 1.0 if visibility[b, n, k] == 2 else 0.5
                        weights[b, k] = max(weights[b, k], base_weight)
        
        # Apply additional UDP processing
        # 1. Normalize heatmaps
        heatmaps = heatmaps / (heatmaps.sum(dim=(-2, -1), keepdim=True) + 1e-6)
        
        # 2. Apply threshold to reduce noise
        heatmaps = torch.where(heatmaps > 0.005, heatmaps, torch.zeros_like(heatmaps))
        
        return heatmaps, weights
    
    def training_step(self, batch, batch_idx):
        """Single training step with UDP and adaptive weighting"""
        # Get inputs and targets
        images = batch['images']
        keypoints = batch['keypoints']  # [B, N, K, 3] - N instances, K keypoints, 3 = (x, y, visibility)
        masks = batch['masks']  # [B, N] boolean mask for valid instances
        
        # Split keypoints into coordinates and visibility
        coords = keypoints[..., :2]  # [B, N, K, 2]
        visibility = keypoints[..., 2]  # [B, N, K]
        
        # Generate target heatmaps and weights with UDP
        target_heatmaps, target_weights = self._generate_target_heatmap(
            coords, visibility, self.heatmap_size
        )
        
        # Forward pass through backbone and pose branch
        # pred_heatmaps = self.model(images)

        vit_pose_output: VitPoseEstimatorOutput = self.model(images)
        pred_heatmaps = vit_pose_output.heatmaps
        
        # Apply online hard example mining (OHEM)
        B, K, H, W = pred_heatmaps.shape
        pred_reshaped = pred_heatmaps.reshape(B, K, -1)
        target_reshaped = target_heatmaps.reshape(B, K, -1)
        
        # Calculate per-pixel losses
        pixel_losses = torch.nn.functional.mse_loss(
            pred_reshaped, target_reshaped, reduction='none'
        ).mean(dim=2)  # [B, K]
        
        # Sort losses and keep top 60% hard examples
        num_keep = int(0.6 * K)
        _, hard_indices = torch.topk(pixel_losses, k=num_keep, dim=1)
        
        # Update target weights for hard examples
        hard_weights = torch.zeros_like(target_weights)
        for b in range(B):
            hard_weights[b, hard_indices[b]] = target_weights[b, hard_indices[b]] * 2.0
        
        # Compute final loss with hard example mining
        loss = self.criterion(pred_heatmaps, target_heatmaps, hard_weights)
        
        # Add regularization for heatmap smoothness
        smoothness_loss = torch.nn.functional.smooth_l1_loss(
            pred_heatmaps[:, :, 1:, :], pred_heatmaps[:, :, :-1, :], reduction='mean'
        ) + torch.nn.functional.smooth_l1_loss(
            pred_heatmaps[:, :, :, 1:], pred_heatmaps[:, :, :, :-1], reduction='mean'
        )
        
        total_loss = loss + 0.1 * smoothness_loss
        
        # Log detailed metrics
        self.log('train/heatmap_loss', loss, prog_bar=True)
        self.log('train/smoothness_loss', smoothness_loss, prog_bar=False)
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        # Log learning rate
        if self.trainer is not None:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', current_lr, prog_bar=False)
        
        return total_loss
    
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

    def _flip_heatmaps(self, heatmaps: torch.Tensor, flip_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """Flip heatmaps horizontally and swap paired keypoints."""
        flipped_heatmaps = torch.flip(heatmaps, dims=[-1])  # Flip horizontally
        
        for left, right in flip_pairs:
            flipped_heatmaps[:, [left, right]] = flipped_heatmaps[:, [right, left]]
            
        return flipped_heatmaps
    
    def validation_step(self, batch, batch_idx):
        """Single validation step with flip test and OKS-based evaluation.
        
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
        keypoints = batch['keypoints']  # [B, N, K, 3]
        masks = batch['masks']  # [B, N]
        image_ids = batch['image_ids']
        boxes = batch['boxes']  # [B, N, 4]
        
        try:
            # Forward pass with flip test
            with torch.no_grad():
                # Original forward pass
                # pred_heatmaps = self.model(images)
                vit_pose_output: VitPoseEstimatorOutput = self.model(images)
                pred_heatmaps = vit_pose_output.heatmaps
                
                # Flipped forward pass
                flipped_images = torch.flip(images, dims=[-1])
                flipped_vit_pose_output: VitPoseEstimatorOutput = self.model(flipped_images)
                flipped_heatmaps = flipped_vit_pose_output.heatmaps

                # print("flipped_heatmaps", flipped_heatmaps)
                
                # Process flipped heatmaps
                flip_pairs = [  # COCO keypoint flip pairs
                    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                    (11, 12), (13, 14), (15, 16)
                ]
                flipped_heatmaps = self._flip_heatmaps(flipped_heatmaps, flip_pairs)
                
                # Average predictions
                pred_heatmaps = (pred_heatmaps + flipped_heatmaps) * 0.5
            
            # Get predicted keypoint coordinates and scores
            pred_coords, pred_scores = self._get_keypoints_from_heatmaps(pred_heatmaps)
            
            # Generate target heatmaps for loss computation
            target_heatmaps, target_weights = self._generate_target_heatmap(
                keypoints[..., :2], keypoints[..., 2], self.heatmap_size
            )
            
            # Debug prints
            print(f"\nPred heatmaps shape: {pred_heatmaps.shape}")
            print(f"Target heatmaps shape: {target_heatmaps.shape}")
            print(f"Target weights shape: {target_weights.shape}")
            print(f"Self.heatmap_size: {self.heatmap_size}")
            
            # Compute loss
            loss = self.criterion(pred_heatmaps, target_heatmaps, target_weights)
            self.log('val_loss', loss, prog_bar=True)
            
            # Store predictions with OKS-based filtering
            sigmas = torch.tensor([  # COCO keypoint sigmas for OKS computation
                .026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062,
                .107, .107, .087, .087, .089, .089
            ], device=pred_coords.device)
            
            batch_size = len(image_ids)
            for b in range(batch_size):
                valid_instances = masks[b]
                num_instances = valid_instances.sum().item()
                
                for n in range(num_instances):
                    # Get instance predictions
                    kpts = pred_coords[b, n]  # [K, 2]
                    scores = pred_scores[b, n]  # [K]
                    box = boxes[b, n]  # [4]
                    
                    # Compute OKS-based confidence
                    box_area = box[2] * box[3]  # width * height
                    d = torch.sqrt(box_area)  # scale
                    # Debug prints
                    print(f"\nKeypoints shape: {kpts.shape}")
                    print(f"Scores shape: {scores.shape}")
                    print(f"Box shape: {box.shape}")
                    print(f"Sigmas shape: {sigmas.shape}")
                    
                    # kpts is [K, 2], sum over coordinate dimension to get squared distance for each keypoint
                    squared_distances = kpts.square().sum(dim=-1)  # [K]
                    print(f"Squared distances shape: {squared_distances.shape}")
                    
                    # Scale factor
                    d = torch.sqrt(box_area)  # scale
                    print(f"Scale d: {d}")
                    
                    # Compute OKS scores
                    oks_scores = torch.exp(-squared_distances / (2 * d * d * sigmas * sigmas))
                    print(f"OKS scores shape: {oks_scores.shape}")
                    instance_score = (oks_scores * scores).mean()
                    
                    # Only keep predictions with good OKS scores
                    if instance_score > self.keypoint_thresh:
                        # Convert to COCO format
                        keypoints_coco = []
                        for kpt, score in zip(kpts, scores):
                            x, y = kpt.cpu().numpy()
                            v = 2 if score > self.keypoint_thresh else 1
                            keypoints_coco.extend([x, y, v])
                        
                        pred = {
                            'image_id': image_ids[b].item(),
                            'category_id': 1,  # person
                            'keypoints': keypoints_coco,
                            'score': instance_score.item(),
                            'bbox': box.cpu().numpy().tolist()
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
                'val_pck': coco_eval.stats[0],  # Use AP as PCK for checkpoint monitoring
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