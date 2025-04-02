"""
Enhanced pose estimation Lightning module using GluonCV components.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
import numpy as np

from gluoncv.loss import L1Loss, HuberLoss
from gluoncv.data.transforms.pose import (
    transform_preds,
    get_max_pred,
    get_final_preds
)
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

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
    """MSE loss for heatmaps with optional OHKM."""
    def __init__(self, use_target_weight=True, use_ohkm=True, topk=8):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.use_ohkm = use_ohkm
        self.topk = topk

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        
        loss = self.criterion(heatmaps_pred, heatmaps_gt).mean(dim=2)
        
        if self.use_target_weight:
            loss = loss * target_weight
            
        if self.use_ohkm:
            # Online Hard Keypoint Mining
            topk_val, topk_idx = torch.topk(loss, k=self.topk, dim=1)
            loss = torch.gather(loss, 1, topk_idx)
            
        return loss.mean()

class PoseEstimationModule(pl.LightningModule):
    """PyTorch Lightning module for pose estimation using GluonCV components."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        use_ohkm: bool = True,
        topk: int = 8,
        num_joints: int = 17,
        flip_test: bool = True,
        post_process: bool = True,
    ):
        """
        Initialize pose estimation module.
        
        Args:
            model: Base model (backbone + adapter)
            learning_rate: Initial learning rate
            use_ohkm: Whether to use Online Hard Keypoint Mining
            topk: Number of top keypoints to consider in OHKM
            num_joints: Number of keypoints
            flip_test: Whether to use flip testing
            post_process: Whether to apply post-processing
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Model
        self.model = model
        self.model.set_task('pose_estimation')
        
        # Loss function
        self.criterion = JointsMSELoss(
            use_target_weight=True,
            use_ohkm=use_ohkm,
            topk=topk
        )
        
        # Metrics
        self.train_acc = AverageMeter()
        self.val_acc = AverageMeter()
        self.eval_results = []
        
        # Testing parameters
        self.flip_test = flip_test
        self.post_process = post_process
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
        
    def _get_accuracy(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor,
        hm_type: str = 'gaussian',
        thr: float = 0.05
    ) -> Tuple[float, torch.Tensor]:
        """Calculate accuracy."""
        num_joints = output.shape[1]
        
        # Get predictions
        preds, maxvals = get_max_pred(output.detach().cpu().numpy())
        
        # Get targets
        target = target.detach().cpu().numpy()
        target_weight = target_weight.detach().cpu().numpy()
        
        norm = np.ones((preds.shape[0], 2)) * preds.shape[1]
        
        # Calculate normalized distance
        dists = np.zeros((preds.shape[0], preds.shape[1]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target_weight[n, c] > 0:
                    dists[n, c] = np.linalg.norm(preds[n, c] - target[n, c]) / norm[n, 0]
                    
        acc = np.zeros(num_joints)
        avg_acc = 0
        cnt = 0
        
        for i in range(num_joints):
            joint_valid = target_weight[:, i] > 0
            if joint_valid.sum() > 0:
                joint_acc = (dists[joint_valid, i] < thr).astype(np.float32).mean()
                acc[i] = joint_acc
                cnt += 1
                avg_acc += joint_acc
                
        if cnt > 0:
            avg_acc = avg_acc / cnt
            
        return avg_acc, torch.from_numpy(acc)
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Get inputs
        images = batch['image']
        target = batch['target']
        target_weight = batch['target_weight']
        
        # Forward pass
        output = self(images)
        
        # Calculate loss
        loss = self.criterion(output, target, target_weight)
        
        # Calculate accuracy
        acc, _ = self._get_accuracy(output, target, target_weight)
        self.train_acc.update(acc, images.size(0))
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc.avg, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Validation step."""
        # Get inputs
        images = batch['image']
        target = batch['target']
        target_weight = batch['target_weight']
        center = batch['center']
        scale = batch['scale']
        
        # Forward pass
        output = self(images)
        
        if self.flip_test:
            # Flip test
            images_flipped = torch.flip(images, [3])
            output_flipped = self(images_flipped)
            
            output_flipped = torch.flip(output_flipped, [3])
            
            # Average heatmaps
            output = (output + output_flipped) * 0.5
        
        # Calculate loss
        loss = self.criterion(output, target, target_weight)
        
        # Calculate accuracy
        acc, _ = self._get_accuracy(output, target, target_weight)
        self.val_acc.update(acc, images.size(0))
        
        # Get predictions for COCO evaluation
        if self.post_process:
            # Convert heatmap to coordinates
            preds, maxvals = get_final_preds(
                output.detach().cpu().numpy(),
                center.cpu().numpy(),
                scale.cpu().numpy()
            )
        else:
            preds, maxvals = get_max_pred(output.detach().cpu().numpy())
        
        # Store predictions
        for idx in range(len(preds)):
            self.eval_results.append({
                'keypoints': preds[idx],
                'scores': maxvals[idx],
                'image_id': batch['image_id'][idx].item(),
                'area': batch['area'][idx].item(),
                'bbox': batch['bbox'][idx].cpu().numpy()
            })
        
        # Log metrics
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc.avg, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'acc': acc}
        
    def on_validation_epoch_end(self) -> None:
        """Compute COCO metrics at the end of validation."""
        if not self.eval_results:
            return
            
        # Reset metrics
        self.val_acc.reset()
        
        # Prepare results for COCO evaluation
        coco = COCO(self.trainer.datamodule.val_dataset.ann_file)
        
        # Convert predictions to COCO format
        coco_pred_results = []
        for result in self.eval_results:
            keypoints = result['keypoints']
            scores = result['scores']
            
            # Format keypoints for COCO [x1,y1,s1,x2,y2,s2,...]
            keypoints_with_scores = np.zeros((17, 3))
            keypoints_with_scores[:, :2] = keypoints
            keypoints_with_scores[:, 2] = scores
            
            coco_pred_results.append({
                'image_id': result['image_id'],
                'category_id': 1,  # person
                'keypoints': keypoints_with_scores.reshape(-1).tolist(),
                'score': float(scores.mean()),
                'bbox': result['bbox'].tolist(),
                'area': result['area']
            })
        
        # Run COCO evaluation
        coco_dt = coco.loadRes(coco_pred_results)
        coco_eval = COCOeval(coco, coco_dt, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Log COCO metrics
        metrics = {
            'val/AP': coco_eval.stats[0],
            'val/AP50': coco_eval.stats[1],
            'val/AP75': coco_eval.stats[2],
            'val/APm': coco_eval.stats[3],
            'val/APl': coco_eval.stats[4]
        }
        
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
            
        # Clear evaluation results
        self.eval_results = []
        
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
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
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }