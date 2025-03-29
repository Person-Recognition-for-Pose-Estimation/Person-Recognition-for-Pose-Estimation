"""
Lightning module for YOLO face detection training using Ultralytics trainer.
"""
import pytorch_lightning as pl
import torch  # type: ignore
from torch.optim import Adam  # type: ignore
from ultralytics.utils.metrics import DetMetrics
from ..ultralytics_wrapper import UltralyticsTrainerWrapper
from ultralytics.cfg import get_cfg
from ..utils import compute_iou

class FaceDetectionModule(pl.LightningModule):
    def __init__(
        self,
        model,
        data_cfg: str,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        **ultralytics_args
    ):
        """
        Initialize YOLO Lightning Module with Ultralytics integration
        
        Args:
            model: Combined multi-task model
            data_cfg: Path to YOLO data config file
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            **ultralytics_args: Additional arguments for Ultralytics trainer
        """
        super().__init__()
        self.model = model
        self.model.set_task('face_detection')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.metrics = DetMetrics(save_dir=None, plot=False, names={0: 'face'})
        
        self.ultralytics_cfg = get_cfg()
        self.ultralytics_cfg.data = data_cfg
        self.ultralytics_cfg.lr0 = learning_rate
        self.ultralytics_cfg.weight_decay = weight_decay
        for k, v in ultralytics_args.items():
            setattr(self.ultralytics_cfg, k, v)
            
        self.trainer_wrapper = UltralyticsTrainerWrapper(self.model, task_name='face_detection')
        
    def setup(self, stage=None):
        """Setup trainer on fit start"""
        if stage == 'fit':
            self.trainer_wrapper.setup_trainer(self.ultralytics_cfg)
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Single training step using Ultralytics trainer"""
        return self.trainer_wrapper.train_step(batch)
    
    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        nms_preds, targets, loss_dict = self.trainer_wrapper.validation_step(batch)
        if nms_preds is None:
            return
            
        # Log validation loss components
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, prog_bar=True)
            
        # Process NMS predictions for metrics
        for i, pred in enumerate(nms_preds):  # Loop through batch
            if len(pred) == 0:  # Skip if no detections
                continue
                
            # Extract boxes, scores, and classes from NMS predictions
            boxes = pred[:, :4]  # [num_det, 4]
            scores = pred[:, 4]  # [num_det]
            pred_cls = pred[:, 5]  # [num_det]
            
            # Get target boxes and classes for this batch item
            batch_mask = targets['batch_idx'] == i
            target_boxes = targets['bboxes'][batch_mask]
            target_cls = targets['cls'][batch_mask]
            
            if len(target_boxes) == 0:  # Skip if no targets
                continue
                
            # Calculate IoU between predictions and targets
            iou = compute_iou(boxes, target_boxes)  # [num_det, num_targets]
            max_iou, _ = iou.max(dim=1)  # [num_det]
            
            # True positives are predictions with IoU > 0.5
            tp = (max_iou > 0.5).float()
            
            # Update metrics - one detection at a time to handle different numbers of detections
            for j in range(len(tp)):
                self.metrics.process(
                    tp[j:j+1],  # [1]
                    scores[j:j+1],  # [1]
                    pred_cls[j:j+1],  # [1]
                    target_cls  # [num_targets]
                )
    
    def on_validation_epoch_end(self):
        """Called at the end of validation"""
        results = self.metrics.results_dict
        maps = self.metrics.maps
        fitness = self.metrics.fitness
        
        self.log("val/fitness", fitness, prog_bar=True)
        
        for k, v in maps.items():
            self.log(f"val/mAP{k}", v, prog_bar=True)
        
        for k, v in results.items():
            self.log(f"val/{k}", v, prog_bar=True)
        
        self.metrics = DetMetrics(save_dir=None, plot=False, names={0: 'face'})
    
    def configure_optimizers(self):
        """Configure optimizers"""
        # Optimize both adapter and YOLO model parameters
        params = list(self.model.yolo_face.adapter.parameters()) + \
                list(self.model.yolo_face.yolo_model.parameters())
                
        optimizer = Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer