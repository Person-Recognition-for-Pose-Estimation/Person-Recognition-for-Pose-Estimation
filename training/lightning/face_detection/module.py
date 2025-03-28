"""
Lightning module for YOLO face detection training using Ultralytics trainer.
"""
import pytorch_lightning as pl
import torch  # type: ignore
from torch.optim import Adam  # type: ignore
from ultralytics.utils.metrics import DetMetrics
from ..ultralytics_wrapper import UltralyticsTrainerWrapper
from ultralytics.cfg import get_cfg
from ..utils import compute_iou as box_iou

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
        preds, targets, loss_dict = self.trainer_wrapper.validation_step(batch)
        if preds is None:
            return
            
        # Log validation loss components
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, prog_bar=True)
            
        # Process predictions for metrics
        # YOLO outputs a list of tensors, where each tensor contains detections
        if isinstance(preds, (list, tuple)) and len(preds) > 0:
            # Get the first output tensor which contains detections
            detections = preds[0]  # Shape: [batch_size, num_anchors, num_classes + 5]
            
            # Extract boxes and scores
            boxes = detections[..., :4]  # [batch_size, num_anchors, 4]
            scores = detections[..., 4]  # [batch_size, num_anchors]
            
            # For face detection, all predictions are class 0
            pred_cls = torch.zeros_like(scores)  # [batch_size, num_anchors]
            
            # Get target information
            target_boxes = targets['bboxes'].to(boxes.device)
            target_cls = targets['cls'].to(boxes.device)
            
            # Calculate IoU between predictions and targets
            boxes_flat = boxes.reshape(-1, 4)
            target_boxes_flat = target_boxes.reshape(-1, 4)
            iou = box_iou(boxes_flat, target_boxes_flat)
            max_iou, _ = iou.max(dim=1)
            
            # True positives are predictions with IoU > 0.5
            tp = (max_iou > 0.5).float().view_as(scores)
            
            # Update metrics
            self.metrics.process(tp, scores, pred_cls, target_cls)
    
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
                list(self.model.yolo_face.yolo.parameters())
                
        optimizer = Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer