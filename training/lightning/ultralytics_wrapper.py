"""
Wrapper for Ultralytics trainer to work with our multi-task model.
"""
import torch  # type: ignore
from ultralytics.utils import LOGGER
from .custom_trainer import CustomUltralyticsTrainer

class UltralyticsTrainerWrapper:
    def __init__(self, combined_model, task_name):
        """
        Initialize wrapper for Ultralytics trainer
        
        Args:
            combined_model: Our multi-task model with backbone and all branches
            task_name: Name of the task ('face_detection' or 'person_detection')
        """
        self.model = combined_model
        self.task_name = task_name
        self.model.set_task(task_name)
        
        # Get the appropriate YOLO branch based on task
        self.yolo_branch = (self.model.yolo_face if task_name == 'face_detection' 
                           else self.model.yolo_person)
        
    def setup_trainer(self, cfg, _callbacks=None):
        """
        Setup Ultralytics trainer with our combined model
        
        Args:
            cfg: Training configuration
            _callbacks: Optional callbacks
        """
        self.trainer = CustomUltralyticsTrainer(
            combined_model=self.model,
            cfg=cfg,
            _callbacks=_callbacks
        )
        return self.trainer
        
    def train(self, cfg):
        """
        Train using Ultralytics trainer
        
        Args:
            cfg: Training configuration
        """
        if not hasattr(self, 'trainer'):
            self.setup_trainer(cfg)
        self.trainer.train()
        
    def process_batch(self, batch):
        """
        Process batch data into standardized format
        
        Args:
            batch: Input batch in YOLO format
            
        Returns:
            tuple: (images, targets) or (None, None) if invalid batch
        """
        if isinstance(batch, dict):
            images = batch['img']
            if 'cls' in batch and 'bboxes' in batch:
                cls = batch['cls'].view(-1, 1) if batch['cls'].dim() == 1 else batch['cls']
                batch_idx = batch['batch_idx'].view(-1, 1) if batch['batch_idx'].dim() == 1 else batch['batch_idx']
                unique_batch_idx = batch_idx.squeeze(-1)
                gt_groups = [(unique_batch_idx == i).sum().item() for i in range(images.shape[0])]
                targets = {
                    'cls': cls.squeeze(-1),
                    'bboxes': batch['bboxes'],
                    'batch_idx': batch_idx.squeeze(-1),
                    'gt_groups': gt_groups
                }
            else:
                LOGGER.warning("Missing cls or bboxes in batch")
                return None, None
        else:
            if not isinstance(batch, (list, tuple)):
                batch = [batch]
            images = batch[0]
            if len(batch) > 1:
                targets = batch[1]
                if isinstance(targets, torch.Tensor) and targets.shape[-1] == 6:
                    batch_idx = targets[:, 0]
                    gt_groups = [(batch_idx == i).sum().item() for i in range(images.shape[0])]
                    targets = {
                        'cls': targets[:, 1],
                        'bboxes': targets[:, 2:],
                        'batch_idx': batch_idx,
                        'gt_groups': gt_groups
                    }
            else:
                LOGGER.warning("No targets in batch")
                return None, None
            
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected images to be a tensor, got {type(images)}")
            
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
            
        return images, targets

    def compute_loss(self, images, targets):
        """
        Compute loss for a batch of data
        
        Args:
            images: Input images tensor
            targets: Target dictionary
            
        Returns:
            tuple: (total_loss, loss_dict, preds) or (None, None, None) if error
        """
        preds = None
        try:
            # Get backbone features
            backbone_features = self.model.backbone(images)
            
            # Pass through appropriate YOLO branch
            preds = self.yolo_branch(backbone_features)
            
            # Get loss from YOLO's internal criterion
            loss_dict = self.yolo_branch.yolo.model.criterion(preds, targets)
            
            # Sum all losses
            total_loss = sum(v for k, v in loss_dict.items() if v.requires_grad)
            return total_loss, loss_dict, preds
            
        except Exception as e:
            LOGGER.error(f"Error computing loss: {str(e)}")
            if preds is not None and isinstance(preds, (list, tuple)):
                LOGGER.error("Predictions shape:")
                for i, pred in enumerate(preds):
                    LOGGER.error(f"pred[{i}].shape = {pred.shape}")
            LOGGER.error("Targets:")
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    LOGGER.error(f"{k}.shape = {v.shape}, device = {v.device}")
            return None, None, None

    def train_step(self, batch):
        """
        Single training step - used by Lightning module
        
        Args:
            batch: Input batch in YOLO format
        """
        images, targets = self.process_batch(batch)
        if targets is None:
            return None
            
        self.yolo_branch.train()
        total_loss, _, _ = self.compute_loss(images, targets)
        return total_loss

    def validation_step(self, batch):
        """
        Single validation step
        
        Args:
            batch: Input batch in YOLO format
            
        Returns:
            tuple: (predictions, targets, loss_dict) or (None, None, None) if error
        """
        images, targets = self.process_batch(batch)
        if targets is None:
            return None, None, None
            
        self.yolo_branch.eval()
        with torch.no_grad():
            _, loss_dict, preds = self.compute_loss(images, targets)
            
        return preds, targets, loss_dict