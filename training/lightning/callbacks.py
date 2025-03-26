"""
Custom callbacks for Lightning training.
"""
from pytorch_lightning.callbacks import Callback
import wandb
import torch

class YOLOLoggingCallback(Callback):
    """Callback for logging YOLO-specific metrics and visualizations"""
    def __init__(self, log_interval: int = 100, logging_method: str = 'none'):
        """
        Initialize callback
        
        Args:
            log_interval: Number of batches between logging visualizations
            logging_method: Logging method to use ('none' or 'wandb')
        """
        super().__init__()
        self.log_interval = log_interval
        self.logging_method = logging_method
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log metrics and visualizations after each training batch"""
        # Log visualizations periodically
        if batch_idx % self.log_interval == 0:
            # Extract images and targets from batch
            if isinstance(batch, dict):
                images = batch['img']
                targets = batch.get('targets', None)
            else:
                # Fallback to previous format if needed
                if not isinstance(batch, (list, tuple)):
                    batch = [batch]
                images = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            
            # Ensure images is a tensor and convert to float32
            if not isinstance(images, torch.Tensor):
                raise ValueError(f"Expected images to be a tensor, got {type(images)}")
            
            # Convert to float32 and normalize to [0, 1]
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            
            # Get backbone features first
            with torch.no_grad():
                features = pl_module.model.backbone(images)
                preds = pl_module.model.yolo_face(features)
            
            # TODO: Create visualization of predictions vs ground truth
            # This will be implemented once we confirm the data format
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at the end of each epoch"""
        metrics = pl_module.metrics.results_dict
        
        # Create metrics dictionary
        val_metrics = {
            "val/mAP50": metrics.get("metrics/mAP50", 0),
            "val/mAP50-95": metrics.get("metrics/mAP50-95", 0),
            "val/precision": metrics.get("metrics/precision", 0),
            "val/recall": metrics.get("metrics/recall", 0)
        }
        
        # Log metrics to appropriate logger
        if self.logging_method == 'wandb':
            if wandb.run is not None:
                wandb.log(val_metrics)
            else:
                print("W&B logging enabled but wandb.run is None. Skipping W&B logging.")
        
        # Always log to console for visibility
        print("\nValidation Metrics:")
        for name, value in val_metrics.items():
            print(f"{name}: {value:.4f}")

class YOLOModelCheckpoint(Callback):
    """Callback for saving YOLO model checkpoints"""
    def __init__(self, save_path: str = "checkpoints"):
        """
        Initialize callback
        
        Args:
            save_path: Directory to save checkpoints
        """
        super().__init__()
        self.save_path = save_path
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Save model if validation metrics improve"""
        current_epoch = trainer.current_epoch
        metrics = pl_module.metrics.results_dict
        
        # Get current mAP
        current_map = metrics.get("metrics/mAP50-95", 0)
        
        # Save if it's the best model so far
        if trainer.checkpoint_callback.best_model_score is None or current_map > trainer.checkpoint_callback.best_model_score:
            trainer.checkpoint_callback.best_model_score = current_map