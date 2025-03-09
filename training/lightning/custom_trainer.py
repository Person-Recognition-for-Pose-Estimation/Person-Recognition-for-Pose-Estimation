"""
Custom Ultralytics trainer for multi-task model.
"""
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER

class CustomUltralyticsTrainer(BaseTrainer):
    def __init__(self, combined_model, **kwargs):
        super().__init__(**kwargs)
        self.combined_model = combined_model
        
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return our combined model instead of loading a new one"""
        return self.combined_model
            
    def preprocess_batch(self, batch):
        """Preprocess batch - no changes needed as our model handles the backbone"""
        return batch
            
    def progress_string(self):
        """Return progress string with task information"""
        return f'task: face_detection\n{super().progress_string()}'
        
    def get_model_state(self):
        """Get model state dict for saving"""
        return {
            'model': self.combined_model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
        }
        
    def set_model_state(self, state_dict):
        """Set model state from state dict"""
        if state_dict.get('model'):
            self.combined_model.load_state_dict(state_dict['model'])
        if state_dict.get('optimizer') and self.optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if state_dict.get('epoch'):
            self.epoch = state_dict['epoch']
        if state_dict.get('best_fitness'):
            self.best_fitness = state_dict['best_fitness']