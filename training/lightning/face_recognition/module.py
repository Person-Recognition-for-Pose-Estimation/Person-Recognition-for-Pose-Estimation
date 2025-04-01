"""
Lightning module for AdaFace face recognition training.
"""
import pytorch_lightning as pl
import torch  # type: ignore
from torch.optim import Adam  # type: ignore
import torch.nn.functional as F  # type: ignore
from typing import Dict, Any, Optional

class FaceRecognitionModule(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes: int = 85742,  # Default from MS1MV2
        embedding_size: int = 512,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        m: float = 0.4,  # Margin
        h: float = 0.333,  # Norm multiplier
        s: float = 64.,  # Scale
        t_alpha: float = 0.01,  # EMA decay rate
        **kwargs
    ):
        """
        Initialize AdaFace Lightning Module
        
        Args:
            model: Combined multi-task model
            num_classes: Number of identity classes in training set
            embedding_size: Size of face embeddings (512 for IR-50)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            m: Base margin for AdaFace
            h: Norm multiplier for margin adaptation
            s: Feature scale
            t_alpha: EMA decay rate for batch statistics
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model = model
        self.model.set_task('face_recognition')
        self.validation_step_outputs = []
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # # Initialize metrics
        # self.train_acc = pl.metrics.Accuracy()
        # self.val_acc = pl.metrics.Accuracy()
        
        # Register buffers for AdaFace statistics
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Single training step"""
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        else:
            images, labels = batch
            
        # Ensure inputs are tensors and properly formatted
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        if images.dtype != torch.float32:
            images = images.float()
            
        # Get embeddings and norms from model through the backbone
        embeddings, norms = self.model(images)
        
        # Update batch statistics
        with torch.no_grad():
            safe_norms = torch.clip(norms, min=0.001, max=100)
            mean = safe_norms.mean()
            std = safe_norms.std()
            self.batch_mean = mean * self.hparams.t_alpha + (1 - self.hparams.t_alpha) * self.batch_mean
            self.batch_std = std * self.hparams.t_alpha + (1 - self.hparams.t_alpha) * self.batch_std
        
        # Calculate margin
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + 1e-3)
        margin_scaler = margin_scaler * self.hparams.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        
        # Calculate adaptive margin
        kernel = F.normalize(self.model.ada_face.head.kernel)
        cosine = F.linear(F.normalize(embeddings), kernel.t())  # transpose kernel for correct matrix multiplication
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        
        # One-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Calculate margins
        g_angular = self.hparams.m * margin_scaler * -1
        theta = torch.acos(cosine)
        theta_m = torch.clip(theta + one_hot * g_angular, min=1e-7, max=3.14159 - 1e-7)
        cosine = torch.cos(theta_m)
        
        # Scale
        output = cosine * self.hparams.s
        
        # Calculate loss and accuracy
        loss = F.cross_entropy(output, labels)
        acc = (output.max(1)[1] == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        else:
            images, labels = batch
            
        # Ensure inputs are tensors and properly formatted
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        if images.dtype != torch.float32:
            images = images.float()
            
        # Get embeddings through the backbone
        embeddings, norms = self.model(images)
        
        # Calculate cosine similarities
        kernel = F.normalize(self.model.ada_face.head.kernel)
        cosine = F.linear(F.normalize(embeddings), kernel.t())  # transpose kernel for correct matrix multiplication
        
        # Scale
        output = cosine * self.hparams.s
        
        # Calculate loss and accuracy
        loss = F.cross_entropy(output, labels)
        acc = (output.max(1)[1] == labels).float().mean()

        # Create output dictionary
        output_dict = {'val_loss': loss, 'val_acc': acc}
        
        # Append to validation outputs
        self.validation_step_outputs.append(output_dict)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return output_dict
    
    def on_validation_epoch_end(self):
        """Called at the end of validation"""
        # Extract losses and accuracies from the validation outputs
        val_losses = torch.stack([x['val_loss'] for x in self.validation_step_outputs])
        val_accs = torch.stack([x['val_acc'] for x in self.validation_step_outputs])
        
        # Calculate means
        avg_loss = val_losses.mean()
        avg_acc = val_accs.mean()
        
        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/acc', avg_acc, prog_bar=True)

        # Clear the outputs list
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers"""
        # Optimize both adapter and AdaFace model parameters
        params = list(self.model.ada_face.adapter.parameters()) + \
                list(self.model.ada_face.parameters())
                
        optimizer = Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer