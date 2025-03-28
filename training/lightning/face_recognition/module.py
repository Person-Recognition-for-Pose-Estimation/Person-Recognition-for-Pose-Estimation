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
        num_classes: int = 70722,  # Default from MS1MV2
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
        images, labels = batch
        
        # Get embeddings and norms from model
        embeddings, norms = self.model.ada_face(images)
        
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
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.model.ada_face.head.kernel))
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
        images, labels = batch
        
        # Get embeddings
        embeddings, norms = self.model.ada_face(images)
        
        # Calculate cosine similarities
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.model.ada_face.head.kernel))
        
        # Scale
        output = cosine * self.hparams.s
        
        # Calculate loss and accuracy
        loss = F.cross_entropy(output, labels)
        acc = (output.max(1)[1] == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        """Called at the end of validation"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/acc', avg_acc, prog_bar=True)
    
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