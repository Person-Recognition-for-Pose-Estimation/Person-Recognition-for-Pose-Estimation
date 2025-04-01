"""
Minimal test case to debug YOLO adapter gradient issues.
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class MinimalYOLOAdapter(nn.Module):
    def __init__(self, yolo_model, input_channels=2048):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(input_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((640, 640)),  # YOLO input size
            nn.Conv2d(512, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU()
        )
        self.yolo = yolo_model

    def forward(self, x):
        x = self.adapter(x)
        x = (x - x.min()) / (x.max() - x.min())  # Normalize to [0,1]
        return self.yolo.model(x)

class MinimalLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Load YOLO model
        yolo_model = YOLO('yolov8n.pt')
        
        # Create adapter
        self.yolo_adapter = MinimalYOLOAdapter(yolo_model)
        
        # Print parameter gradients
        print("\nInitial parameter gradient settings:")
        for name, param in self.yolo_adapter.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")

    def forward(self, x):
        return self.yolo_adapter(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        # Just use a dummy loss for testing
        loss = sum(p.sum() for p in preds)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        # Just use a dummy loss for testing
        loss = sum(p.sum() for p in preds)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    # Create dummy data
    batch_size = 2
    input_channels = 2048
    input_size = 64  # Small size for testing
    
    # Create random inputs and targets
    x = torch.randn(batch_size, input_channels, input_size, input_size)
    y = torch.randn(batch_size, 1)  # Dummy targets
    
    # Create dataset and dataloader
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset, batch_size=batch_size)
    
    # Initialize model and trainer
    model = MinimalLightningModule()
    
    # Print model structure
    print("\nModel structure:")
    for name, module in model.named_modules():
        print(f"{name}: {type(module).__name__}")
    
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False
    )
    
    try:
        # Try training
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        
        # Print gradient information for debugging
        print("\nGradient information at error:")
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}, "
                  f"grad_fn = {param.grad_fn if hasattr(param, 'grad_fn') else None}")

if __name__ == "__main__":
    main()