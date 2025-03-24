"""
Test script for YOLO face detection training with Lightning and Ultralytics integration.
"""
import os
import torch
import pathlib
import argparse
import pytorch_lightning as pl
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
from ultralytics import YOLO

from lightning.face_detection.module import YOLOLightningModule
from lightning.face_detection.datamodule import YOLOFaceDataModule
from lightning.callbacks import YOLOLoggingCallback, YOLOModelCheckpoint
# from lightning.models import (
#     CombinedModel,
#     MultiTaskResNetFeatureExtractor,
#     CustomYOLO,
#     CustomAdaFace,
#     CustomVitPose
# )

from modify_models import CustomAdaFace, CustomYOLO, CustomVitPose, MultiTaskResNetFeatureExtractor, CombinedModel, create_combined_model

def load_model():
    """Load the combined model with proper error handling"""
    try:
        # Get the path relative to this script
        filepath = Path(__file__).parent.resolve()
        components_dir = filepath.parent / "edited_components"
        model_path = components_dir / "combined_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Combined model not found at {model_path}. "
                "Please run the model creation notebook first."
            )

        # Create a new instance of CombinedModel with components
        model = create_combined_model(save_components=True)
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model
        
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        print("\nDebug information:")
        print(f"Script location: {filepath}")
        print(f"Model path: {model_path}")
        print(f"Components directory: {components_dir}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available files in components dir: {list(components_dir.glob('*.pth'))}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO face detection model')
    parser.add_argument('--logging', choices=['none', 'wandb'], default='none', help='Logging method to use')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--img-size', type=int, default=256, help='Image size') # 416
    parser.add_argument('--workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--accumulate', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--save-period', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--val-check-interval', type=float, default=1.0, help='Validation check interval')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print logging configuration
    print(f"Logging method: {args.logging}")
    
    # Configuration dictionary
    config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "amp": True,  # Always use AMP for memory efficiency
        "workers": args.workers,
        "val_check_interval": args.val_check_interval,
        "save_period": args.save_period,
        "accumulate": args.accumulate,
    }
    
    # Initialize logger
    logger = None
    if args.logging == 'wandb':
        print("Initializing W&B logging...")
        wandb.init(
            project="yolo-face-detection",
            name="training-run",
            config=config
        )
        logger = WandbLogger()
    
    # Print training configuration
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Add memory checks
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"Total GPU Memory: {gpu_memory:.0f}MB")
        if gpu_memory < 4000:  # Less than 4GB
            print("Limited GPU memory detected. Using memory-efficient settings.")
    
    # Load model
    model = load_model()
    
    # Setup data configuration
    filepath = pathlib.Path(__file__).parent.resolve()
    data_cfg = Path(os.path.join(filepath, "..", "dataset_folders", "yolo_face", "data.yaml"))
    if not data_cfg.exists():
        raise FileNotFoundError(f"Data config not found at {data_cfg}")
    
    # Create Lightning module with Ultralytics integration
    lightning_model = YOLOLightningModule(
        model=model,
        data_cfg=str(data_cfg),
        learning_rate=config["learning_rate"],
        # Ultralytics specific args
        epochs=config["epochs"],
        batch=config["batch_size"],
        imgsz=config["img_size"],
        device=config["device"],
        amp=config["amp"],
        workers=config["workers"],
        save_period=config["save_period"],
    )
    
    # Create data module
    data_module = YOLOFaceDataModule(
        data_dir=str(data_cfg.parent),
        batch_size=config["batch_size"],
        num_workers=config["workers"],
    )
    
    # Setup callbacks
    callbacks = [
        YOLOLoggingCallback(log_interval=100, logging_method=args.logging),
        YOLOModelCheckpoint(save_path="checkpoints"),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="yolo-face-{epoch:02d}-{val_mAP50-95:.2f}",
            monitor="val/mAP50-95",
            mode="max",
            save_top_k=3,
            every_n_epochs=config["save_period"],
        )
    ]
    
    # Setup trainer with optional W&B logger
    trainer_kwargs = {
        'max_epochs': config["epochs"],
        'accelerator': "gpu" if torch.cuda.is_available() else "cpu",
        'devices': 1,
        'callbacks': callbacks,
        'val_check_interval': config["val_check_interval"],
        'precision': 16 if config["amp"] else 32,
        'gradient_clip_val': 0.1,
        'accumulate_grad_batches': config["accumulate"],
        'log_every_n_steps': 10,
        'logger': logger,  # This will be None if not using W&B
    }
    
    # Create trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Setup memory monitoring
    def check_memory():
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"GPU Memory: Allocated={memory_allocated:.0f}MB, Reserved={memory_reserved:.0f}MB")
            if memory_allocated > 2500:  # Over 2.5GB
                print("WARNING: High GPU memory usage detected!")
    
    # Train model
    try:
        # Initial memory check
        print("\nInitial GPU Memory Status:")
        check_memory()
        
        # Start training
        print("\nStarting training...")
        trainer.fit(lightning_model, data_module)
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nERROR: Out of Memory!")
            print("Suggestions:")
            print(f"1. Reduce batch_size (currently {config['batch_size']})")
            print(f"2. Reduce image size (currently {config['img_size']})")
            print(f"3. Increase gradient accumulation (currently {config['accumulate']})")
        else:
            print(f"\nTraining failed with error: {str(e)}")
        raise
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise
    finally:
        # Final memory check
        print("\nFinal GPU Memory Status:")
        check_memory()
        
        # Save final model state
        if trainer.global_rank == 0:  # Save only on main process
            save_path = Path("checkpoints/final_model.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                save_dict = {
                    'model_state': model.state_dict(),
                    'config': config,
                    'epoch': trainer.current_epoch,
                }
                torch.save(save_dict, save_path)
                print(f"\nModel saved to {save_path}")
            except Exception as e:
                print(f"\nFailed to save model: {str(e)}")
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.logging == 'wandb':
            print("Finishing W&B run")
            wandb.finish()

if __name__ == "__main__":
    main()