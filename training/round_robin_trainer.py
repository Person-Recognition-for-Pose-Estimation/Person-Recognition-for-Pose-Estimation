"""
Round-robin trainer for multi-task learning that alternates between different tasks.

Current Tasks:
- Face Detection: YOLO-based face detection
- Person Detection: YOLO-based person detection using COCO dataset

Future Tasks (TODO):
- Face Recognition: AdaFace-based face recognition and embedding
- Pose Estimation: ViTPose-based human pose estimation

Training Strategy:
1. Each task is trained for one epoch in a round-robin fashion
2. The model switches between tasks using the set_task() method
3. Each task has its own:
   - Lightning module for training logic
   - Data module for data loading
   - Callbacks for logging and checkpointing
   - Metrics for evaluation

The trainer saves checkpoints after each task's epoch and maintains the overall
model state across task switches. This allows for gradual improvement in all
tasks while managing memory efficiently.
"""
import os
import torch
import pathlib
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Detection modules
from lightning.face_detection.module import YOLOLightningModule as FaceYOLOModule
from lightning.face_detection.datamodule import YOLOFaceDataModule
from lightning.person_detection.coco_module import COCOYOLOModule
from lightning.person_detection.coco_datamodule import COCOYOLODataModule

from lightning.face_recognition.module import AdaFaceLightningModule
from lightning.face_recognition.datamodule import FaceRecognitionDataModule

# TODO: Import pose estimation modules once implemented
# from lightning.pose_estimation.module import VitPoseLightningModule
# from lightning.pose_estimation.datamodule import PoseEstimationDataModule

from lightning.callbacks import YOLOLoggingCallback, YOLOModelCheckpoint
from modify_models import create_combined_model

@dataclass
class TaskConfig:
    """Configuration for a single task"""
    name: str
    module_class: type
    datamodule_class: type
    data_config: Dict
    module_config: Dict
    wandb_project: str

class RoundRobinTrainer:
    def __init__(
        self,
        model,
        tasks: List[TaskConfig],
        base_config: Dict,
        logging_method: str = 'none',
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize round-robin trainer.
        
        Args:
            model: Combined multi-task model
            tasks: List of task configurations
            base_config: Base configuration shared across tasks
            logging_method: Logging method ('none' or 'wandb')
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.tasks = tasks
        self.base_config = base_config
        self.logging_method = logging_method
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize task-specific components
        self.setup_tasks()
        
    def setup_tasks(self):
        """Setup trainers and data modules for each task"""
        self.task_trainers = {}
        self.task_datamodules = {}
        
        for task in self.tasks:
            # Create data module
            data_module = task.datamodule_class(**task.data_config)
            
            # Setup wandb logger if needed
            logger = None
            if self.logging_method == 'wandb':
                logger = WandbLogger(
                    project=task.wandb_project,
                    name=f"{task.name}-training",
                    config={**self.base_config, **task.module_config}
                )
            
            # Setup callbacks
            callbacks = [
                YOLOLoggingCallback(
                    log_interval=100,
                    logging_method=self.logging_method
                ),
                YOLOModelCheckpoint(
                    save_path=str(self.checkpoint_dir / task.name)
                ),
                ModelCheckpoint(
                    dirpath=str(self.checkpoint_dir / task.name),
                    filename=f"{task.name}-{{epoch:02d}}-{{val_mAP50-95:.2f}}",
                    monitor="val/mAP50-95",
                    mode="max",
                    save_top_k=3,
                    every_n_epochs=self.base_config.get("save_period", 1)
                )
            ]
            
            # Create Lightning module
            lightning_module = task.module_class(
                model=self.model,
                **task.module_config
            )
            
            # Setup trainer
            trainer = pl.Trainer(
                max_epochs=1,  # We'll control overall epochs in round-robin
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                callbacks=callbacks,
                val_check_interval=self.base_config.get("val_check_interval", 1.0),
                precision=16 if self.base_config.get("amp", True) else 32,
                gradient_clip_val=0.1,
                accumulate_grad_batches=self.base_config.get("accumulate", 1),
                log_every_n_steps=10,
                logger=logger,
                enable_checkpointing=True,
            )
            
            self.task_trainers[task.name] = trainer
            self.task_datamodules[task.name] = data_module
            
    def train(self, total_epochs: int):
        """
        Train all tasks in round-robin fashion.
        
        Args:
            total_epochs: Total number of epochs to train
        """
        try:
            for epoch in range(total_epochs):
                self.logger.info(f"\nStarting epoch {epoch + 1}/{total_epochs}")
                
                # Train each task for one epoch
                for task in self.tasks:
                    self.logger.info(f"\nTraining {task.name}")
                    
                    # Set model to current task
                    self.model.set_task(task.name)
                    
                    # Get trainer and data module
                    trainer = self.task_trainers[task.name]
                    data_module = self.task_datamodules[task.name]
                    
                    # Train for one epoch
                    trainer.fit(
                        trainer.lightning_module,
                        data_module,
                        ckpt_path=None  # Don't resume, we manage state
                    )
                    
                    # Save combined model state
                    self.save_checkpoint(epoch, task.name)
                    
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            # Cleanup
            if self.logging_method == 'wandb':
                wandb.finish()
                
    def save_checkpoint(self, epoch: int, task_name: str):
        """Save combined model checkpoint"""
        save_path = self.checkpoint_dir / f"combined_model_epoch{epoch}_{task_name}.pt"
        try:
            save_dict = {
                'model_state': self.model.state_dict(),
                'epoch': epoch,
                'last_task': task_name
            }
            torch.save(save_dict, save_path)
            self.logger.info(f"Saved combined model to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Round-robin training of multi-task model')
    # Basic training arguments
    parser.add_argument('--logging', choices=['none', 'wandb'], default='none')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    # Face detection arguments
    parser.add_argument('--face-data-cfg', type=str, required=True,
                      help='Path to YOLO format data config for face detection')
    
    # Person detection arguments
    parser.add_argument('--coco-train-path', type=str, required=True,
                      help='Path to COCO train annotations')
    parser.add_argument('--coco-val-path', type=str, required=True,
                      help='Path to COCO validation annotations')
    parser.add_argument('--coco-train-img-dir', type=str, required=True,
                      help='Path to COCO train images directory')
    parser.add_argument('--coco-val-img-dir', type=str, required=True,
                      help='Path to COCO validation images directory')
    
    # Face Recognition arguments
    parser.add_argument('--face-train-rec', type=str,
                      help='Path to face recognition training .rec file')
    parser.add_argument('--face-train-idx', type=str,
                      help='Path to face recognition training .idx file')
    parser.add_argument('--face-val-rec', type=str,
                      help='Path to face recognition validation .rec file')
    parser.add_argument('--face-val-idx', type=str,
                      help='Path to face recognition validation .idx file')
    parser.add_argument('--face-num-classes', type=int, default=70722,
                      help='Number of identity classes in face recognition dataset')
    parser.add_argument('--face-embedding-size', type=int, default=512,
                      help='Size of face embeddings')
    parser.add_argument('--face-margin', type=float, default=0.4,
                      help='Margin for AdaFace loss')
    parser.add_argument('--face-norm-multiplier', type=float, default=0.333,
                      help='Norm multiplier for AdaFace')
    parser.add_argument('--face-scale', type=float, default=64.0,
                      help='Scale for AdaFace')
    parser.add_argument('--face-ema-decay', type=float, default=0.01,
                      help='EMA decay rate for AdaFace batch statistics')
    # parser.add_argument('--face-val-dir', type=str,
    #                   help='Path to face recognition validation directory')
    
    # TODO: Add Pose Estimation arguments once implemented
    # parser.add_argument('--pose-train-annotations', type=str,
    #                   help='Path to pose estimation training annotations')
    # parser.add_argument('--pose-val-annotations', type=str,
    #                   help='Path to pose estimation validation annotations')
    args = parser.parse_args()
    
    # Base configuration
    base_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "amp": True,
        "workers": 4,
        "accumulate": 4,
        "save_period": 1,
        "val_check_interval": 1.0,
    }
    
    # Task configurations
    tasks = [
        # Face Detection Task
        TaskConfig(
            name="face_detection",
            module_class=FaceYOLOModule,
            datamodule_class=YOLOFaceDataModule,
            data_config={
                "data_dir": str(Path(args.face_data_cfg).parent),
                "batch_size": args.batch_size,
                "num_workers": base_config["workers"],
            },
            module_config={
                "data_cfg": args.face_data_cfg,
                "learning_rate": args.learning_rate,
                "epochs": 1,  # Single epoch per round
                "batch": args.batch_size,
                "imgsz": 640,
                "device": 0 if torch.cuda.is_available() else "cpu",
                "amp": base_config["amp"],
                "workers": base_config["workers"],
                "save_period": base_config["save_period"],
            },
            wandb_project="yolo-face-detection"
        ),
        
        # Person Detection Task
        TaskConfig(
            name="object_detection",
            module_class=COCOYOLOModule,
            datamodule_class=COCOYOLODataModule,
            data_config={
                "train_path": args.coco_train_path,
                "val_path": args.coco_val_path,
                "train_img_dir": args.coco_train_img_dir,
                "val_img_dir": args.coco_val_img_dir,
                "batch_size": args.batch_size,
                "num_workers": base_config["workers"],
                "img_size": 640,
            },
            module_config={
                "data_cfg": None,  # Not using YOLO data.yaml
                "num_classes": 80,  # COCO classes
                "learning_rate": args.learning_rate,
                "epochs": 1,  # Single epoch per round
                "batch": args.batch_size,
                "imgsz": 640,
                "device": 0 if torch.cuda.is_available() else "cpu",
                "amp": base_config["amp"],
                "workers": base_config["workers"],
                "save_period": base_config["save_period"],
            },
            wandb_project="yolo-person-detection"
        ),
        
        # Face Recognition Task
        TaskConfig(
            name="face_recognition",
            module_class=AdaFaceLightningModule,
            datamodule_class=FaceRecognitionDataModule,
            data_config={
                "data_dir": os.path.dirname(args.face_train_rec),
                "train_rec": os.path.basename(args.face_train_rec),
                "train_idx": os.path.basename(args.face_train_idx),
                "val_rec": os.path.basename(args.face_val_rec),
                "val_idx": os.path.basename(args.face_val_idx),
                "batch_size": args.batch_size,
                "num_workers": base_config["workers"],
            },
            module_config={
                "num_classes": args.face_num_classes,
                "embedding_size": args.face_embedding_size,
                "learning_rate": args.learning_rate,
                "m": args.face_margin,
                "h": args.face_norm_multiplier,
                "s": args.face_scale,
                "t_alpha": args.face_ema_decay,
            },
            wandb_project="adaface-recognition"
        ),
        
        # TODO: Add Pose Estimation Task once implemented
        # Expected configuration:
        # TaskConfig(
        #     name="pose_estimation",
        #     module_class=VitPoseLightningModule,
        #     datamodule_class=PoseEstimationDataModule,
        #     data_config={
        #         "train_annotations": "path/to/train.json",
        #         "val_annotations": "path/to/val.json",
        #         "batch_size": args.batch_size,
        #         "num_workers": base_config["workers"],
        #     },
        #     module_config={
        #         "learning_rate": args.learning_rate,
        #         "num_keypoints": 17,  # For COCO pose format
        #         ...
        #     },
        #     wandb_project="pose-estimation"
        # ),
    ]
    
    # Load model
    model = create_combined_model(save_components=True)
    
    # Create and run trainer
    trainer = RoundRobinTrainer(
        model=model,
        tasks=tasks,
        base_config=base_config,
        logging_method=args.logging
    )
    
    trainer.train(total_epochs=args.epochs)

if __name__ == "__main__":
    main()