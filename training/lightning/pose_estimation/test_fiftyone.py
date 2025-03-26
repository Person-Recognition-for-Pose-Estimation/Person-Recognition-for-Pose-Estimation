"""
Test script for FiftyOne COCO keypoint data loading and preprocessing.
Tests data loading and batch formation without requiring the model.
"""
import torch
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .fiftyone_datamodule import FiftyOneCOCOKeypointDataModule

# COCO keypoint names for visualization
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Keypoint connections for visualization
KEYPOINT_CONNECTIONS = [
    # Face
    ('left_ear', 'left_eye'), ('right_ear', 'right_eye'),
    ('left_eye', 'nose'), ('right_eye', 'nose'),
    # Arms
    ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'), ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    # Torso
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    # Legs
    ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')
]

def test_data_loading(data_module, num_batches=2):
    """Test basic data loading functionality"""
    print("\nTesting data loading...")
    
    # Setup data module
    data_module.setup()
    
    # Test train loader
    print("\nTesting train loader:")
    train_loader = data_module.train_dataloader()
    for i, batch in enumerate(tqdm(train_loader)):
        if i >= num_batches:
            break
        print(f"\nBatch {i} contents:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{k}: type={type(v)}")
                
        # Check image values
        images = batch['images']
        print(f"\nImage statistics:")
        print(f"Min value: {images.min().item():.3f}")
        print(f"Max value: {images.max().item():.3f}")
        print(f"Mean value: {images.mean().item():.3f}")
        print(f"Std value: {images.std().item():.3f}")
        
        # Check keypoint statistics
        keypoints = batch['keypoints']
        masks = batch['masks']
        valid_keypoints = keypoints[masks]
        if len(valid_keypoints) > 0:
            print(f"\nKeypoint statistics:")
            print(f"Valid keypoints: {len(valid_keypoints)}")
            print(f"X range: [{valid_keypoints[..., 0].min():.3f}, {valid_keypoints[..., 0].max():.3f}]")
            print(f"Y range: [{valid_keypoints[..., 1].min():.3f}, {valid_keypoints[..., 1].max():.3f}]")
            print(f"Visibility distribution:")
            for v in range(3):
                count = (valid_keypoints[..., 2] == v).sum().item()
                print(f"  {v}: {count} ({count/valid_keypoints.numel()*100:.1f}%)")

def visualize_batch(batch, save_dir='test_outputs/keypoints'):
    """Visualize a batch of images with their keypoint annotations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    images = batch['images']
    keypoints = batch['keypoints']
    masks = batch['masks']
    
    # Create keypoint name to index mapping
    keypoint_idx = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
    
    for i in range(min(4, len(images))):
        # Convert image back to numpy and denormalize
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # Plot keypoints for each person
        valid_instances = masks[i]
        instance_keypoints = keypoints[i][valid_instances]
        
        for person_keypoints in instance_keypoints:
            # Plot each keypoint
            for j, (x, y, v) in enumerate(person_keypoints):
                if v > 0:  # Only plot visible or occluded keypoints
                    color = 'yellow' if v == 2 else 'red'  # yellow for visible, red for occluded
                    plt.plot(x * width, y * height, 'o', color=color, markersize=5)
            
            # Plot connections
            for start_name, end_name in KEYPOINT_CONNECTIONS:
                start_idx = keypoint_idx[start_name]
                end_idx = keypoint_idx[end_name]
                
                start_kp = person_keypoints[start_idx]
                end_kp = person_keypoints[end_idx]
                
                if start_kp[2] > 0 and end_kp[2] > 0:  # Both keypoints should be annotated
                    plt.plot(
                        [start_kp[0] * width, end_kp[0] * width],
                        [start_kp[1] * height, end_kp[1] * height],
                        '-', color='cyan', linewidth=1, alpha=0.7
                    )
        
        plt.axis('off')
        plt.savefig(save_dir / f'keypoint_sample_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

def test_memory_usage(data_module, num_batches=5):
    """Test memory usage during data loading"""
    print("\nTesting memory usage...")
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    max_memory = initial_memory
    
    for i, batch in enumerate(tqdm(train_loader, total=num_batches)):
        if i >= num_batches:
            break
            
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            max_memory = max(max_memory, current_memory)
            
        # Clear batch to free memory
        del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {initial_memory / 1024**2:.1f}MB")
        print(f"Peak GPU memory: {max_memory / 1024**2:.1f}MB")
        print(f"Memory increase: {(max_memory - initial_memory) / 1024**2:.1f}MB")

def analyze_keypoint_statistics(data_module, num_batches=5):
    """Analyze statistics about keypoint annotations"""
    print("\nAnalyzing keypoint statistics...")
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    keypoints_per_image = []
    visibility_stats = {0: 0, 1: 0, 2: 0}  # Count for each visibility value
    keypoint_presence = {name: 0 for name in KEYPOINT_NAMES}  # Count of visible keypoints by type
    
    total_instances = 0
    total_keypoints = 0
    
    for i, batch in enumerate(tqdm(train_loader, total=num_batches)):
        if i >= num_batches:
            break
            
        masks = batch['masks']
        keypoints = batch['keypoints']
        
        # Count instances per image
        instances_per_image = masks.sum(dim=1).tolist()
        keypoints_per_image.extend(instances_per_image)
        
        # Analyze visible instances
        valid_keypoints = keypoints[masks]
        total_instances += len(valid_keypoints)
        total_keypoints += valid_keypoints.numel() // 3
        
        # Count visibility values
        for v in range(3):
            visibility_stats[v] += (valid_keypoints[..., 2] == v).sum().item()
        
        # Count keypoint presence by type
        for idx, name in enumerate(KEYPOINT_NAMES):
            visible = (valid_keypoints[:, idx, 2] > 0).sum().item()
            keypoint_presence[name] += visible
    
    print("\nDataset statistics:")
    print(f"Average instances per image: {np.mean(keypoints_per_image):.1f}")
    print(f"Max instances in an image: {max(keypoints_per_image)}")
    print(f"Total person instances: {total_instances}")
    print(f"Total keypoints: {total_keypoints}")
    
    print("\nVisibility statistics:")
    for v, count in visibility_stats.items():
        print(f"Visibility {v}: {count} ({count/total_keypoints*100:.1f}%)")
    
    print("\nKeypoint presence statistics:")
    for name, count in keypoint_presence.items():
        print(f"{name}: {count} ({count/total_instances*100:.1f}% of instances)")

def main():
    parser = argparse.ArgumentParser(description='Test FiftyOne COCO keypoint data loading')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--train-samples', type=int, default=100)
    parser.add_argument('--val-samples', type=int, default=20)
    parser.add_argument('--img-size', type=int, default=640)
    args = parser.parse_args()
    
    # Create data module
    data_module = FiftyOneCOCOKeypointDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        img_size=args.img_size,
    )
    
    try:
        # Test data loading
        test_data_loading(data_module)
        
        # Get a batch for visualization
        data_module.setup()
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        # Visualize batch
        print("\nGenerating visualizations...")
        visualize_batch(batch)
        
        # Test memory usage
        test_memory_usage(data_module)
        
        # Analyze keypoint statistics
        analyze_keypoint_statistics(data_module)
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()