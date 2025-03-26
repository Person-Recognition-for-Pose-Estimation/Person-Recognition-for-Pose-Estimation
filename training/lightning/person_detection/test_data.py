"""
Test script for COCO data loading and preprocessing without model.
"""
import os
import torch
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .coco_datamodule import COCOYOLODataModule

def test_single_batch(data_module, batch_idx=0):
    """Test loading and preprocessing of a single batch"""
    # Setup data module
    data_module.setup()
    
    # Get dataloader
    train_loader = data_module.train_dataloader()
    
    # Get a single batch
    for i, batch in enumerate(train_loader):
        if i == batch_idx:
            break
    
    print("\nBatch contents:")
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
    
    return batch

def visualize_batch(batch, save_dir='test_outputs'):
    """Visualize a batch of images with their bounding boxes"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    images = batch['images']
    boxes = batch['boxes']
    labels = batch['labels']
    
    for i in range(min(4, len(images))):
        # Convert image back to numpy
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize
        
        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # Plot boxes
        valid_boxes = boxes[i][labels[i] != -1]  # Filter padding
        for box in valid_boxes:
            x1, y1, x2, y2 = box.tolist()
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-')
        
        plt.axis('off')
        plt.savefig(save_dir / f'sample_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

def test_memory_usage(data_module, num_batches=10):
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

def test_batch_collation(data_module, num_batches=5):
    """Test batch collation with different numbers of objects"""
    print("\nTesting batch collation...")
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    # Track statistics
    box_counts = []
    padding_ratios = []
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
            
        # Count valid boxes (non-padding)
        valid_boxes = (batch['labels'] != -1).sum(dim=1)
        box_counts.extend(valid_boxes.tolist())
        
        # Calculate padding ratio
        total_boxes = batch['boxes'].shape[1]
        padding_ratios.extend((1 - valid_boxes / total_boxes).tolist())
    
    print(f"Box count statistics:")
    print(f"Mean boxes per image: {np.mean(box_counts):.1f}")
    print(f"Min boxes: {min(box_counts)}")
    print(f"Max boxes: {max(box_counts)}")
    print(f"\nPadding statistics:")
    print(f"Mean padding ratio: {np.mean(padding_ratios):.2%}")
    print(f"Max padding ratio: {max(padding_ratios):.2%}")

def main():
    parser = argparse.ArgumentParser(description='Test COCO data loading')
    parser.add_argument('--train-path', type=str, required=True,
                      help='Path to COCO train annotations')
    parser.add_argument('--val-path', type=str, required=True,
                      help='Path to COCO val annotations')
    parser.add_argument('--train-img-dir', type=str, required=True,
                      help='Path to COCO train images')
    parser.add_argument('--val-img-dir', type=str, required=True,
                      help='Path to COCO val images')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    
    # Create data module
    data_module = COCOYOLODataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        train_img_dir=args.train_img_dir,
        val_img_dir=args.val_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Run tests
    try:
        # Test single batch
        print("Testing single batch loading...")
        batch = test_single_batch(data_module)
        
        # Visualize batch
        print("\nVisualizing batch...")
        visualize_batch(batch)
        
        # Test memory usage
        test_memory_usage(data_module)
        
        # Test batch collation
        test_batch_collation(data_module)
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()