"""
Test script for FiftyOne COCO data loading and preprocessing.
Tests data loading and batch formation without requiring the model.
"""
import torch
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .fiftyone_datamodule import FiftyOneCOCODataModule

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

def visualize_batch(batch, save_dir='test_outputs'):
    """Visualize a batch of images with their bounding boxes"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    images = batch['images']
    boxes = batch['boxes']
    labels = batch['labels']

    # print("boxes!!!", boxes)
    
    for i in range(min(4, len(images))):
        # Convert image back to numpy and denormalize
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Assuming normalization was done
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # Plot boxes
        valid_boxes = boxes[i][labels[i] != -1]  # Filter padding
        for box in valid_boxes:
            # Scale coordinates to image dimensions
            x1, y1, x2, y2 = box.tolist()
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
            
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
        
        plt.axis('off')
        plt.savefig(save_dir / f'sample_{i}.png', bbox_inches='tight', pad_inches=0)
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

def analyze_person_statistics(data_module, num_batches=5):
    """Analyze statistics about person detections"""
    print("\nAnalyzing person detection statistics...")
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    persons_per_image = []
    box_sizes = []
    
    for i, batch in enumerate(tqdm(train_loader, total=num_batches)):
        if i >= num_batches:
            break
            
        # Count valid person detections
        valid_labels = batch['labels'] != -1
        persons_per_image.extend(valid_labels.sum(dim=1).tolist())
        
        # Calculate box sizes
        boxes = batch['boxes']
        valid_boxes = boxes[valid_labels]
        if len(valid_boxes) > 0:
            widths = valid_boxes[:, 2] - valid_boxes[:, 0]
            heights = valid_boxes[:, 3] - valid_boxes[:, 1]
            areas = widths * heights
            box_sizes.extend(areas.tolist())
    
    print("\nPerson detection statistics:")
    print(f"Average persons per image: {np.mean(persons_per_image):.1f}")
    print(f"Max persons in an image: {max(persons_per_image)}")
    print(f"Images with no persons: {sum(np.array(persons_per_image) == 0)}")
    
    if box_sizes:
        print("\nBounding box statistics:")
        print(f"Average box area: {np.mean(box_sizes):.1f}")
        print(f"Min box area: {min(box_sizes):.1f}")
        print(f"Max box area: {max(box_sizes):.1f}")

def main():
    parser = argparse.ArgumentParser(description='Test FiftyOne COCO data loading')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--train-samples', type=int, default=100)
    parser.add_argument('--val-samples', type=int, default=20)
    parser.add_argument('--img-size', type=int, default=640)
    args = parser.parse_args()
    
    # Create data module
    data_module = FiftyOneCOCODataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        classes=['person'],
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
        
        # Analyze person statistics
        analyze_person_statistics(data_module)
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()