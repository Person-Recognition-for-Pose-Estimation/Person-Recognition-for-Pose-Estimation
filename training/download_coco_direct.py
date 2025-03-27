"""
Script to download COCO dataset directly without using FiftyOne.
Downloads images and annotations for person detection and keypoints.
"""
import os
import json
import requests
from pathlib import Path
import argparse
from tqdm import tqdm
import zipfile
import shutil

COCO_URLS = {
    # Images
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'test_images': 'http://images.cocodataset.org/zips/test2017.zip',
    # Annotations
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    # Person Keypoints
    'keypoints': 'http://images.cocodataset.org/annotations/person_keypoints_trainval2017.zip',
}

def download_file(url: str, dest_path: Path, desc: str = None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def extract_zip(zip_path: Path, extract_path: Path, desc: str = None):
    """Extract a zip file with progress bar."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in tqdm(zf.infolist(), desc=desc):
            try:
                zf.extract(member, extract_path)
            except zipfile.error as e:
                print(f"Error extracting {member.filename}: {e}")

def filter_person_annotations(ann_file: Path, output_file: Path):
    """Filter annotations to keep only person class and required fields."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Filter categories to keep only person
    person_cat = next(cat for cat in data['categories'] if cat['name'] == 'person')
    data['categories'] = [person_cat]
    
    # Filter annotations to keep only person instances
    data['annotations'] = [
        ann for ann in data['annotations']
        if ann['category_id'] == person_cat['id']
    ]
    
    # Get image IDs that have person annotations
    image_ids = set(ann['image_id'] for ann in data['annotations'])
    
    # Filter images to keep only those with person annotations
    data['images'] = [
        img for img in data['images']
        if img['id'] in image_ids
    ]
    
    # Save filtered annotations
    with open(output_file, 'w') as f:
        json.dump(data, f)

def download_coco(data_dir: str, splits: list = None):
    """
    Download and prepare COCO dataset.
    
    Args:
        data_dir: Directory to store the dataset
        splits: List of splits to download ('train', 'val', 'test')
    """
    if splits is None:
        splits = ['train', 'val']
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for downloads
    temp_dir = data_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download and extract images for each split
        for split in splits:
            split_key = f'{split}_images' if split != 'val' else 'val_images'
            if split_key in COCO_URLS:
                zip_path = temp_dir / f'{split}2017.zip'
                
                # Download images
                if not zip_path.exists():
                    print(f"\nDownloading {split} images...")
                    download_file(COCO_URLS[split_key], zip_path, f"Downloading {split} images")
                
                # Extract images
                images_dir = data_dir / 'images' / f'{split}2017'
                if not images_dir.exists() or not any(images_dir.iterdir()):
                    print(f"\nExtracting {split} images...")
                    images_dir.parent.mkdir(exist_ok=True)
                    extract_zip(zip_path, data_dir / 'images', f"Extracting {split} images")
        
        # Download and extract annotations and keypoints
        for ann_type in ['annotations', 'keypoints']:
            zip_path = temp_dir / f'{ann_type}.zip'
            if not zip_path.exists():
                print(f"\nDownloading {ann_type}...")
                download_file(COCO_URLS[ann_type], zip_path, f"Downloading {ann_type}")
            
            # Extract annotations
            ann_dir = data_dir / 'annotations'
            if not ann_dir.exists() or not any(ann_dir.iterdir()):
                print(f"\nExtracting {ann_type}...")
                extract_zip(zip_path, data_dir, f"Extracting {ann_type}")
        
        # Filter annotations to keep only person class
        print("\nFiltering annotations for person class...")
        for split in ['train', 'val']:
            if split in splits:
                orig_file = ann_dir / f'instances_{split}2017.json'
                filtered_file = ann_dir / f'person_instances_{split}2017.json'
                if not filtered_file.exists():
                    print(f"Filtering {split} annotations...")
                    filter_person_annotations(orig_file, filtered_file)
        
    finally:
        # Cleanup temporary files
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
    
    print("\nCOCO dataset download and preparation completed!")

def main():
    parser = argparse.ArgumentParser(description="Download COCO dataset directly")
    parser.add_argument(
        '--data-dir',
        type=str,
        default=os.path.expanduser('~/coco'),
        help='Directory to store the dataset. Use absolute path or ~/coco for home directory. Default: ~/coco'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=['train', 'val', 'test'],
        default=['train', 'val'],
        help='Dataset splits to download'
    )
    args = parser.parse_args()
    
    download_coco(args.data_dir, args.splits)

if __name__ == '__main__':
    main()