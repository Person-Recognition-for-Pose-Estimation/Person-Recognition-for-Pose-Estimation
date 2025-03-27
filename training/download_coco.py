"""
Script to pre-download COCO dataset with keypoints using FiftyOne.
This ensures all images are downloaded before training starts.
"""
import fiftyone as fo
import fiftyone.zoo as foz
import argparse

def download_coco(split="all"):
    """
    Download COCO dataset with keypoints.
    
    Args:
        split: Dataset split to download ("train", "validation", "test", or "all")
    """
    # Define splits to download
    if split == "all":
        splits = ["train", "validation", "test"]
    else:
        splits = [split]
    
    for split_name in splits:
        print(f"\nDownloading COCO {split_name} split...")
        
        # Create unique dataset name
        dataset_name = f"coco-2017-keypoints-{split_name}"
        
        try:
            # Try to load existing dataset
            dataset = fo.load_dataset(dataset_name)
            print(f"Found existing dataset '{dataset_name}'")
        except:
            print(f"Creating new dataset '{dataset_name}'")
            # Download dataset with keypoints and person detections
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split=split_name,
                label_types=["detections", "keypoints"],
                classes=["person"],
                only_matching=True,  # Only load samples with person annotations
                dataset_name=dataset_name
            )
        
        # Force download of all images by iterating through them
        print(f"Ensuring all images are downloaded...")
        total = len(dataset)
        for i, sample in enumerate(dataset.iter_samples(progress=True)):
            try:
                # Access filepath to trigger download
                _ = sample.filepath
            except Exception as e:
                print(f"Error with sample {sample.id}: {str(e)}")
        
        # Print statistics
        print(f"\nDataset statistics for {split_name}:")
        print(f"Total samples: {len(dataset)}")
        person_counts = [len([det for det in sample.ground_truth.detections if det.label == 'person']) 
                        for sample in dataset.iter_samples()]
        print(f"Total person detections: {sum(person_counts)}")
        print(f"Average persons per image: {sum(person_counts)/len(dataset):.1f}")
        
        # Print keypoint statistics
        keypoint_counts = [len([kp for kp in sample.ground_truth.keypoints if kp.label == 'person']) 
                          for sample in dataset.iter_samples()]
        print(f"Total keypoint annotations: {sum(keypoint_counts)}")
        print(f"Average keypoint annotations per image: {sum(keypoint_counts)/len(dataset):.1f}")

def main():
    parser = argparse.ArgumentParser(description="Download COCO dataset with keypoints")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Dataset split to download"
    )
    args = parser.parse_args()
    
    download_coco(args.split)

if __name__ == "__main__":
    main()