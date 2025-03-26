"""
Tests for Lightning module and Ultralytics wrapper.
"""
import unittest
import torch
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from training.modify_models import create_combined_model
from training.lightning.face_detection.module import FaceYOLOModule
from training.lightning.ultralytics_wrapper import UltralyticsTrainerWrapper

class TestYOLOLightning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model and data once for all tests"""
        # Get device
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create model
        cls.model = create_combined_model(save_components=False, device=cls.device)
        cls.model.set_task('face_detection')
        
        # Create sample data
        cls.batch_size = 2
        cls.img_size = 640
        cls.num_targets = 3
        
        # Sample images
        cls.images = torch.rand(cls.batch_size, 3, cls.img_size, cls.img_size).to(cls.device)
        
        # Sample targets in YOLO format
        # Each target: [batch_idx, class_idx, x, y, w, h]
        cls.targets = torch.zeros(cls.num_targets, 6).to(cls.device)
        cls.targets[:, 0] = torch.tensor([0, 0, 1], device=cls.device)  # batch indices
        cls.targets[:, 1] = 0  # class 0 (face)
        cls.targets[:, 2:] = torch.rand(cls.num_targets, 4, device=cls.device)  # random boxes
        
        # Create dict format targets
        cls.dict_targets = {
            'img': cls.images,
            'cls': cls.targets[:, 1].to(cls.device),
            'bboxes': cls.targets[:, 2:].to(cls.device),
            'batch_idx': cls.targets[:, 0].to(cls.device)
        }
        
        # Create Lightning module
        data_cfg = os.path.join(project_root, "dataset_folders", "yolo_face", "data.yaml")
        if not os.path.exists(data_cfg):
            # Create a temporary data.yaml for testing
            os.makedirs(os.path.dirname(data_cfg), exist_ok=True)
            with open(data_cfg, 'w') as f:
                f.write("""
path: dataset_folders/yolo_face
train: train/images
val: val/images
names:
  0: face
                """.strip())
                
        cls.lightning_module = FaceYOLOModule(
            model=cls.model,
            data_cfg=str(data_cfg)
        )
        
    def test_batch_processing(self):
        """Test batch processing with different formats"""
        wrapper = UltralyticsTrainerWrapper(self.model)
        
        # Test list format
        images, targets = wrapper.process_batch([self.images, self.targets])
        self.assertIsNotNone(images)
        self.assertIsNotNone(targets)
        self.assertTrue(torch.equal(images, self.images))
        self.assertEqual(set(targets.keys()), {'cls', 'bboxes', 'batch_idx', 'gt_groups'})
        
        # Test dict format
        images, targets = wrapper.process_batch(self.dict_targets)
        self.assertIsNotNone(images)
        self.assertIsNotNone(targets)
        self.assertTrue(torch.equal(images, self.images))
        self.assertEqual(set(targets.keys()), {'cls', 'bboxes', 'batch_idx', 'gt_groups'})
        
    def test_forward_pass(self):
        """Test forward pass through model"""
        wrapper = UltralyticsTrainerWrapper(self.model)
        
        # Process batch
        images, targets = wrapper.process_batch([self.images, self.targets])
        
        # Get backbone features
        backbone_features = self.model.backbone(images)
        self.assertEqual(backbone_features.shape[1], 2048)  # ResNet50 output channels
        
        # Test YOLO branch
        preds = self.model.yolo_face(backbone_features)
        self.assertIsInstance(preds, (list, tuple))
        self.assertTrue(len(preds) > 0)
        
    def test_loss_computation(self):
        """Test loss computation"""
        wrapper = UltralyticsTrainerWrapper(self.model)
        
        # Process batch
        images, targets = wrapper.process_batch([self.images, self.targets])
        
        # Compute loss
        total_loss, loss_dict, preds = wrapper.compute_loss(images, targets)
        
        # Check loss values
        self.assertIsNotNone(total_loss)
        self.assertIsNotNone(loss_dict)
        self.assertGreater(len(loss_dict), 0)
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in loss_dict.values()))
        
    def test_training_step(self):
        """Test full training step"""
        # Test with list format
        loss = self.lightning_module.training_step([self.images, self.targets], 0)
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)
        
        # Test with dict format
        loss = self.lightning_module.training_step(self.dict_targets, 0)
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)
        
    def test_validation_step(self):
        """Test validation step and metrics"""
        # Run validation step
        self.lightning_module.validation_step([self.images, self.targets], 0)
        
        # Check metrics
        results = self.lightning_module.metrics.results_dict
        self.assertIsNotNone(results)
        self.assertIn('metrics/precision', results)
        self.assertIn('metrics/recall', results)
        
        # Check mAP
        maps = self.lightning_module.metrics.maps
        self.assertIsNotNone(maps)
        self.assertIn(50, maps)  # mAP@0.5
        
if __name__ == '__main__':
    unittest.main()