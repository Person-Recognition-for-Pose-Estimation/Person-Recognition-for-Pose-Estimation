import pathlib
import torch  # type: ignore
import torchvision.models as models  # type: ignore
import torch.nn as nn  # type: ignore
import os
from typing import Optional, Tuple

def get_project_root() -> pathlib.Path:
    """Get the project root directory."""
    return pathlib.Path(__file__).parent.parent.resolve()

def get_model_paths() -> Tuple[pathlib.Path, pathlib.Path]:
    """Get paths to component_models and edited_components directories."""
    root = get_project_root()
    component_models = root / "component_models"
    edited_components = root / "edited_components"
    
    if not edited_components.exists():
        edited_components.mkdir(parents=True)
        
    return component_models, edited_components

#################### YOLO ####################

import sys
from pathlib import Path


import sys
sys.path.append('/home/ubuntu/thesis/Person-Recognition-for-Pose-Estimation/training/yolopt')


from yolopt.nets.nn import YOLO, yolo_v11_n, Conv, Head

class CustomYOLO(nn.Module):
    def __init__(self, yolo_model, backbone_channels: int = 2048):
        super().__init__()
        
        # Feature adaptation with spatial upsampling and channel reduction
        self.adapter = nn.Sequential(
            # Initial dimensionality reduction to save memory before upsampling
            nn.Conv2d(backbone_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            
            # Upsample to larger spatial dimensions for YOLO
            nn.Upsample(size=(160, 160), mode='bilinear', align_corners=True),
            
            # Spatial feature processing with residual connection
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            
            # Progressive channel reduction while maintaining spatial information
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            # Final adaptation to match YOLO input (3 channels)
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.SiLU()
        )
        
        # First create with 80 classes (COCO default)
        self.yolo = yolo_model
        
    def forward(self, x):
        print("\n=== CustomYOLO Forward Pass ===")
        print(f"Input shape: {x.shape}")
        
        # Adapter network
        x = self.adapter(x)
        print(f"After adapter shape: {x.shape}")
        
        x = x - x.mean(dim=(2, 3), keepdim=True)
        x = x / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
        x = torch.sigmoid(x)
        print(f"After normalization shape: {x.shape}")
        
        # YOLO backbone and FPN
        darknet_features = self.yolo.net(x)
        print(f"DarkNet output shapes: {[f.shape for f in darknet_features]}")
        
        fpn_features = self.yolo.fpn(darknet_features)
        print(f"FPN output shapes: {[f.shape for f in fpn_features]}")
        
        head_output = self.yolo.head(list(fpn_features))
        if isinstance(head_output, list):
            print(f"Head output (training): {[h.shape for h in head_output]}")
        else:
            print(f"Head output (inference) shape: {head_output.shape}")
        
        if self.training:
            return head_output  # List of feature maps for training
        else:
            # Process predictions into boxes and scores
            return self.process_predictions(head_output)
            
    def process_predictions(self, predictions):
        """Process YOLO predictions into boxes and scores."""
        print("\n=== Processing Predictions ===")
        print(f"Input predictions type: {type(predictions)}")
        if isinstance(predictions, list):
            print("Processing training mode predictions")
            # Training mode - process feature maps
            all_boxes = []
            all_scores = []
            for i, feat_map in enumerate(predictions):
                print(f"\nFeature map {i} shape: {feat_map.shape}")
                # Split predictions into boxes and scores
                boxes = feat_map[..., :4]  # First 4 channels are box predictions
                scores = feat_map[..., 4:]  # Remaining channels are class scores
                print(f"Split shapes - boxes: {boxes.shape}, scores: {scores.shape}")
                
                # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
                boxes = self.convert_boxes_cxcywh_to_xyxy(boxes)
                print(f"After conversion - boxes shape: {boxes.shape}")
                
                all_boxes.append(boxes)
                all_scores.append(scores)
            
            # Concatenate predictions from all feature levels
            pred_boxes = torch.cat(all_boxes, dim=1)
            pred_scores = torch.cat(all_scores, dim=1)
            print(f"\nFinal concatenated shapes - boxes: {pred_boxes.shape}, scores: {pred_scores.shape}")
            
            final_output = torch.cat([pred_boxes, pred_scores], dim=1)
            print(f"Final output shape: {final_output.shape}")
            return final_output
        else:
            print("Processing inference mode predictions")
            print(f"Predictions shape: {predictions.shape}")
            return predictions
            
    def convert_boxes_cxcywh_to_xyxy(self, boxes):
        """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format"""
        if boxes.size(-1) != 4:
            boxes = boxes.transpose(-1, -2)  # Move channels to last dimension if needed
            
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return torch.stack([x1, y1, x2, y2], dim=-1)

def modify_yolo(model_path):
    print("Loading YOLO model...")
    model = torch.load(str(model_path), map_location='cpu', weights_only=False)
    print(f"\nCheckpoint type: {type(model)}")
    
    # Extract model
    if isinstance(model, dict):
        if 'model' in model:
            print("Found 'model' key in checkpoint")
            model = model['model']

    width = [3, 16, 32, 64, 128, 256]
    new_head = Head(nc=1, filters=(width[3], width[4], width[5]))

    for i in range(len(model.head.box)):
        new_head.box[i].load_state_dict(model.head.box[i].state_dict())

    for i in range(len(model.head.cls)):
        # Transfer all layers except the final classification layer
        for j in range(len(new_head.cls[i]) - 1):
            new_head.cls[i][j].load_state_dict(model.head.cls[i][j].state_dict())

    model.head = new_head
    
    return CustomYOLO(model)


def create_yolo_branches(
    component_models_dir: pathlib.Path,
    edited_components_dir: pathlib.Path,
    save_components: bool = False,
    device: str = 'cpu'
) -> Tuple[CustomYOLO, CustomYOLO]:
    """Create YOLO branches for person and face detection."""

    # Person detection (1 class - person)
    person_model_path = component_models_dir / "yolo11n.pt"
    person_detect_branch = modify_yolo(person_model_path)
    print("Initialized new classification head for person detection")

    # face detection (1 class - face)
    face_model_path = component_models_dir / "yolo11n.pt"
    face_detect_branch = modify_yolo(face_model_path)
    print("Initialized new classification head for person detection")
    
    if save_components:
        torch.save(person_detect_branch.state_dict(), edited_components_dir / "custom_yolo.pth")
        torch.save(face_detect_branch.state_dict(), edited_components_dir / "custom_yolo_face.pth")
    
    return person_detect_branch.to(device), face_detect_branch.to(device)


#################### AdaFace ####################


import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs'))

if project_root not in sys.path:
    sys.path.append(project_root)


import net_adaface
from net_adaface import build_model
from head_adaface import build_head


class CustomAdaFace(nn.Module):
    def __init__(self, pretrained_path, config, backbone_channels=2048):
        super(CustomAdaFace, self).__init__()
        
        # Define adapter network for backbone features (2048 channels)
        self.adapter = nn.Sequential(
            # Initial channel reduction
            nn.Conv2d(backbone_channels, 512, kernel_size=1),  # 1x1 conv for channel reduction
            nn.BatchNorm2d(512),
            nn.PReLU(512),
            
            # Spatial upsampling to match AdaFace input size (112x112)
            nn.Upsample(size=(112, 112), mode='bilinear', align_corners=True),
            
            # Feature processing after upsampling
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(256),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            
            # Final adaptation to match AdaFace input
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        
        # Load the complete pre-trained AdaFace model
        self.adaface_model = build_model(model_name=config.arch)
        
        checkpoint = torch.load(pretrained_path, weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load weights except for input layer
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('input_layer')}
        self.adaface_model.load_state_dict(filtered_state_dict, strict=False)
        
        # Replace input layer with a simpler one since adapter already did most processing
        self.adaface_model.input_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        
        # Original AdaFace head
        self.head = build_head(
            head_type=config.head,
            embedding_size=512,
            class_num=config.num_classes,
            m=config.m,
            h=config.h,
            t_alpha=config.t_alpha,
            s=config.s
        )
    
    def forward(self, x, labels=None):
        # Pass through adapter first
        x = self.adapter(x)
        
        # Forward through modified AdaFace model
        embeddings, norms = self.adaface_model(x)
        
        if labels is not None:
            return self.head(embeddings, norms, labels)
        return embeddings, norms


class Config:
    def __init__(self):
        self.arch = 'ir_50'
        self.head = 'adaface'
        self.num_classes = 85742  # Updated to match actual number of identities
        self.embedding_size = 512
        self.backbone_channels = 2048
        self.m = 0.4
        self.h = 0.333
        self.t_alpha = 0.01
        self.s = 64.0

config = Config()


def create_adaface_branch(
    component_models_dir: pathlib.Path,
    edited_components_dir: pathlib.Path,
    save_components: bool = False,
    device: str = 'cpu'
) -> CustomAdaFace:
    """Create AdaFace branch for face recognition."""
    
    config = Config()
    ada_face_model_path = component_models_dir / "adaface_ir50_ms1mv2.ckpt"
    face_rec_branch = CustomAdaFace(str(ada_face_model_path), config)
    
    if save_components:
        torch.save(face_rec_branch.state_dict(), edited_components_dir / "custom_ada_face.pth")
    
    return face_rec_branch.to(device)


#################### ViTPose ####################

from transformers import AutoProcessor, VitPoseForPoseEstimation, VitPoseConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class CustomVitPose(nn.Module):
    def __init__(self, vit_pose_model: VitPoseForPoseEstimation, backbone_channels=2048):
        super(CustomVitPose, self).__init__()
        
        self.adapter = nn.Sequential(
            # Initial channel reduction
            nn.Conv2d(backbone_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            
            # Upsample to ViTPose input size (256x192)
            nn.Upsample(size=(256, 192), mode='bilinear', align_corners=True),
            
            # Progressive channel reduction to 3 channels
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # Final adaptation to RGB-like channels
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.GELU()
        )
        
        self.vit_pose = vit_pose_model
        
        # Freeze normalization layers in ViT for stability
        for module in self.vit_pose.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                module.eval()
    
    def forward(self, backbone_features):
        x = self.adapter(backbone_features)
        return self.vit_pose(x)


def create_vitpose_branch(
    edited_components_dir: pathlib.Path,
    save_components: bool = False,
    device: str = 'cpu'
) -> CustomVitPose:
    """Create ViTPose branch for pose estimation."""

    # config = VitPoseConfig(return_dict=False) 

    hf_vit_pose_model = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-base-simple",
        device_map=device,
        # config=config
    )
    pose_detect_branch = CustomVitPose(hf_vit_pose_model)
    
    if save_components:
        torch.save(pose_detect_branch.state_dict(), edited_components_dir / "custom_vit_pose.pth")
    
    return pose_detect_branch.to(device)


#################### Backbone ####################


class MultiTaskResNetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(MultiTaskResNetFeatureExtractor, self).__init__()
        
        # Just the core ResNet layers
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        # Basic feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # Output shape: [batch_size, 2048, H/32, W/32]


def create_backbone(
    edited_components_dir: pathlib.Path,
    save_components: bool = False,
    device: str = 'cpu'
) -> MultiTaskResNetFeatureExtractor:
    """Create ResNet backbone for feature extraction."""
    resnet_model = models.resnet50(pretrained=True)
    feature_extractor = MultiTaskResNetFeatureExtractor(resnet_model)
    
    if save_components:
        torch.save(feature_extractor.state_dict(), edited_components_dir / "resnet_feature_extractor.pth")
    
    return feature_extractor.to(device)


# feature_extractor = MultiTaskResNetFeatureExtractor(resnet_model)
# torch.save(feature_extractor, os.path.join(filepath, "..", "edited_components", "resnet_feature_extractor.pth"))


#################### Combined Model ####################


class CombinedModel(nn.Module):
    def __init__(self, backbone, yolo_face, yolo_person, ada_face, vit_pose):
        super(CombinedModel, self).__init__()

        if any(m is None for m in [backbone, yolo_face, yolo_person, ada_face, vit_pose]):
            raise ValueError("All models must be provided")

        self.current_task = 'person_detection'
        self.backbone = backbone
        self.yolo_face = yolo_face
        self.yolo_person = yolo_person
        self.ada_face = ada_face
        self.vit_pose = vit_pose

    def set_task(self, task_name):
        supported_tasks = ['face_detection', 'person_detection', 'pose_estimation', 'face_recognition']
        if task_name not in supported_tasks:
            raise ValueError(f"Task {task_name} not supported. Available tasks: {', '.join(supported_tasks)}")
        self.current_task = task_name
        
    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)

        # Route features to the correct task branch
        if self.current_task == 'pose_estimation':
            return self.vit_pose(features)
        elif self.current_task == 'person_detection':
            return self.yolo_person(features)
        elif self.current_task == 'face_detection':
            return self.yolo_face(features)
        else:  # face_recognition
            return self.ada_face(features)


def create_combined_model(
    save_components: bool = True,
    device: Optional[str] = None
) -> CombinedModel:
    """
    Create and return the combined model with all its components.
    
    Args:
        save_components: If True, save individual components to edited_components directory
        device: Device to load models to ('cuda', 'cpu', etc.). If None, use available device.
    
    Returns:
        CombinedModel instance with all components initialized
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    component_models_dir, edited_components_dir = get_model_paths()
    
    # Create all components
    backbone = create_backbone(edited_components_dir, save_components, device)
    person_detect, face_detect = create_yolo_branches(component_models_dir, edited_components_dir, save_components, device)
    face_rec = create_adaface_branch(component_models_dir, edited_components_dir, save_components, device)
    pose_detect = create_vitpose_branch(edited_components_dir, save_components, device)
    
    # Create combined model
    model = CombinedModel(
        backbone=backbone,
        yolo_face=face_detect,
        yolo_person=person_detect,
        ada_face=face_rec,
        vit_pose=pose_detect
    ).to(device)
    
    if save_components:
        torch.save(model.state_dict(), edited_components_dir / "combined_model.pth")
    
    return model


if __name__ == "__main__":
    # When run as a script, create and save the model
    model = create_combined_model(save_components=True)
    print("Model created and saved successfully.")