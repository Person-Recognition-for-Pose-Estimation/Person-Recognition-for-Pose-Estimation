import pathlib
import torch
import torchvision.models as models
import torch.nn as nn
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


from ultralytics import YOLO


class CustomYOLO(nn.Module):
    def __init__(self, yolo_model, backbone_channels=2048):
        super(CustomYOLO, self).__init__()
        
        # Define target size that YOLO expects (typically 640x640 or similar)
        self.target_size = 640  # or whatever size your YOLO model expects
        
        self.adapter = nn.Sequential(
            nn.Conv2d(backbone_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            
            # Adaptive pooling to get consistent spatial dimensions
            nn.AdaptiveAvgPool2d((self.target_size // 32, self.target_size // 32)),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            # Upsample to match YOLO's expected input size
            nn.Upsample(size=(self.target_size, self.target_size), mode='bilinear', align_corners=True),
            
            # Final adaptation to 3 channels
            nn.Conv2d(256, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU()
        )
        
        self.yolo = yolo_model.model
    
    def forward(self, backbone_features):
        # print(f"Input backbone features shape: {backbone_features.shape}")
        x = self.adapter(backbone_features)
        # print(f"After adapter shape: {x.shape}")
        
        # Ensure the input is properly scaled
        x = (x - x.min()) / (x.max() - x.min())  # Normalize to [0,1]
        
        detections = self.yolo(x)
        return detections

        
        # TODO: If inference, process detections
        # processed_detections = []
        # for detection in detections:
        #     processed = self._process_detection(detection)
        #     processed_detections.append(processed)
        # return processed_detections


def create_yolo_branches(
    component_models_dir: pathlib.Path,
    edited_components_dir: pathlib.Path,
    save_components: bool = False,
    device: str = 'cpu'
) -> Tuple[CustomYOLO, CustomYOLO]:
    """Create YOLO branches for person and face detection."""
    # Person detection
    person_model_path = component_models_dir / "yolo11n.pt"
    yolo_model = YOLO(str(person_model_path))
    person_detect_branch = CustomYOLO(yolo_model)
    
    # Face detection
    face_model_path = component_models_dir / "yolov11n-face.pt"
    yolo_face_model = YOLO(str(face_model_path))
    face_detect_branch = CustomYOLO(yolo_face_model)
    
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
    def __init__(self, pretrained_path, config):
        super(CustomAdaFace, self).__init__()
        
        # Load the complete pre-trained AdaFace model
        self.adaface_model = build_model(model_name=config.arch)
        
        checkpoint = torch.load(pretrained_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        
        # Load weights except for input layer
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('input_layer')}
        self.adaface_model.load_state_dict(filtered_state_dict, strict=False)
        
        # Replace input layer with an adapter that takes backbone features
        self.adaface_model.input_layer = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1),  # First reduce channels
            nn.BatchNorm2d(64),
            nn.PReLU(64),  # AdaFace uses PReLU
            # Additional layers to match the processing of original input_layer
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
        # Forward through modified AdaFace model
        embeddings, norms = self.adaface_model(x)
        
        if labels is not None:
            return self.head(embeddings, norms, labels)
        return embeddings, norms


class Config:
    def __init__(self):
        self.arch = 'ir_50'
        self.head = 'adaface'
        self.num_classes = 1000  # Update this based on your dataset
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

from transformers import AutoProcessor, VitPoseForPoseEstimation

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomVitPose(nn.Module):
    def __init__(self, vit_pose_model: VitPoseForPoseEstimation, backbone_channels=2048):
        super(CustomVitPose, self).__init__()
        
        hidden_size = vit_pose_model.backbone.config.hidden_size
        patch_size = vit_pose_model.backbone.config.patch_size
        
        self.adapter = nn.Sequential(
            # Initial channel reduction
            nn.Conv2d(backbone_channels, hidden_size * 2, kernel_size=1),
            nn.LayerNorm([hidden_size * 2]),
            nn.GELU(),
            
            # Spatial processing
            nn.Conv2d(hidden_size * 2, hidden_size * 2, 
                     kernel_size=3, padding=1, groups=hidden_size * 2),  # Depth-wise
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1),  # Point-wise
            nn.LayerNorm([hidden_size]),
            nn.GELU(),
            
            nn.Conv2d(hidden_size, hidden_size, kernel_size=patch_size, 
                     stride=patch_size, padding=0),  # Match ViT patch size
        )
        
        self.vit_pose = vit_pose_model
        self.vit_pose.backbone.embeddings.patch_embeddings = nn.Identity()
        
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
    hf_vit_pose_model = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-base-simple",
        device_map=device
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
        supported_tasks = ['face_detection', 'person_detection', 'pose_estimation', 'face_identification']
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
        else:  # face_identification
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