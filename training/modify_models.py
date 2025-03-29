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


from ultralytics import YOLO


class Conv(nn.Module):
    """Basic convolution block with batch normalization and activation."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DFL(nn.Module):
    """Distribution Focal Loss head for better box regression."""
    def __init__(self, ch: int = 16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)

class CustomYOLO(nn.Module):
    def __init__(self, num_classes: int = 1, backbone_channels: int = 2048):
        super().__init__()
        
        # Feature adaptation
        self.adapter = nn.Sequential(
            Conv(backbone_channels, 512, k=1),
            Conv(512, 256, k=3, p=1),
            Conv(256, 256, k=1)
        )
        
        # Detection head parameters
        self.nc = num_classes  # number of classes
        self.ch = 16  # DFL channels
        self.no = num_classes + self.ch * 4  # number of outputs per anchor
        
        # Box regression branch with DFL
        self.box = nn.Sequential(
            Conv(256, 256, k=3, p=1),
            Conv(256, 256, k=3, p=1),
            nn.Conv2d(256, 4 * self.ch, 1)  # 4 box coordinates * DFL channels
        )
        
        # Classification branch
        self.cls = nn.Sequential(
            Conv(256, 256, k=3, p=1),
            Conv(256, 256, k=3, p=1),
            nn.Conv2d(256, self.nc, 1)
        )
        
        self.dfl = DFL(self.ch)
        
        # Initialize biases
        self.initialize_biases()
    
    def initialize_biases(self):
        """Initialize biases for better training stability."""
        for conv in self.box:
            if isinstance(conv, nn.Conv2d):
                conv.bias.data.fill_(1.0)
        for conv in self.cls:
            if isinstance(conv, nn.Conv2d):
                b = conv.bias.view(1, -1)
                b.data.fill_(-4.595)  # -log((1 - 0.01) / 0.01)
    
    def forward(self, x):
        """
        Forward pass returning box predictions and class logits.
        Args:
            x: Backbone features [B, C, H, W]
        Returns:
            Tuple of:
            - pred_boxes: Box predictions [B, 4, HW]
            - pred_scores: Class predictions [B, nc, HW]
        """
        x = self.adapter(x)
        
        # Get predictions
        box_pred = self.box(x)  # [B, 4*ch, H, W]
        cls_pred = self.cls(x)  # [B, nc, H, W]
        
        # Reshape and process boxes through DFL
        B, _, H, W = x.shape
        box_pred = box_pred.view(B, 4, self.ch, H * W)
        box_pred = self.dfl(box_pred.view(B, 4 * self.ch, H * W))  # [B, 4, HW]
        
        # Reshape class predictions
        cls_pred = cls_pred.view(B, self.nc, H * W)  # [B, nc, HW]
        
        return box_pred, cls_pred

        
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
    # Person detection (1 class - person)
    person_detect_branch = CustomYOLO(num_classes=1, backbone_channels=2048)
    
    # Face detection (1 class - face)
    face_detect_branch = CustomYOLO(num_classes=1, backbone_channels=2048)
    
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
        
        # Define adapter network
        self.adapter = nn.Sequential(
            # Initial channel reduction and processing
            nn.Conv2d(backbone_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.PReLU(256),  # AdaFace uses PReLU
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            
            # Adaptive pooling to get consistent spatial dimensions
            nn.AdaptiveAvgPool2d((112, 112)),  # Common input size for face recognition
            
            # Final adaptation to match AdaFace input
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        
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
        self.num_classes = 70722
        self.embedding_size: int = 512,
        self.backbone_channels: int = 2048,
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