import pathlib
import torch
import torchvision.models as models
import torch.nn as nn
import os


filepath = pathlib.Path(__file__).parent.resolve()
edited_components_dir = os.path.join(filepath, "..", "edited_components")

if not os.path.exists(edited_components_dir):
    os.makedirs(edited_components_dir)


#################### YOLO ####################


from ultralytics import YOLO


model_path = os.path.join(filepath, "..", "component_models", "yolov8n.pt")
yolo_model = YOLO(model_path)


class CustomYOLO(nn.Module):
    def __init__(self, yolo_model, backbone_channels=512):
        super(CustomYOLO, self).__init__()
        
        # Adapter to match YOLO's expected features (32 channels)
        self.adapter = nn.Conv2d(backbone_channels, 32, kernel_size=1)
        
        # Remove the first two layers of YOLO, which are for processing input
        self.yolo = nn.Sequential(*list(yolo_model.model.model)[2:])
        
    def forward(self, backbone_features):
        x = self.adapter(backbone_features)
        x = self.yolo(x)
        return x


# Person
model_path = os.path.join(filepath, "..", "component_models", "yolov8n.pt")
yolo_model = YOLO(model_path)
person_detect_branch = CustomYOLO(yolo_model)
torch.save(person_detect_branch, os.path.join(filepath, "..", "edited_components", "custom_yolo.pth"))


# Face
model_path = os.path.join(filepath, "..", "component_models", "yolov8n-face.pt")
yolo_face_model = YOLO(model_path)
face_detect_branch = CustomYOLO(yolo_face_model)
torch.save(face_detect_branch, os.path.join(filepath, "..", "edited_components", "custom_yolo_face.pth"))


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

ada_face_model_path = os.path.join(filepath, "..", "component_models", "adaface_ir50_ms1mv2.ckpt")


face_rec_branch = CustomAdaFace(ada_face_model_path, config)
torch.save(face_rec_branch, os.path.join(filepath, "..", "edited_components", "custom_ada_face.pth"))


#################### ViTPose ####################

from transformers import AutoProcessor, VitPoseForPoseEstimation

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomVitPose(nn.Module):
    def __init__(self, vit_pose_model: VitPoseForPoseEstimation, backbone_channels=2048):
        super(CustomVitPose, self).__init__()
        
        # Get the hidden size from the model's config
        hidden_size = vit_pose_model.backbone.config.hidden_size
        
        self.adapter = nn.Sequential(
            nn.Conv2d(backbone_channels, hidden_size, kernel_size=1),
            nn.LayerNorm([hidden_size]),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LayerNorm([hidden_size]),
        )
        
        self.vit_pose = vit_pose_model
        self.vit_pose.backbone.embeddings.patch_embeddings = nn.Identity()

    def forward(self, backbone_features):
        x = self.adapter(backbone_features)
        return self.vit_pose(x)


hf_vit_pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)
pose_detect_branch = CustomVitPose(hf_vit_pose_model)
torch.save(pose_detect_branch, os.path.join(filepath, "..", "edited_components", "custom_vit_pose.pth"))


#################### Backbone ####################

resnet_model = models.resnet50(pretrained=True)


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


feature_extractor = MultiTaskResNetFeatureExtractor(resnet_model)
torch.save(feature_extractor, os.path.join(filepath, "..", "edited_components", "resnet_feature_extractor.pth"))


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
        # Get features from backbone for the task
        features = self.backbone(x, self.current_task)

        # Use the correct branch for the inputted task
        # torch.where is used here for efficiency
        return torch.where(self.current_task == 'pose_estimation',
            self.vit_pose(features),
            torch.where(self.current_task == 'person_detection',
                self.yolo_person(features),
                torch.where(self.current_task == 'face_detection',
                    self.yolo_face(features),
                    self.ada_face(features)
                )
            )
        )


combined_model = CombinedModel(feature_extractor, face_detect_branch, person_detect_branch, face_rec_branch, pose_detect_branch)
torch.save(combined_model, os.path.join(filepath, "..", "edited_components", "combined_model.pth"))