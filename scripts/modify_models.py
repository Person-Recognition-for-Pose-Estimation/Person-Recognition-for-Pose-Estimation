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

import net_adaface

ada_face_model_path = 'component_models/adaface_ir50_ms1mv2.ckpt'
ada_face_model = net_adaface.build_model('ir_50')
statedict = torch.load(ada_face_model_path)['state_dict']
model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
ada_face_model.load_state_dict(model_statedict)

class CustomAdaFace(nn.Module):
    def __init__(self, ada_face_model: net_adaface.Backbone, backbone_channels=512):
        super(CustomAdaFace, self).__init__()
        
        # Adapter to match YOLO's expected features (32 channels)
        self.adapter = nn.Conv2d(2048, 64, kernel_size=1)
        
        # Save the body and output
        self.body = ada_face_model.body
        self.output_layer = ada_face_model.output_layer

    def forward(self, backbone_features):
        
        x = self.adapter(backbone_features)

        for idx, module in enumerate(self.body):
            x = module(x)

        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output, norm

face_rec_branch = CustomAdaFace(ada_face_model)
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

from transformers import AutoImageProcessor, ResNetForImageClassification

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

class MultiTaskResNetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(MultiTaskResNetFeatureExtractor, self).__init__()

        # Define the layers of the original model
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        # self.flatten = nn.Flatten()
        
        # # Define separate heads for different outputs, which have different sizes
        # self.yolo_adapter = nn.Conv2d(2048, 512, kernel_size=1)
        # self.ada_face_adapter = nn.Conv2d(2048, 64, kernel_size=1)
        # self.vit_pose_adapter = 0 # TODO Change the output size to match ViT Pose

    def forward(self, x, current_task):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x

    #     # Get outputs for each head
    #     return torch.where(current_task == 'person_detection' or current_task == 'face_detection',
    #         self.yolo_forward(x),
    #         torch.where(current_task == 'face_identification',
    #             self.ada_face_forward(x),
    #             self.vit_pose_forward(x)))
    
    # def yolo_forward(self, x):
    #     return self.yolo_head(x)
    
    # def ada_face_forward(self, x):
    #     return self.ada_face_head(x)
    
    # def vit_pose_forward(self, x):
    #     return self.vit_pose_head(x)

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