"""
Wrapper for YOLO model to work with our multi-task model.
"""
import torch  # type: ignore
from ultralytics.utils import LOGGER
import torch.nn.functional as F
import torchvision.ops
from .utils import compute_iou

class UltralyticsTrainerWrapper:
    def __init__(self, combined_model, task_name):
        """
        Initialize wrapper for YOLO model
        
        Args:
            combined_model: Our multi-task model with backbone and all branches
            task_name: Name of the task ('face_detection' or 'person_detection')
        """
        self.model = combined_model
        self.task_name = task_name
        self.model.set_task(task_name)
        
        # Get the appropriate YOLO branch based on task
        self.yolo_branch = (self.model.yolo_face if task_name == 'face_detection' 
                           else self.model.yolo_person)
        
    def setup_trainer(self, cfg, _callbacks=None):
        """Store configuration"""
        self.cfg = cfg
        
    def process_batch(self, batch):
        """
        Process batch data into standardized format
        
        Args:
            batch: Input batch in YOLO format
            
        Returns:
            tuple: (images, targets) or (None, None) if invalid batch
        """
        if isinstance(batch, dict):
            images = batch['img']
            if 'cls' in batch and 'bboxes' in batch:
                cls = batch['cls'].view(-1, 1) if batch['cls'].dim() == 1 else batch['cls']
                batch_idx = batch['batch_idx'].view(-1, 1) if batch['batch_idx'].dim() == 1 else batch['batch_idx']
                unique_batch_idx = batch_idx.squeeze(-1)
                gt_groups = [(unique_batch_idx == i).sum().item() for i in range(images.shape[0])]
                targets = {
                    'cls': cls.squeeze(-1),
                    'bboxes': batch['bboxes'],
                    'batch_idx': batch_idx.squeeze(-1),
                    'gt_groups': gt_groups
                }
            else:
                LOGGER.warning("Missing cls or bboxes in batch")
                return None, None
        else:
            if not isinstance(batch, (list, tuple)):
                batch = [batch]
            images = batch[0]
            if len(batch) > 1:
                targets = batch[1]
                if isinstance(targets, torch.Tensor) and targets.shape[-1] == 6:
                    batch_idx = targets[:, 0]
                    gt_groups = [(batch_idx == i).sum().item() for i in range(images.shape[0])]
                    targets = {
                        'cls': targets[:, 1],
                        'bboxes': targets[:, 2:],
                        'batch_idx': batch_idx,
                        'gt_groups': gt_groups
                    }
            else:
                LOGGER.warning("No targets in batch")
                return None, None
            
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected images to be a tensor, got {type(images)}")
            
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
            
        return images, targets

    def compute_loss(self, preds, targets):
        """
        Compute detection losses
        
        Args:
            preds: Model predictions
            targets: Target dictionary
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        device = preds.device if isinstance(preds, torch.Tensor) else preds[0].device
        
        # Handle different prediction formats
        if isinstance(preds, (list, tuple)):
            # Multiple prediction heads format
            pred_boxes = preds[0][..., :4]  # [batch, anchors, 4]
            pred_conf = preds[0][..., 4]    # [batch, anchors]
            pred_cls = preds[0][..., 5:]    # [batch, anchors, num_classes]
        else:
            # Single tensor format
            pred_boxes = preds[..., :4]     # [batch, anchors, 4]
            pred_conf = preds[..., 4]       # [batch, anchors]
            pred_cls = preds[..., 5:]       # [batch, anchors, num_classes]
        
        # Get target boxes and classes
        target_boxes = targets['bboxes'].to(device)
        target_cls = targets['cls'].to(device).long()
        
        # Box loss - CIoU loss
        iou = compute_iou(pred_boxes.reshape(-1, 4), target_boxes.reshape(-1, 4))
        box_loss = (1.0 - iou).mean()
        
        # Classification loss (only for positive samples)
        pos_mask = (iou > 0.5).float().reshape(-1)
        if pos_mask.sum() > 0:
            cls_loss = F.cross_entropy(
                pred_cls.reshape(-1, pred_cls.size(-1))[pos_mask > 0], 
                target_cls.reshape(-1)[pos_mask > 0],
                reduction='mean'
            )
        else:
            cls_loss = torch.tensor(0.0, device=device)
        
        # Confidence loss
        conf_loss = F.binary_cross_entropy_with_logits(
            pred_conf.reshape(-1),
            pos_mask,
            reduction='mean'
        )
        
        # Total loss with weighting
        box_weight = 7.5  # Standard YOLO box loss weight
        cls_weight = 0.5  # Standard YOLO class loss weight
        conf_weight = 1.0 # Standard YOLO confidence loss weight
        
        total_loss = (box_weight * box_loss + 
                     cls_weight * cls_loss + 
                     conf_weight * conf_loss)
        
        loss_dict = {
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'conf_loss': conf_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict

    def train_step(self, batch):
        """Single training step"""
        images, targets = self.process_batch(batch)
        if targets is None:
            return None
            
        # Get backbone features and predictions
        backbone_features = self.model.backbone(images)
        preds = self.yolo_branch(backbone_features)
        
        # Handle different prediction formats
        if isinstance(preds, (list, tuple)):
            pred_tensor = preds[0]
        else:
            pred_tensor = preds
            
        # Compute loss
        total_loss, _ = self.compute_loss(pred_tensor, targets)
        return total_loss

    def validation_step(self, batch):
        """Single validation step"""
        images, targets = self.process_batch(batch)
        if targets is None:
            return None, None, None
            
        with torch.no_grad():
            # Get backbone features and predictions
            backbone_features = self.model.backbone(images)
            preds = self.yolo_branch(backbone_features)
            
            # Handle different prediction formats
            if isinstance(preds, (list, tuple)):
                pred_tensor = preds[0]
            else:
                pred_tensor = preds
                
            # Compute loss
            _, loss_dict = self.compute_loss(pred_tensor, targets)
            
            # Prepare predictions for NMS
            # Extract boxes, confidence scores, and class predictions
            boxes = pred_tensor[..., :4]  # [batch_size, num_anchors, 4]
            obj_conf = pred_tensor[..., 4:5]  # [batch_size, num_anchors, 1]
            cls_conf = pred_tensor[..., 5:]  # [batch_size, num_anchors, num_classes]
            
            # Combine confidence scores
            conf = obj_conf * cls_conf  # [batch_size, num_anchors, num_classes]
            
            # Get max confidence and corresponding class for each prediction
            scores, classes = conf.max(dim=-1)  # [batch_size, num_anchors]
            
            # Convert boxes from center format to corner format if needed
            # Assuming boxes are in [x1, y1, x2, y2] format already
            # If they're in [cx, cy, w, h] format, uncomment these lines:
            # x1y1 = boxes[..., :2] - boxes[..., 2:] / 2
            # x2y2 = boxes[..., :2] + boxes[..., 2:] / 2
            # boxes = torch.cat((x1y1, x2y2), dim=-1)
            
            # Prepare predictions for NMS
            nms_preds = []
            for i in range(boxes.shape[0]):  # Loop through batch
                # Combine predictions for this image
                img_preds = torch.cat([
                    boxes[i],  # [num_anchors, 4]
                    scores[i].unsqueeze(-1),  # [num_anchors, 1]
                    classes[i].float().unsqueeze(-1)  # [num_anchors, 1]
                ], dim=-1)  # [num_anchors, 6]
                
                # Filter out low confidence predictions before NMS
                conf_mask = scores[i] > 0.001
                img_preds = img_preds[conf_mask]
                
                # Apply NMS
                if len(img_preds):
                    nms_idx = torchvision.ops.nms(
                        img_preds[:, :4],
                        img_preds[:, 4],
                        iou_threshold=0.65
                    )
                    nms_preds.append(img_preds[nms_idx])
                else:
                    nms_preds.append(torch.zeros((0, 6), device=boxes.device))
            
        return nms_preds, targets, loss_dict