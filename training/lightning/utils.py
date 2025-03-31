"""
Utility functions for YOLO training, adapted from YOLOv11.
"""
import torch  # type: ignore
import math
import torchvision  # type: ignore

def compute_iou(box1, box2, eps=1e-7, CIoU=False):
    """
    Returns Intersection over Union (IoU) of box1(N,4) to box2(M,4)
    
    Args:
        box1: First box coordinates [N, 4] in [x1, y1, x2, y2] format
        box2: Second box coordinates [M, 4] in [x1, y1, x2, y2] format
        eps: Small value to prevent division by zero
        CIoU: Whether to compute CIoU instead of IoU
        
    Returns:
        IoU scores [N, M]
    """
    # Expand dimensions to compute IoU between all pairs of boxes
    # box1: [N, 4] -> [N, 1, 4]
    # box2: [M, 4] -> [1, M, 4]
    box1 = box1.unsqueeze(1)  # [N, 1, 4]
    box2 = box2.unsqueeze(0)  # [1, M, 4]
    
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.split(1, dim=-1)  # [N, 1, 1] each
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.split(1, dim=-1)  # [1, M, 1] each
    
    # Calculate width and height
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1
    
    # Calculate area
    area1 = (w1.clamp(min=0) * h1.clamp(min=0))  # [N, 1, 1]
    area2 = (w2.clamp(min=0) * h2.clamp(min=0))  # [1, M, 1]

    # Intersection area
    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h  # [N, M, 1]

    # Union Area
    union = area1 + area2 - inter_area + eps
    
    # IoU
    iou = inter_area / union
    
    if not CIoU:
        return iou.squeeze(-1)  # Remove last singleton dimension
    
    # Complete IoU
    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)  # convex width
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)  # convex height
    c2 = (cw ** 2 + ch ** 2) + eps  # convex diagonal squared
    
    # Center distance squared
    rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + 
            (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
    
    # Aspect ratio consistency
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
    )
    
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    
    return (iou - (rho2 / c2 + v * alpha)).squeeze(-1)  # CIoU

def make_anchors(x, strides, offset=0.5):
    """
    Make anchors from features.
    
    Args:
        x: List of feature maps
        strides: List of strides for each feature map
        offset: Anchor offset
        
    Returns:
        tuple: (anchor_points, stride_tensor)
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def non_max_suppression(outputs, confidence_threshold=0.001, iou_threshold=0.65):
    """
    Perform non-maximum suppression on YOLO outputs.
    
    Args:
        outputs: YOLO model outputs
        confidence_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of filtered detections
    """
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold  # candidates

    # Settings
    output = [torch.zeros((0, 6), device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4]
        box = torch.cat((
            box[:, :2] - box[:, 2:] / 2,  # x1, y1
            box[:, :2] + box[:, 2:] / 2   # x2, y2
        ), dim=1)

        # Detections matrix nx6 (xyxy, conf, cls)
        if nc > 1:
            i, j = (x[:, 4:] > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 4, None], j[:, None].float()), 1)
        else:  # best class only
            conf = x[:, 4:].max(1, keepdim=True)
            x = torch.cat((box, conf, torch.zeros((conf.shape[0], 1), device=conf.device)), 1)[conf.view(-1) > confidence_threshold]

        # Filter by class
        if x.shape[0]:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
            i = i[:max_det]  # limit detections
            output[index] = x[i]

    return output