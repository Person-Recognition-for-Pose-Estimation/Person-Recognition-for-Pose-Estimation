"""
Utility functions for YOLO training, adapted from YOLOv11.
"""
import torch  # type: ignore
import math
import torchvision  # type: ignore

def compute_iou(box1, box2, eps=1e-7):
    """
    Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
    
    Args:
        box1: First box coordinates [x1, y1, x2, y2]
        box2: Second box coordinates [x1, y1, x2, y2]
        eps: Small value to prevent division by zero
        
    Returns:
        IoU scores
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU

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