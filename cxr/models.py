"""Detection models: torchvision Faster R-CNN pretrained on COCO, head replaced for one
class (lung opacity) + background. Images are fed at their native 512 px (no internal
resize to 800 px)."""

from torchvision.models.detection import faster_rcnn

SIZE = 512

ARCHITECTURES = {
    "fasterrcnn_resnet50_fpn": (faster_rcnn.fasterrcnn_resnet50_fpn,
                                faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
    "fasterrcnn_resnet50_fpn_v2": (faster_rcnn.fasterrcnn_resnet50_fpn_v2,
                                   faster_rcnn.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    "fasterrcnn_mobilenet_v3_large_fpn": (faster_rcnn.fasterrcnn_mobilenet_v3_large_fpn,
                                          faster_rcnn.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1),
}


def build_model(architecture: str, size: int = SIZE, pretrained: bool = True):
    builder, weights = ARCHITECTURES[architecture]
    model = builder(weights=weights if pretrained else None, min_size=size, max_size=size,
                    box_score_thresh=0.01, box_detections_per_img=20)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 2)
    return model


def weights_url(architecture: str) -> str:
    return ARCHITECTURES[architecture][1].url
