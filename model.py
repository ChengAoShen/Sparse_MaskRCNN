import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .SparseRoIHead import SparseRoIHeads


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def get_sparse_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    roi_heads = SparseRoIHeads(
        model.roi_heads.box_roi_pool,
        model.roi_heads.box_head,
        FastRCNNPredictor(in_features, num_classes),
        model.roi_heads.fg_iou_thresh,
        model.roi_heads.bg_iou_thresh,
        model.roi_heads.batch_size_per_image,
        model.roi_heads.positive_fraction,
        model.roi_heads.bbox_reg_weights,
        model.roi_heads.score_thresh,
        model.roi_heads.nms_thresh,
        model.roi_heads.detections_per_img,
        model.roi_heads.mask_roi_pool,
        model.roi_heads.mask_head,
        MaskRCNNPredictor(in_features_mask, 256, num_classes),
    )

    model.roi_heads = roi_heads
    return model
