"""Segunda fonte de detecção da Parte 1: detector pré-treinado do
torchvision, em modo de inferência, filtrado para a classe `person` do COCO
(id=1). O modelo já faz seu próprio NMS internamente como parte padrão do
forward pass de inferência (isso é inerente a usar um detector pré-treinado
do torchvision "em modo de inferência", permitido pelo enunciado) — mas
depois de filtrar por classe e score, rodamos TAMBÉM a nossa própria NMS
(pa2.nms, sem torchvision.ops.nms) como post-processing explícito sob nosso
controle, já que é esse post-processing que de fato escrevemos.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from pa2.nms import nms_xywh

COCO_PERSON_CLASS_ID = 1


def load_person_detector():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


@torch.no_grad()
def detect_persons(model, preprocess, image_rgb_uint8: np.ndarray, score_threshold: float = 0.5, nms_iou_threshold: float = 0.5):
    """image_rgb_uint8: array (H,W,3) uint8. Retorna array (N,5) [x,y,w,h,score]."""
    tensor = torch.from_numpy(image_rgb_uint8).permute(2, 0, 1)
    batch = [preprocess(tensor)]
    output = model(batch)[0]

    boxes = output["boxes"].numpy()  # x1,y1,x2,y2
    scores = output["scores"].numpy()
    labels = output["labels"].numpy()

    keep = (labels == COCO_PERSON_CLASS_ID) & (scores >= score_threshold)
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return np.zeros((0, 5))

    xywh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)

    keep_idx = nms_xywh(xywh, scores, iou_threshold=nms_iou_threshold)
    xywh, scores = xywh[keep_idx], scores[keep_idx]

    return np.concatenate([xywh, scores[:, None]], axis=1)
