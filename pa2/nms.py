"""Non-Maximum Suppression implementado do zero (torchvision.ops.nms é proibido)."""

from __future__ import annotations

import numpy as np

from metrics import iou_matrix


def nms_xywh(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """NMS guloso clássico sobre caixas (x,y,w,h).

    Ordena por score decrescente; mantém a caixa de maior score e suprime
    qualquer outra ainda não suprimida cuja IoU com ela exceda o limiar;
    repete para a próxima caixa restante de maior score. O(N^2) em IoU, que é
    suficiente para o número de detecções por quadro deste PA.

    Retorna os índices (no array original) das caixas mantidas, em ordem de score.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(boxes)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(-scores)
    iou = iou_matrix(boxes, boxes)

    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        overlapping = iou[idx] >= iou_threshold
        overlapping[idx] = False
        suppressed |= overlapping

    return np.array(keep, dtype=np.int64)
