"""Average Precision (AP) para detecção de objeto de uma classe só
(pedestrian), usada como "mAP por quadro" no gráfico obrigatório da Parte 1
item 5 — mede qualidade de DETECÇÃO (quadro a quadro), em contraste com
IDF1, que mede qualidade de IDENTIDADE (ao longo do tempo). Implementado do
zero (AP não está na lista de métricas de rastreamento proibidas, mas como
alimenta um gráfico obrigatório, preferimos não depender de lib externa).

Segue a convenção "all-point interpolation" (VOC2012/COCO): a curva
precisão-recall é tornada monotonicamente decrescente da direita pra
esquerda antes de integrar, o que evita que "serrilhados" da curva bruta
subestimem a área.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from metrics import iou_matrix


def average_precision(gt_detections, detections, iou_threshold: float = 0.5) -> dict:
    """gt_detections: lista de (frame, id, x, y, w, h) [id é ignorado aqui].
    detections: lista de (frame, x, y, w, h, score).

    Retorna dict com 'ap', 'precision' e 'recall' (arrays, para quem quiser
    plotar a curva também) e 'n_gt'/'n_det'.
    """
    gt_by_frame = defaultdict(list)
    for frame, _id, x, y, w, h in gt_detections:
        gt_by_frame[int(frame)].append((x, y, w, h))

    n_gt = sum(len(v) for v in gt_by_frame.values())
    matched = {f: np.zeros(len(boxes), dtype=bool) for f, boxes in gt_by_frame.items()}

    dets_sorted = sorted(detections, key=lambda d: d[5], reverse=True)
    n_det = len(dets_sorted)

    tp = np.zeros(n_det)
    fp = np.zeros(n_det)

    for i, (frame, x, y, w, h, score) in enumerate(dets_sorted):
        frame = int(frame)
        gt_boxes = gt_by_frame.get(frame, [])
        if not gt_boxes:
            fp[i] = 1
            continue
        det_box = np.array([[x, y, w, h]])
        gt_arr = np.array(gt_boxes)
        ious = iou_matrix(det_box, gt_arr)[0]
        best_j = int(np.argmax(ious))
        best_iou = ious[best_j]
        if best_iou >= iou_threshold and not matched[frame][best_j]:
            tp[i] = 1
            matched[frame][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)

    ap = _ap_from_pr(precision, recall)
    return dict(ap=ap, precision=precision, recall=recall, n_gt=n_gt, n_det=n_det)


def _ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    if len(precision) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap
