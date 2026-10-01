"""Métricas de rastreamento (MOT) implementadas do zero.

Implementa:
  - IDF1 / IDP / IDR, via o assignment global de identidades (Ristani et al. 2016,
    "Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking"),
    resolvido com atribuição húngara (scipy.optimize.linear_sum_assignment) sobre
    uma matriz de custo quadrada com linhas/colunas-dummy para permitir identidades
    sem correspondência.
  - Contagem de ID switches e fragmentações, seguindo a convenção CLEAR-MOT /
    MOTChallenge: o casamento quadro-a-quadro prefere manter a correspondência
    anterior sempre que ainda for geometricamente válida (acima do limiar de IoU),
    e só refaz o assignment húngaro para o que sobrar. Um switch é contado quando a
    última identidade conhecida de uma identidade verdadeira muda; uma fragmentação
    é contada quando uma identidade verdadeira estava sendo rastreada, perde a
    correspondência por pelo menos um quadro, e depois volta a ser casada.
  - Erro de contagem de identidades únicas (|pred ids| - |gt ids|, e variantes).

Formato de entrada: cada detecção é uma tupla (frame, id, x, y, w, h), com (x, y) o
canto superior-esquerdo e (w, h) largura/altura — o mesmo formato de bb_left,
bb_top, bb_width, bb_height do gt.txt do MOT17. `frame` e `id` são inteiros.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

Detection = tuple  # (frame:int, id:int, x:float, y:float, w:float, h:float)


def iou_xywh(box_a, box_b) -> float:
    """IoU entre duas caixas no formato (x, y, w, h), x,y = canto superior-esquerdo."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU vetorizado entre dois conjuntos de caixas (N,4) e (M,4) em formato xywh."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    ax1 = boxes_a[:, 0][:, None]
    ay1 = boxes_a[:, 1][:, None]
    ax2 = (boxes_a[:, 0] + boxes_a[:, 2])[:, None]
    ay2 = (boxes_a[:, 1] + boxes_a[:, 3])[:, None]

    bx1 = boxes_b[:, 0][None, :]
    by1 = boxes_b[:, 1][None, :]
    bx2 = (boxes_b[:, 0] + boxes_b[:, 2])[None, :]
    by2 = (boxes_b[:, 1] + boxes_b[:, 3])[None, :]

    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h

    area_a = (boxes_a[:, 2] * boxes_a[:, 3])[:, None]
    area_b = (boxes_b[:, 2] * boxes_b[:, 3])[None, :]
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def _frames_index(detections):
    """Agrupa detecções por quadro: {frame: {id: (x,y,w,h)}}."""
    by_frame = defaultdict(dict)
    for frame, obj_id, x, y, w, h in detections:
        by_frame[int(frame)][int(obj_id)] = (float(x), float(y), float(w), float(h))
    return by_frame


def _match_frame_with_continuity(gt_boxes: dict, pred_boxes: dict, last_match: dict, iou_threshold: float):
    """Casa GT<->pred em um quadro, preferindo manter a correspondência anterior.

    gt_boxes / pred_boxes: {id: (x,y,w,h)} presentes neste quadro.
    last_match: {gt_id: last_pred_id} conhecido até aqui (não é limpo em gaps).
    Retorna dict {gt_id: pred_id} dos casamentos deste quadro.
    """
    matches = {}
    used_gt, used_pred = set(), set()

    # 1) preserva correspondências anteriores que continuam válidas geometricamente.
    for gt_id, prev_pred in last_match.items():
        if gt_id in used_gt or gt_id not in gt_boxes:
            continue
        if prev_pred is None or prev_pred not in pred_boxes or prev_pred in used_pred:
            continue
        if iou_xywh(gt_boxes[gt_id], pred_boxes[prev_pred]) >= iou_threshold:
            matches[gt_id] = prev_pred
            used_gt.add(gt_id)
            used_pred.add(prev_pred)

    # 2) para o que sobrou, assignment húngaro maximizando IoU.
    rem_gt = [g for g in gt_boxes if g not in used_gt]
    rem_pred = [p for p in pred_boxes if p not in used_pred]
    if rem_gt and rem_pred:
        boxes_g = np.array([gt_boxes[g] for g in rem_gt])
        boxes_p = np.array([pred_boxes[p] for p in rem_pred])
        iou = iou_matrix(boxes_g, boxes_p)
        cost = 1.0 - iou
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= iou_threshold:
                matches[rem_gt[r]] = rem_pred[c]

    return matches


@dataclass
class MOTAccumulator:
    """Varre a sequência quadro a quadro e acumula tudo que as métricas precisam."""

    iou_threshold: float = 0.5
    ntp: dict = field(default_factory=lambda: defaultdict(int))  # (gt_id,pred_id) -> #quadros casados
    nf_gt: dict = field(default_factory=lambda: defaultdict(int))  # gt_id -> #quadros em que existe
    nf_pred: dict = field(default_factory=lambda: defaultdict(int))  # pred_id -> #quadros em que existe
    num_switches: int = 0
    num_fragmentations: int = 0

    def _reset_counts(self):
        self.ntp = defaultdict(int)
        self.nf_gt = defaultdict(int)
        self.nf_pred = defaultdict(int)
        self.num_switches = 0
        self.num_fragmentations = 0

    def accumulate(self, gt_detections, pred_detections):
        self._reset_counts()
        gt_by_frame = _frames_index(gt_detections)
        pred_by_frame = _frames_index(pred_detections)
        all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))

        last_match: dict[int, int | None] = {}
        was_tracked: dict[int, bool] = defaultdict(bool)

        for frame in all_frames:
            gt_boxes = gt_by_frame.get(frame, {})
            pred_boxes = pred_by_frame.get(frame, {})

            for gt_id in gt_boxes:
                self.nf_gt[gt_id] += 1
            for pred_id in pred_boxes:
                self.nf_pred[pred_id] += 1

            frame_matches = _match_frame_with_continuity(gt_boxes, pred_boxes, last_match, self.iou_threshold)

            for gt_id, pred_id in frame_matches.items():
                self.ntp[(gt_id, pred_id)] += 1

            for gt_id in gt_boxes:
                if gt_id in frame_matches:
                    pred_now = frame_matches[gt_id]
                    prev = last_match.get(gt_id)
                    if prev is not None and prev != pred_now:
                        self.num_switches += 1
                    if (not was_tracked[gt_id]) and prev is not None:
                        self.num_fragmentations += 1
                    was_tracked[gt_id] = True
                    last_match[gt_id] = pred_now
                else:
                    was_tracked[gt_id] = False
                    last_match.setdefault(gt_id, None)

        return self

    # ---- métricas derivadas do acumulado ----

    def idf1(self):
        gt_ids = sorted(self.nf_gt)
        pred_ids = sorted(self.nf_pred)
        n_gt, n_pred = len(gt_ids), len(pred_ids)

        if n_gt == 0 and n_pred == 0:
            return dict(idf1=1.0, idp=1.0, idr=1.0, idtp=0, idfp=0, idfn=0)
        if n_gt == 0:
            idfp = sum(self.nf_pred.values())
            return dict(idf1=0.0, idp=0.0, idr=1.0, idtp=0, idfp=idfp, idfn=0)
        if n_pred == 0:
            idfn = sum(self.nf_gt.values())
            return dict(idf1=0.0, idp=1.0, idr=0.0, idtp=0, idfp=0, idfn=idfn)

        n = max(n_gt, n_pred)
        # custo(g,p) = nf_gt[g] + nf_pred[p] - 2*ntp(g,p)  == IDFN+IDFP se g<->p for o par escolhido
        cost = np.zeros((n, n), dtype=np.float64)
        for i, g in enumerate(gt_ids):
            for j, p in enumerate(pred_ids):
                cost[i, j] = self.nf_gt[g] + self.nf_pred[p] - 2 * self.ntp.get((g, p), 0)
        # linhas/colunas dummy: gt sem par custa todo o seu próprio comprimento (tudo IDFN);
        # pred sem par custa todo o seu próprio comprimento (tudo IDFP); dummy-dummy custa 0.
        for i in range(n_gt, n):  # dummy gt
            for j, p in enumerate(pred_ids):
                cost[i, j] = self.nf_pred[p]
        for j in range(n_pred, n):  # dummy pred
            for i, g in enumerate(gt_ids):
                cost[i, j] = self.nf_gt[g]
        # (dummy, dummy) já é 0 por inicialização

        row_ind, col_ind = linear_sum_assignment(cost)
        total_cost = cost[row_ind, col_ind].sum()  # = IDFN + IDFP

        total_gt_frames = sum(self.nf_gt.values())
        total_pred_frames = sum(self.nf_pred.values())

        # idtp = soma dos ntp dos pares (gt real, pred real) efetivamente casados
        idtp = 0
        for i, j in zip(row_ind, col_ind):
            if i < n_gt and j < n_pred:
                idtp += self.ntp.get((gt_ids[i], pred_ids[j]), 0)

        idfn = total_gt_frames - idtp
        idfp = total_pred_frames - idtp
        assert abs((idfn + idfp) - total_cost) < 1e-6, "inconsistência no custo do assignment"

        idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0
        idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (idtp + idfp + idfn) > 0 else 1.0

        return dict(idf1=idf1, idp=idp, idr=idr, idtp=idtp, idfp=idfp, idfn=idfn)

    def id_switches(self) -> int:
        return self.num_switches

    def fragmentations(self) -> int:
        return self.num_fragmentations

    def unique_id_count_error(self):
        n_gt, n_pred = len(self.nf_gt), len(self.nf_pred)
        return dict(n_gt_ids=n_gt, n_pred_ids=n_pred, diff=n_pred - n_gt, ratio=(n_pred / n_gt if n_gt else float("nan")))


def compute_mot_metrics(gt_detections, pred_detections, iou_threshold: float = 0.5) -> dict:
    """Interface principal: recebe listas de (frame,id,x,y,w,h) e devolve todas as métricas."""
    acc = MOTAccumulator(iou_threshold=iou_threshold).accumulate(gt_detections, pred_detections)
    out = acc.idf1()
    out["id_switches"] = acc.id_switches()
    out["fragmentations"] = acc.fragmentations()
    out.update(acc.unique_id_count_error())
    return out
