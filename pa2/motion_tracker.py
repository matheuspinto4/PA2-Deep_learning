"""Parte 2, Trilha A — tracker que usa o MotionGRU no lugar da hipótese de
velocidade zero do tracker ingênuo (pa2/baseline_tracker.py).

Única diferença estrutural em relação ao NaiveIoUTracker: a associação usa
IoU entre a CAIXA PREVISTA pela rede (não a última caixa observada) e as
detecções do quadro atual. Nascimento/morte de track seguem a mesma regra
(ID novo pra detecção sem par, morte após `max_age` quadros sem
observação). Quando uma track não casa com nada no quadro, o estado da GRU
roda pra frente em free-running (alimentando a própria previsão anterior,
confiança 0) -- exatamente o mecanismo de pa2/models/motion_rnn.py, usado
de forma idêntica aqui e no treino.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from metrics import iou_matrix
from pa2.association import greedy_match, hungarian_match


@dataclass
class _MotionTrack:
    track_id: int
    h: torch.Tensor  # (1, hidden_size)
    pred_box: torch.Tensor  # (4,) -- previsão feita no passo anterior pra ESTE quadro
    time_since_update: int = 0
    hits: int = 1


class MotionRNNTracker:
    def __init__(self, model, iou_threshold: float = 0.3, max_age: int = 5, min_hits: int = 1, match_method: str = "hungarian"):
        self.model = model
        self.model.eval()
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        assert match_method in ("hungarian", "greedy")
        self.match_fn = hungarian_match if match_method == "hungarian" else greedy_match

        self.tracks: dict[int, _MotionTrack] = {}
        self._next_id = 1

    def reset(self):
        self.tracks = {}
        self._next_id = 1

    @torch.no_grad()
    def step(self, detections: np.ndarray, scores: np.ndarray) -> list[tuple[int, float, float, float, float]]:
        """detections: (N,4) caixas (x,y,w,h) JÁ NORMALIZADAS pelo tamanho
        da imagem (mesma normalização usada no treino). scores: (N,) confiança.
        """
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        track_ids = list(self.tracks.keys())
        pred_boxes = (
            np.stack([self.tracks[t].pred_box.numpy() for t in track_ids]) if track_ids else np.zeros((0, 4))
        )

        iou = iou_matrix(pred_boxes, detections)
        matches, unmatched_tracks, unmatched_dets = self.match_fn(iou, self.iou_threshold)

        for row, col in matches:
            tid = track_ids[row]
            tr = self.tracks[tid]
            box_t = torch.from_numpy(detections[col]).unsqueeze(0)
            conf_t = torch.tensor([scores[col]], dtype=torch.float32)
            h_new, pred_next = self.model.step(tr.h, box_t, conf_t)
            tr.h = h_new
            tr.pred_box = pred_next.squeeze(0)
            tr.time_since_update = 0
            tr.hits += 1

        for row in unmatched_tracks:
            tid = track_ids[row]
            tr = self.tracks[tid]
            box_t = tr.pred_box.unsqueeze(0)  # free-running: entra com a própria previsão
            conf_t = torch.zeros(1, dtype=torch.float32)
            h_new, pred_next = self.model.step(tr.h, box_t, conf_t)
            tr.h = h_new
            tr.pred_box = pred_next.squeeze(0)
            tr.time_since_update += 1

        dead_ids = [tid for tid, tr in self.tracks.items() if tr.time_since_update > self.max_age]
        for tid in dead_ids:
            del self.tracks[tid]

        new_track_ids = []
        for col in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            h0 = self.model.init_hidden(1)
            box_t = torch.from_numpy(detections[col]).unsqueeze(0)
            conf_t = torch.tensor([scores[col]], dtype=torch.float32)
            h_new, pred_next = self.model.step(h0, box_t, conf_t)
            self.tracks[tid] = _MotionTrack(track_id=tid, h=h_new, pred_box=pred_next.squeeze(0))
            new_track_ids.append(tid)

        outputs = []
        for row, col in matches:
            tid = track_ids[row]
            if self.tracks[tid].hits >= self.min_hits:
                outputs.append((tid, *detections[col]))
        for tid, col in zip(new_track_ids, unmatched_dets):
            if self.tracks[tid].hits >= self.min_hits:
                outputs.append((tid, *detections[col]))

        return outputs

    def run(self, detections_by_frame: dict[int, np.ndarray], scores_by_frame: dict[int, np.ndarray]) -> list[tuple]:
        self.reset()
        results = []
        for frame in sorted(detections_by_frame):
            dets = detections_by_frame[frame]
            scores = scores_by_frame.get(frame, np.ones(len(dets), dtype=np.float32))
            out = self.step(dets, scores)
            for tid, x, y, w, h in out:
                results.append((frame, tid, x, y, w, h))
        return results
