"""Tracker baseline por quadro (Parte 0 item 4 / Parte 1 item 2).

Regra de associação: IoU entre as detecções do quadro t e a última caixa
observada de cada track viva no quadro t-1. Matching guloso ou húngaro,
limiar fixo de IoU. ID novo sempre que uma detecção não casa com nenhuma
track viva. Track morta depois de `max_age` quadros consecutivos sem
observação. Não há nenhum modelo de movimento: a "previsão" da posição no
quadro seguinte é simplesmente a última caixa observada (hipótese de
velocidade zero) — é exatamente o que a Parte 2 vai substituir por uma rede
recorrente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from metrics import iou_matrix
from pa2.association import greedy_match, hungarian_match


@dataclass
class Track:
    track_id: int
    box: tuple  # (x,y,w,h), última caixa observada
    time_since_update: int = 0
    hits: int = 1
    age: int = 0  # quadros desde a criação


class NaiveIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, min_hits: int = 1, match_method: str = "hungarian"):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        assert match_method in ("hungarian", "greedy")
        self.match_fn = hungarian_match if match_method == "hungarian" else greedy_match

        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self):
        self.tracks = {}
        self._next_id = 1

    def step(self, detections: np.ndarray) -> list[tuple[int, float, float, float, float]]:
        """Processa um quadro. `detections`: array (N,4) de caixas (x,y,w,h).

        Retorna lista de (track_id, x, y, w, h) para tracks confirmadas e
        observadas neste quadro (a saída "identity-aware" do quadro).
        """
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 4)

        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[t].box for t in track_ids]) if track_ids else np.zeros((0, 4))

        iou = iou_matrix(track_boxes, detections)
        matches, unmatched_tracks, unmatched_dets = self.match_fn(iou, self.iou_threshold)

        outputs = []

        for row, col in matches:
            tid = track_ids[row]
            tr = self.tracks[tid]
            tr.box = tuple(detections[col])
            tr.time_since_update = 0
            tr.hits += 1
            tr.age += 1

        for row in unmatched_tracks:
            tid = track_ids[row]
            tr = self.tracks[tid]
            tr.time_since_update += 1
            tr.age += 1

        dead_ids = [tid for tid, tr in self.tracks.items() if tr.time_since_update > self.max_age]
        for tid in dead_ids:
            del self.tracks[tid]

        for col in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(track_id=tid, box=tuple(detections[col]))

        for tid, tr in self.tracks.items():
            if tr.time_since_update == 0 and tr.hits >= self.min_hits:
                outputs.append((tid, *tr.box))

        return outputs

    def run(self, detections_by_frame: dict[int, np.ndarray]) -> list[tuple]:
        """Roda a sequência inteira. Retorna lista de (frame, id, x, y, w, h)."""
        self.reset()
        results = []
        for frame in sorted(detections_by_frame):
            out = self.step(detections_by_frame[frame])
            for tid, x, y, w, h in out:
                results.append((frame, tid, x, y, w, h))
        return results
