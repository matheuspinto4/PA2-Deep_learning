"""Com o MotionGRU recém-inicializado (cabeça de saída zerada -> delta=0
sempre -> previsão = última entrada, exatamente a hipótese de velocidade
zero do tracker ingênuo), o MotionRNNTracker tem que se comportar
IDENTICAMENTE ao NaiveIoUTracker da Parte 0/1 nas mesmas detecções. Essa é
a forma mais direta de confirmar que a troca "última caixa observada" ->
"caixa prevista pela rede" foi implementada certa: antes de a rede aprender
qualquer coisa, as duas regras de associação têm que coincidir."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa2.baseline_tracker import NaiveIoUTracker  # noqa: E402
from pa2.models.motion_rnn import MotionGRU  # noqa: E402
from pa2.motion_tracker import MotionRNNTracker  # noqa: E402

BOX_A = np.array([10.0, 10.0, 20.0, 20.0])
BOX_A2 = np.array([12.0, 11.0, 20.0, 20.0])  # leve deslocamento
BOX_A3 = np.array([15.0, 13.0, 20.0, 20.0])  # mais deslocamento, ainda sobrepõe o congelado


def build_detections():
    # 7 quadros; sumiço nos quadros 3-6 (coasting); reaparece no 7 na MESMA
    # posição do último congelado (IoU=1 garantido, não depende de limiar).
    return {
        1: (np.array([BOX_A]), np.array([0.9])),
        2: (np.array([BOX_A2]), np.array([0.8])),
        3: (np.zeros((0, 4)), np.zeros(0)),
        4: (np.zeros((0, 4)), np.zeros(0)),
        5: (np.zeros((0, 4)), np.zeros(0)),
        6: (np.zeros((0, 4)), np.zeros(0)),
        7: (np.array([BOX_A2]), np.array([0.95])),  # mesma posição do último congelado (quadro 2)
    }


def test_motion_tracker_com_pesos_zero_reproduz_o_tracker_ingenuo():
    dets = build_detections()
    det_by_frame = {f: boxes for f, (boxes, _) in dets.items()}
    score_by_frame = {f: scores for f, (_, scores) in dets.items()}

    naive = NaiveIoUTracker(iou_threshold=0.3, max_age=5, match_method="hungarian")
    naive_result = naive.run(det_by_frame)

    torch.manual_seed(0)
    model = MotionGRU(hidden_size=8)
    motion = MotionRNNTracker(model, iou_threshold=0.3, max_age=5, match_method="hungarian")
    motion_result = motion.run(det_by_frame, score_by_frame)

    assert naive_result == motion_result, (naive_result, motion_result)
    # confirma que de fato sobreviveu ao buraco de 4 quadros (3-6) e reconectou no 7
    frames_with_output = sorted(set(f for f, *_ in naive_result))
    assert frames_with_output == [1, 2, 7]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
