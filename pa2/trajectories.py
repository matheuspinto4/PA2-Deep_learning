"""Extração e janelamento de trajetórias para treinar o MotionGRU (Parte 2).

Reaproveitável entre o gerador sintético (Parte 0) e o MOT17 real: ambos
produzem a mesma forma básica -- uma lista de (frame, id, x, y, w, h) mais
um mapa opcional de visibilidade por (frame, id).

A "confiança" usada como entrada do modelo durante o treino é a
VISIBILIDADE do ground truth, não um score de detector de verdade (o
treino é feito em trajetórias do gt, como o enunciado pede -- não há
detecção real envolvida aqui). Isso é uma escolha deliberada: visibilidade
é um sinal de qualidade real e variável (0 a 1, cai sob oclusão), joga o
mesmo papel que um score de detector jogaria, e é exatamente o campo que
usamos para decidir se um quadro conta como "observado" ou "oculto" durante
o rollout -- um quadro com visibilidade abaixo de `vis_threshold` é tratado
como se um detector real não tivesse gerado nenhuma caixa ali, forçando o
modelo a rodar em free-running (ver pa2/models/motion_rnn.py) exatamente
como aconteceria na inferência de verdade.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def extract_contiguous_trajectories(detections, visibilities=None, image_size=(1.0, 1.0), min_length=2):
    """Agrupa detecções por id e quebra em trechos de quadros CONSECUTIVOS
    (sem buracos no índice de quadro -- um buraco de verdade vira o fim de
    um trecho e o início de outro, não é "costurado").

    Retorna lista de trajetórias, cada uma um array (L,5):
    colunas [x,y,w,h] normalizadas por image_size=(W,H), e visibilidade.
    """
    W, H = image_size
    by_id = defaultdict(list)
    for frame, oid, x, y, w, h in detections:
        vis = 1.0
        if visibilities is not None:
            vis = visibilities.get((frame, oid), 1.0)
        by_id[oid].append((int(frame), x / W, y / H, w / W, h / H, vis))

    trajectories = []
    for oid, rows in by_id.items():
        rows.sort(key=lambda r: r[0])
        run = [rows[0]]
        for prev, cur in zip(rows, rows[1:]):
            if cur[0] == prev[0] + 1:
                run.append(cur)
            else:
                if len(run) >= min_length:
                    trajectories.append(np.array([r[1:] for r in run], dtype=np.float32))
                run = [cur]
        if len(run) >= min_length:
            trajectories.append(np.array([r[1:] for r in run], dtype=np.float32))

    return trajectories


def make_windows(trajectories, window: int, stride: int):
    """Corta cada trajetória em janelas fixas de `window` quadros, com
    passo `stride`. Trajetórias mais curtas que `window` são descartadas
    (simplicidade -- há sempre trajetórias longas o suficiente nos nossos
    dados)."""
    windows = []
    for traj in trajectories:
        L = len(traj)
        if L < window:
            continue
        for start in range(0, L - window + 1, stride):
            windows.append(traj[start : start + window])
    return windows


class TrajectoryWindowDataset(Dataset):
    def __init__(self, windows: list[np.ndarray], vis_threshold: float = 0.1):
        self.windows = windows
        self.vis_threshold = vis_threshold

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]  # (T,5): x,y,w,h,vis
        boxes = torch.from_numpy(w[:, :4].copy())
        vis = torch.from_numpy(w[:, 4].copy())
        observed_mask = vis >= self.vis_threshold
        observed_mask[0] = True  # a janela sempre "nasce" de uma observação real
        confs = vis.clone()
        return boxes, confs, observed_mask
