"""Algoritmos de associação (matching) entre dois conjuntos de caixas por IoU.

Dois métodos, ambos usados ao longo do PA para comparar "ingênuo guloso" vs
"ótimo global por quadro":
  - greedy_match: guloso, ordena todos os pares por IoU decrescente e vai
    casando o que sobra (rápido, sub-ótimo, é o que SORT/variantes simples fazem).
  - hungarian_match: ótimo (scipy.optimize.linear_sum_assignment) para o quadro
    em questão, maximizando IoU total sujeito ao limiar.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def greedy_match(iou_mat: np.ndarray, iou_threshold: float = 0.3):
    """Retorna (matches, unmatched_rows, unmatched_cols).

    matches: lista de (row, col). Varre todos os pares (r,c) em ordem de IoU
    decrescente e casa gulosamente, pulando linhas/colunas já usadas ou pares
    abaixo do limiar.
    """
    n_rows, n_cols = iou_mat.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    pairs = [(iou_mat[r, c], r, c) for r in range(n_rows) for c in range(n_cols)]
    pairs.sort(key=lambda t: t[0], reverse=True)

    used_rows, used_cols = set(), set()
    matches = []
    for iou, r, c in pairs:
        if iou < iou_threshold:
            break
        if r in used_rows or c in used_cols:
            continue
        matches.append((r, c))
        used_rows.add(r)
        used_cols.add(c)

    unmatched_rows = [r for r in range(n_rows) if r not in used_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in used_cols]
    return matches, unmatched_rows, unmatched_cols


def hungarian_match(iou_mat: np.ndarray, iou_threshold: float = 0.3):
    """Assignment ótimo (maximiza soma de IoU) para o quadro, com corte por limiar."""
    n_rows, n_cols = iou_mat.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    used_rows, used_cols = set(), set()
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_threshold:
            matches.append((int(r), int(c)))
            used_rows.add(r)
            used_cols.add(c)

    unmatched_rows = [r for r in range(n_rows) if r not in used_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in used_cols]
    return matches, unmatched_rows, unmatched_cols
