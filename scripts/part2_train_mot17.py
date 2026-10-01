"""Parte 2, Trilha A — treino do MotionGRU em trajetórias de ground truth
do MOT17 real (MOT17-02 = treino, MOT17-09 = validação -- a MESMA divisão
documentada na Parte 1; MOT17-11 continua intocada, held-out pra avaliação
final). Começa do checkpoint pré-treinado no sintético (warm start).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pa2.models.motion_rnn import MotionGRU
from pa2.mot17 import MOT17Sequence, gt_to_tracks
from pa2.trajectories import TrajectoryWindowDataset, extract_contiguous_trajectories, make_windows

OUT_DIR = ROOT / "outputs" / "part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / "checkpoints"

WINDOW = 16
STRIDE = 4
VIS_THRESHOLD = 0.1
HIDDEN_SIZE = 64
PRETRAIN_CKPT = CKPT_DIR / "motion_gru_pretrain_synthetic.pt"
FINAL_CKPT = CKPT_DIR / "motion_gru_mot17.pt"


def load_windows(split, name):
    seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
    gt_rows = seq.load_gt()  # já filtra classe pedestrian + conf==1 (ver pa2/mot17.py)
    gt_with_vis = [(f, i, x, y, w, h) for f, i, x, y, w, h, _ in gt_rows]
    vis_map = {(f, i): v for f, i, _, _, _, _, v in gt_rows}
    image_size = (seq.info["imWidth"], seq.info["imHeight"])
    trajs = extract_contiguous_trajectories(gt_with_vis, visibilities=vis_map, image_size=image_size, min_length=WINDOW)
    windows = make_windows(trajs, window=WINDOW, stride=STRIDE)
    print(f"[{name}] {len(gt_rows)} linhas de gt -> {len(trajs)} trajetórias -> {len(windows)} janelas")
    return windows


def evaluate(model, loader, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for boxes, confs, observed_mask in loader:
            preds = model.rollout(boxes, confs, observed_mask)
            losses.append(loss_fn(preds[:, :-1], boxes[:, 1:]).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(model, train_loader, val_loader, epochs, lr=5e-4, weight_decay=1e-4):
    """Early stopping por perda de validação: o MOT17-02 só tem 57
    trajetórias reais (mesmo com milhares de janelas sobrepostas, que são
    bem correlacionadas entre si), então o modelo decora rápido -- a perda
    de treino cai a quase zero em poucas épocas enquanto a de validação
    piora depois de um ponto. Guardamos os pesos da época de MENOR perda de
    validação, não os da última época."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for boxes, confs, observed_mask in train_loader:
            preds = model.rollout(boxes, confs, observed_mask)
            loss = loss_fn(preds[:, :-1], boxes[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        train_loss = total / n
        val_loss = evaluate(model, val_loader, loss_fn)
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}")

    print(f"[early stopping] melhor época: {best_epoch} (val={best_val:.6f})")
    model.load_state_dict(best_state)
    return train_hist, val_hist, best_epoch


def plot_curves(train_hist, val_hist, best_epoch):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(train_hist, label="treino (MOT17-02)")
    ax.plot(val_hist, label="validação (MOT17-09)")
    ax.axvline(best_epoch, color="gray", linestyle=":", label=f"checkpoint salvo (época {best_epoch})")
    ax.set_xlabel("época")
    ax.set_ylabel("smooth-L1")
    ax.set_yscale("log")
    ax.set_title("Treino do MotionGRU no MOT17 (warm start do sintético)\nMOT17-02 só tem 57 trajetórias -- decora rápido, daí o early stopping")
    ax.legend()
    plt.tight_layout()
    out_path = OUT_DIR / "03_mot17_train_loss.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] curva salva em {out_path}")


if __name__ == "__main__":
    torch.manual_seed(0)

    train_windows = load_windows("train", "MOT17-02")
    val_windows = load_windows("val", "MOT17-09")

    train_ds = TrajectoryWindowDataset(train_windows, vis_threshold=VIS_THRESHOLD)
    val_ds = TrajectoryWindowDataset(val_windows, vis_threshold=VIS_THRESHOLD)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = MotionGRU(hidden_size=HIDDEN_SIZE)
    if PRETRAIN_CKPT.exists():
        model.load_state_dict(torch.load(PRETRAIN_CKPT, weights_only=True))
        print(f"[ok] warm start a partir de {PRETRAIN_CKPT}")
    else:
        print("[aviso] checkpoint do pré-treino sintético não encontrado, treinando do zero")

    train_hist, val_hist, best_epoch = train(model, train_loader, val_loader, epochs=150)
    plot_curves(train_hist, val_hist, best_epoch)

    torch.save(model.state_dict(), FINAL_CKPT)
    print(f"[ok] checkpoint final salvo em {FINAL_CKPT}")
