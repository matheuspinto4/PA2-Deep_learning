"""Parte 2, Trilha A — pré-treino do MotionGRU no ambiente sintético da
Parte 0, antes de arriscar tempo de CPU no MOT17 real.

Por quê: mesma lógica da Parte 0 inteira -- um ambiente onde controlamos a
dificuldade (velocidade, duração de oclusão) deixa validar que o modelo
aprende a tarefa certa (prever a caixa seguinte, inclusive atravessando um
buraco de oclusão via free-running) antes de gastar tempo no MOT17.
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
from pa2.synthetic import SyntheticVideoGenerator
from pa2.trajectories import TrajectoryWindowDataset, extract_contiguous_trajectories, make_windows

OUT_DIR = ROOT / "outputs" / "part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
WINDOW = 16
STRIDE = 4
VIS_THRESHOLD = 0.1
HIDDEN_SIZE = 64


def build_dataset(seeds, n_videos_desc=""):
    rng = np.random.default_rng(0)
    all_trajs = []
    for seed in seeds:
        num_objects = int(rng.integers(6, 14))
        speed = float(rng.uniform(1.0, 4.0))
        occlusion_duration = int(rng.integers(0, 22))
        gen = SyntheticVideoGenerator(
            num_objects=num_objects, speed=speed, occlusion_duration=occlusion_duration, seed=int(seed)
        )
        _, gt, vis, _ = gen.generate()
        trajs = extract_contiguous_trajectories(gt, visibilities=vis, image_size=IMAGE_SIZE, min_length=WINDOW)
        all_trajs.extend(trajs)
    windows = make_windows(all_trajs, window=WINDOW, stride=STRIDE)
    print(f"[dataset {n_videos_desc}] {len(seeds)} vídeos -> {len(all_trajs)} trajetórias -> {len(windows)} janelas")
    return windows


def train(model, dataset, epochs=40, batch_size=64, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    history = []
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for boxes, confs, observed_mask in loader:
            preds = model.rollout(boxes, confs, observed_mask)
            loss = loss_fn(preds[:, :-1], boxes[:, 1:])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / n_batches
        history.append(avg)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  loss={avg:.6f}")
    return history


def plot_loss_curve(history):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history)
    ax.set_xlabel("época")
    ax.set_ylabel("smooth-L1 (treino)")
    ax.set_title("Pré-treino do MotionGRU no sintético")
    ax.set_yscale("log")
    plt.tight_layout()
    out_path = OUT_DIR / "01_pretrain_loss.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] curva de perda salva em {out_path}")


@torch.no_grad()
def qualitative_check(model, seed=999):
    """Gera UM vídeo sintético novo (nunca visto no treino), roda o modelo
    em modo de inferência de verdade (observed_mask vindo da visibilidade
    real, igual ao treino) no objeto que foi roteirizado pra sofrer
    oclusão, e plota previsto vs. verdadeiro, com a janela de oclusão
    destacada -- a pergunta que essa figura responde: o modelo consegue
    "adivinhar" o movimento do objeto escondido, ou só mantém a última
    posição conhecida parada?"""
    gen = SyntheticVideoGenerator(num_objects=8, speed=2.0, occlusion_duration=14, seed=seed, n_frames=50)
    _, gt, vis, meta = gen.generate()
    occ_id = meta["occluded_id"]
    w0, w1 = meta["window"]

    traj_rows = sorted([(f, x, y, w, h, vis.get((f, occ_id), 1.0)) for f, oid, x, y, w, h in gt if oid == occ_id])
    frames = [r[0] for r in traj_rows]
    boxes_np = np.array([[x / 128, y / 128, w / 128, h / 128] for _, x, y, w, h, _ in traj_rows], dtype=np.float32)
    vis_np = np.array([r[5] for r in traj_rows], dtype=np.float32)

    boxes = torch.from_numpy(boxes_np).unsqueeze(0)
    confs = torch.from_numpy(vis_np).unsqueeze(0)
    observed_mask = (confs >= VIS_THRESHOLD)
    observed_mask[:, 0] = True

    preds = model.rollout(boxes, confs, observed_mask)[0].numpy()
    true_centers = boxes_np[:, 0] * 128 + boxes_np[:, 2] * 128 / 2
    pred_centers_x = preds[:, 0] * 128 + preds[:, 2] * 128 / 2  # previsão feita em t pra t+1

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(frames, true_centers, "o-", color="limegreen", label="posição verdadeira (centro x)", markersize=3)
    ax.plot(frames[1:], pred_centers_x[:-1], "s--", color="tab:red", label="previsão (feita 1 quadro antes)", markersize=3)
    ax.axvspan(w0, w1, color="gray", alpha=0.25, label=f"janela de oclusão roteirizada ({w1-w0} quadros)")
    ax.set_xlabel("quadro")
    ax.set_ylabel("posição x do centro (px)")
    ax.set_title("MotionGRU: previsão vs. verdade atravessando uma oclusão (vídeo nunca visto no treino)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = OUT_DIR / "02_pretrain_occlusion_rollout.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] figura qualitativa salva em {out_path}")

    mae_during_occlusion = np.mean(np.abs(pred_centers_x[w0:w1] - true_centers[w0 + 1 : w1 + 1])) if w1 > w0 else float("nan")
    mae_freeze = np.mean(np.abs(true_centers[w0] - true_centers[w0 + 1 : w1 + 1])) if w1 > w0 else float("nan")
    # extrapolação de velocidade constante a partir dos 2 últimos quadros observados antes da oclusão
    v_const = true_centers[w0] - true_centers[w0 - 1] if w0 > 0 else 0.0
    steps = np.arange(1, w1 - w0 + 1)
    mae_const_vel = np.mean(np.abs((true_centers[w0] + v_const * steps) - true_centers[w0 + 1 : w1 + 1])) if w1 > w0 else float("nan")
    print(f"[info] MAE (px) durante a oclusão -- MotionGRU: {mae_during_occlusion:.2f}  "
          f"congelado: {mae_freeze:.2f}  velocidade constante: {mae_const_vel:.2f}")


if __name__ == "__main__":
    torch.manual_seed(0)

    train_windows = build_dataset(range(0, 60), "treino")
    val_windows = build_dataset(range(900, 910), "validação (não usada no treino)")

    dataset = TrajectoryWindowDataset(train_windows, vis_threshold=VIS_THRESHOLD)
    val_dataset = TrajectoryWindowDataset(val_windows, vis_threshold=VIS_THRESHOLD)

    model = MotionGRU(hidden_size=HIDDEN_SIZE)
    history = train(model, dataset, epochs=200)
    plot_loss_curve(history)

    # perda de validação (sanity check rápido de que não é só decorar o treino)
    loader = DataLoader(val_dataset, batch_size=64)
    loss_fn = nn.SmoothL1Loss()
    with torch.no_grad():
        val_losses = []
        for boxes, confs, observed_mask in loader:
            preds = model.rollout(boxes, confs, observed_mask)
            val_losses.append(loss_fn(preds[:, :-1], boxes[:, 1:]).item())
        print(f"[info] perda de validação (sintético, seeds nunca vistas): {np.mean(val_losses):.6f}")

    qualitative_check(model)

    ckpt_path = CKPT_DIR / "motion_gru_pretrain_synthetic.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[ok] checkpoint salvo em {ckpt_path}")
