"""Parte 3 — Ablação, Eixo 1: a célula recorrente.

RNN simples vs. LSTM vs. GRU, no mesmo orçamento aproximado de parâmetros
(ver pa2.models.motion_rnn.matched_hidden_size), variando o comprimento da
janela de BPTT truncado T ∈ {4, 8, 16, 32}, com 3 seeds por configuração
(média ± desvio). Pergunta específica do enunciado: onde a RNN simples
quebra, e isso bate com a história de gradiente que some contada na aula?

Rodado no sintético da Parte 0 (não no MOT17): a pergunta é sobre
comportamento estrutural da célula recorrente sob dependências de longo
alcance, e o gerador sintético deixa controlar exatamente essa dependência
(duração da oclusão) de um jeito que o MOT17 real não permite controlar.
Reaproveita a mesma função de treino/rollout usada na Parte 2 (o mecanismo
de free-running sob oclusão é idêntico).

Duas métricas por configuração, 3 seeds cada:
  1. perda de validação (smooth-L1, mesma janela T do treino);
  2. erro de free-running numa sonda de oclusão LONGA (duração=24 quadros,
     fixa, maior que a maioria dos T testados) -- mede especificamente se o
     modelo consegue carregar informação por mais tempo do que foi treinado
     a ver de uma vez (BPTT truncado em T).

Mais uma análise, direta: a norma de ∂L_T/∂h_{T-k} em função de k (RNN
simples vs. GRU, T=32) -- a curva de gradiente que some dos slides, medida
no nosso próprio modelo e dados.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

# Tem que vir ANTES de qualquer outra coisa usar torch: o modelo é minúsculo
# (poucas centenas de parâmetros por passo) e roda num laço Python por
# quadro -- o paralelismo multi-thread padrão do PyTorch em CPU gasta mais
# tempo sincronizando threads do que a conta em si leva pra rodar, e deixa
# tudo MUITO mais lento (medido: ~6-7x mais devagar com threads
# default do que com 1 thread só, neste sweep especificamente).
torch.set_num_threads(1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from pa2.models.motion_rnn import MotionRNN, matched_hidden_size
from pa2.synthetic import SyntheticVideoGenerator
from pa2.trajectories import TrajectoryWindowDataset, extract_contiguous_trajectories, make_windows

OUT_DIR = ROOT / "outputs" / "part3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / "checkpoints" / "part3_ablation"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "sweep_results.json"

IMAGE_SIZE = (128, 128)
T_VALUES = [4, 8, 16, 32]
CELL_TYPES = ["rnn", "lstm", "gru"]
SEEDS = [0, 1, 2]
VIS_THRESHOLD = 0.1
EPOCHS = 80
GRAD_CLIP = 5.0
LONG_OCCLUSION_DURATION = 24

TARGET_PARAMS = MotionRNN(cell_type="gru", hidden_size=64).num_cell_parameters()
HIDDEN_SIZES = {ct: matched_hidden_size(ct, TARGET_PARAMS) for ct in CELL_TYPES}


def build_trajectory_pool(seeds):
    rng = np.random.default_rng(0)
    all_trajs = []
    for seed in seeds:
        num_objects = int(rng.integers(6, 14))
        speed = float(rng.uniform(1.0, 4.0))
        occlusion_duration = int(rng.integers(0, 22))
        gen = SyntheticVideoGenerator(num_objects=num_objects, speed=speed, occlusion_duration=occlusion_duration, seed=int(seed))
        _, gt, vis, _ = gen.generate()
        trajs = extract_contiguous_trajectories(gt, visibilities=vis, image_size=IMAGE_SIZE, min_length=max(T_VALUES))
        all_trajs.extend(trajs)
    return all_trajs


def build_long_occlusion_probe(n_videos=10, seed_start=5000):
    """Vídeos fixos com oclusão mais longa que a maioria dos T testados --
    mede especificamente a capacidade de carregar informação além do que o
    BPTT truncado viu de uma vez."""
    probes = []
    for i in range(n_videos):
        gen = SyntheticVideoGenerator(
            num_objects=8, speed=2.0, occlusion_duration=LONG_OCCLUSION_DURATION, seed=seed_start + i, n_frames=50
        )
        _, gt, vis, meta = gen.generate()
        occ_id = meta["occluded_id"]
        rows = sorted([(f, x, y, w, h, vis.get((f, occ_id), 1.0)) for f, oid, x, y, w, h in gt if oid == occ_id])
        boxes = np.array([[x / 128, y / 128, w / 128, h / 128] for _, x, y, w, h, _ in rows], dtype=np.float32)
        vis_arr = np.array([r[5] for r in rows], dtype=np.float32)
        probes.append(dict(boxes=boxes, vis=vis_arr, window=meta["window"]))
    return probes


def evaluate_long_occlusion(model, probes):
    model.eval()
    errors = []
    with torch.no_grad():
        for p in probes:
            boxes = torch.from_numpy(p["boxes"]).unsqueeze(0)
            confs = torch.from_numpy(p["vis"]).unsqueeze(0)
            observed_mask = confs >= VIS_THRESHOLD
            observed_mask[:, 0] = True
            preds = model.rollout(boxes, confs, observed_mask)[0].numpy()
            w0, w1 = p["window"]
            true_centers_x = p["boxes"][:, 0] + p["boxes"][:, 2] / 2
            pred_centers_x = preds[:, 0] + preds[:, 2] / 2
            if w1 > w0:
                err = np.mean(np.abs(pred_centers_x[w0:w1] - true_centers_x[w0 + 1 : w1 + 1]))
                errors.append(err)
    model.train()
    return float(np.mean(errors)) if errors else float("nan")


def train_one(cell_type, hidden_size, train_loader, val_loader, seed, epochs=EPOCHS, lr=5e-4):
    torch.manual_seed(seed)
    model = MotionRNN(cell_type=cell_type, hidden_size=hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val, best_state = float("inf"), None
    for _epoch in range(epochs):
        model.train()
        for boxes, confs, mask in train_loader:
            preds = model.rollout(boxes, confs, mask)
            loss = loss_fn(preds[:, :-1], boxes[:, 1:])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for boxes, confs, mask in val_loader:
                preds = model.rollout(boxes, confs, mask)
                val_losses.append(loss_fn(preds[:, :-1], boxes[:, 1:]).item())
            val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val


def load_results() -> list[dict]:
    """Progresso salvo de uma rodada anterior (se existir). Cada combinação
    (célula, T, seed) só é treinada uma vez -- rodar o script de novo
    retoma de onde parou, em vez de refazer tudo."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results: list[dict]):
    """Reescreve o arquivo inteiro a cada combinação concluída -- caro seria
    um problema se fossem milhares de linhas, mas aqui são no máximo 36, e
    em troca cada combinação concluída fica salva em disco imediatamente
    (interromper o script a qualquer momento não perde nada já terminado)."""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_sweep():
    print("Hidden sizes equalizados por orçamento de parâmetros:")
    for ct in CELL_TYPES:
        p = MotionRNN(cell_type=ct, hidden_size=HIDDEN_SIZES[ct]).num_cell_parameters()
        print(f"  {ct:5s}: hidden_size={HIDDEN_SIZES[ct]:4d}  params_celula={p}")

    results = load_results()
    done = {(r["cell_type"], r["T"], r["seed"]) for r in results}
    total = len(CELL_TYPES) * len(T_VALUES) * len(SEEDS)
    print(f"\n[progresso] {len(done)}/{total} combinações já concluídas em execuções anteriores "
          f"({RESULTS_PATH.name})")

    if len(done) == total:
        print("[ok] sweep inteiro já concluído -- nada a treinar, só agregando/plotando.")
        return results

    all_trajs = build_trajectory_pool(range(0, 60))
    val_trajs = build_trajectory_pool(range(900, 915))
    long_probes = build_long_occlusion_probe()
    print(f"{len(all_trajs)} trajetórias de treino, {len(val_trajs)} de validação, "
          f"{len(long_probes)} sondas de oclusão longa ({LONG_OCCLUSION_DURATION} quadros)")

    t0_total = time.time()
    for T in T_VALUES:
        # só monta os DataLoaders deste T se sobrar alguma combinação pra treinar nele
        if all((ct, T, s) in done for ct in CELL_TYPES for s in SEEDS):
            continue

        stride = max(1, T // 2)
        train_windows = make_windows(all_trajs, window=T, stride=stride)
        val_windows = make_windows(val_trajs, window=T, stride=stride)
        train_loader = DataLoader(TrajectoryWindowDataset(train_windows, VIS_THRESHOLD), batch_size=64, shuffle=True)
        val_loader = DataLoader(TrajectoryWindowDataset(val_windows, VIS_THRESHOLD), batch_size=64)

        for cell_type in CELL_TYPES:
            H = HIDDEN_SIZES[cell_type]
            for seed in SEEDS:
                if (cell_type, T, seed) in done:
                    continue

                t0 = time.time()
                model, val_loss = train_one(cell_type, H, train_loader, val_loader, seed)
                long_err = evaluate_long_occlusion(model, long_probes)
                dt = time.time() - t0

                results.append(dict(
                    cell_type=cell_type, T=T, seed=seed,
                    val_loss=float(val_loss), long_occ_error=float(long_err),
                ))
                save_results(results)  # grava IMEDIATAMENTE -- cada linha concluída persiste na hora
                done.add((cell_type, T, seed))

                print(f"  [{len(done):2d}/{total}] T={T:3d} cell={cell_type:5s} seed={seed} -> "
                      f"val_loss={val_loss:.6f} long_occ_err={long_err:.4f} ({dt:.1f}s)")
                if seed == 0:
                    torch.save(model.state_dict(), CKPT_DIR / f"{cell_type}_T{T}_seed0.pt")

    print(f"\n[ok] {len(done)}/{total} combinações concluídas (esta execução: {time.time()-t0_total:.1f}s)")
    if len(done) < total:
        print(f"[info] faltam {total - len(done)} combinações -- rode o script de novo pra continuar.")
    return results


def aggregate(results):
    agg = {}
    for ct in CELL_TYPES:
        for T in T_VALUES:
            rows = [r for r in results if r["cell_type"] == ct and r["T"] == T]
            val_losses = [r["val_loss"] for r in rows]
            long_errs = [r["long_occ_error"] for r in rows]
            agg[(ct, T)] = dict(
                val_mean=np.mean(val_losses), val_std=np.std(val_losses),
                long_mean=np.mean(long_errs), long_std=np.std(long_errs),
            )
    return agg


def plot_ablation(agg):
    colors = {"rnn": "tab:red", "lstm": "tab:orange", "gru": "tab:blue"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ct in CELL_TYPES:
        means = [agg[(ct, T)]["val_mean"] for T in T_VALUES]
        stds = [agg[(ct, T)]["val_std"] for T in T_VALUES]
        axes[0].errorbar(T_VALUES, means, yerr=stds, marker="o", label=ct.upper(), color=colors[ct], capsize=4)
    axes[0].set_xlabel("T (comprimento da janela de BPTT truncado)")
    axes[0].set_ylabel("perda de validação (smooth-L1)")
    axes[0].set_yscale("log")
    axes[0].set_title("Perda de validação por T\n(média ± desvio, 3 seeds)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for ct in CELL_TYPES:
        means = [agg[(ct, T)]["long_mean"] for T in T_VALUES]
        stds = [agg[(ct, T)]["long_std"] for T in T_VALUES]
        axes[1].errorbar(T_VALUES, means, yerr=stds, marker="s", label=ct.upper(), color=colors[ct], capsize=4)
    axes[1].set_xlabel("T (comprimento da janela de BPTT truncado)")
    axes[1].set_ylabel(f"erro de free-running\nna sonda de oclusão longa ({LONG_OCCLUSION_DURATION} quadros)")
    axes[1].set_title("Erro na oclusão longa por T\n(média ± desvio, 3 seeds)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "01_ablacao_celula_recorrente.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")


def print_table(agg):
    print("\nTabela completa (média ± desvio, 3 seeds):")
    print(f"{'célula':6s} {'T':>4s}  {'val_loss':>18s}  {'erro_oclusao_longa':>20s}")
    for ct in CELL_TYPES:
        for T in T_VALUES:
            a = agg[(ct, T)]
            print(f"{ct:6s} {T:4d}  {a['val_mean']:.6f} ± {a['val_std']:.6f}   "
                  f"{a['long_mean']:7.4f} ± {a['long_std']:.4f}")


def gradient_vanishing_curve(model, boxes, confs, observed_mask):
    """||∂L/∂h_{T-2-k}|| em função de k, onde L é a perda da ÚLTIMA previsão
    válida da sequência (feita no passo T-2, tentando acertar boxes[T-1] --
    a previsão do passo T-1 tentaria prever um quadro T que não existe, por
    isso o laço só roda até T-2, nunca gera essa previsão fora do range).
    h.retain_grad() porque são tensores intermediários (não-folha), que não
    guardam gradiente por padrão."""
    model.eval()
    model.zero_grad()
    B, T, _ = boxes.shape
    state = model.init_hidden(B)
    cur_box, cur_conf = boxes[:, 0], confs[:, 0]
    hs = []
    pred = None
    for t in range(T - 1):  # produz previsões pra boxes[1..T-1], nunca além
        state, pred = model.step(state, cur_box, cur_conf)
        h = state[0]
        h.retain_grad()
        hs.append(h)
        obs_next = observed_mask[:, t + 1].unsqueeze(-1).float()
        cur_box = obs_next * boxes[:, t + 1] + (1 - obs_next) * pred
        cur_conf = observed_mask[:, t + 1].float() * confs[:, t + 1]

    loss_fn = nn.SmoothL1Loss()
    final_loss = loss_fn(pred, boxes[:, -1])  # última previsão (passo T-2) vs. última caixa real
    final_loss.backward()

    norms = [h.grad.norm().item() if h.grad is not None else 0.0 for h in hs]
    norms = norms[::-1]  # norms[0] = ||dL/dh_{T-2}|| (k=0), norms[k] = ||dL/dh_{T-2-k}||
    return norms


def plot_gradient_vanishing(T=32):
    """RNN simples vs. GRU, mesma janela T, usando os checkpoints seed=0 já
    treinados no sweep principal (reaproveitados, como a Parte 4 vai fazer
    de novo com mais detalhe)."""
    gen = SyntheticVideoGenerator(num_objects=8, speed=2.0, occlusion_duration=0, seed=7654, n_frames=T)
    _, gt, vis, _ = gen.generate()
    by_id = {}
    for f, oid, x, y, w, h in gt:
        by_id.setdefault(oid, []).append((f, x, y, w, h, vis.get((f, oid), 1.0)))
    traj = max(by_id.values(), key=len)
    traj = sorted(traj)[:T]
    boxes_np = np.array([[x / 128, y / 128, w / 128, h / 128] for _, x, y, w, h, _ in traj], dtype=np.float32)
    vis_np = np.array([r[5] for r in traj], dtype=np.float32)
    boxes = torch.from_numpy(boxes_np).unsqueeze(0)
    confs = torch.from_numpy(vis_np).unsqueeze(0)
    observed_mask = torch.ones(1, T, dtype=torch.bool)  # sem oclusão aqui -- queremos só o efeito do BPTT, isolado

    fig, ax = plt.subplots(figsize=(7, 5))
    for ct, color in [("rnn", "tab:red"), ("gru", "tab:blue")]:
        ckpt_path = CKPT_DIR / f"{ct}_T{T}_seed0.pt"
        model = MotionRNN(cell_type=ct, hidden_size=HIDDEN_SIZES[ct])
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        norms = gradient_vanishing_curve(model, boxes, confs, observed_mask)
        ax.plot(range(len(norms)), norms, "o-", color=color, label=ct.upper())

    ax.set_xlabel("k (quadros pra trás a partir do último passo)")
    ax.set_ylabel("||∂L_T / ∂h_{T-1-k}||")
    ax.set_yscale("log")
    ax.set_title(f"Gradiente que some: RNN simples vs. GRU (T={T}, checkpoints seed=0)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "02_gradiente_que_some.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")


if __name__ == "__main__":
    results = run_sweep()

    if results:
        agg = aggregate(results)
        print_table(agg)
        plot_ablation(agg)
        print("\n[info] tabela/figura acima refletem só as combinações já concluídas "
              "(rode de novo depois de completar o sweep pra ter a versão final).")

    rnn_ckpt = CKPT_DIR / "rnn_T32_seed0.pt"
    gru_ckpt = CKPT_DIR / "gru_T32_seed0.pt"
    if rnn_ckpt.exists() and gru_ckpt.exists():
        plot_gradient_vanishing(T=32)
    else:
        print("[info] ainda faltam os checkpoints rnn/gru em T=32 (seed 0) pra plotar "
              "a curva de gradiente que some -- gerada automaticamente quando essas "
              "combinações forem concluídas.")
