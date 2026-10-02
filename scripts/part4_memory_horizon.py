"""Parte 4 — horizonte de memória efetivo, as duas formas obrigatórias.

1. Analítica: ||∂L_t/∂h_{t-k}|| em função de k, no modelo final (o
   MotionGRU treinado no MOT17, Parte 2) e numa trajetória real do MOT17 --
   mais a comparação já feita na Parte 3 (RNN simples vs. GRU, mesma
   janela, checkpoints reaproveitados).

2. Empírica: quantos quadros o estado sobrevive a uma oclusão antes da
   track morrer ou trocar de ID, em função da duração da oclusão --
   medido no sintético (onde controlamos a duração exatamente), com o
   tracker de verdade (MotionRNNTracker, max_age atual = 5). Comparado
   contra a distribuição REAL de duração de oclusão do MOT17 (medida a
   partir do campo visibility do gt, nas 4 sequências que usamos).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pa2.models.motion_rnn import MotionGRU, MotionRNN, effective_memory_horizon, gradient_vanishing_curve
from pa2.motion_tracker import MotionRNNTracker
from pa2.mot17 import MOT17Sequence, gt_to_tracks
from pa2.synthetic import SyntheticVideoGenerator, simulate_detections

OUT_DIR = ROOT / "outputs" / "part4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOT17_CKPT = ROOT / "checkpoints" / "motion_gru_mot17.pt"
SYNTH_CKPT = ROOT / "checkpoints" / "motion_gru_pretrain_synthetic.pt"
HIDDEN_SIZE = 64
VIS_THRESHOLD = 0.1

SEQUENCES = [("MOT17-09", "val"), ("MOT17-11", "test"), ("MOT17-02", "train"), ("MOT17-04", "diagnostic")]

DURATIONS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40]
N_TRIALS_PER_DURATION = 15
MAX_AGES_TO_COMPARE = [5, 15, 25]  # 5 = o valor atual; os outros antecipam a correção


# ---------------------------------------------------------------------------
# 1) Analítica
# ---------------------------------------------------------------------------
def real_mot17_trajectory(min_length=32):
    """Pega a trajetória de gt mais longa e com maior visibilidade média do
    MOT17-02 (a sequência de treino do checkpoint) -- queremos uma
    trajetória "limpa" pra medir o horizonte de memória do próprio
    mecanismo de BPTT, não confundido por uma oclusão real no meio."""
    seq = MOT17Sequence(ROOT / "data" / "MOT17" / "train" / "MOT17-02")
    gt = seq.load_gt()
    by_id = defaultdict(list)
    for f, i, x, y, w, h, vis in gt:
        by_id[i].append((f, x, y, w, h, vis))

    best, best_score = None, -1
    W, H = seq.info["imWidth"], seq.info["imHeight"]
    for i, rows in by_id.items():
        rows.sort()
        # maior trecho contíguo
        run, best_run = [rows[0]], [rows[0]]
        for prev, cur in zip(rows, rows[1:]):
            if cur[0] == prev[0] + 1:
                run.append(cur)
            else:
                if len(run) > len(best_run):
                    best_run = run
                run = [cur]
        if len(run) > len(best_run):
            best_run = run
        if len(best_run) >= min_length:
            mean_vis = np.mean([r[5] for r in best_run])
            score = mean_vis  # prioriza visibilidade alta (trajetória "limpa")
            if score > best_score:
                best_score, best = score, best_run

    rows = best[:min_length]
    boxes = np.array([[x / W, y / H, w / W, h / H] for _, x, y, w, h, _ in rows], dtype=np.float32)
    vis = np.array([r[5] for r in rows], dtype=np.float32)
    return boxes, vis


def analytic_memory_horizon():
    boxes_np, vis_np = real_mot17_trajectory(min_length=32)
    boxes = torch.from_numpy(boxes_np).unsqueeze(0)
    confs = torch.from_numpy(vis_np).unsqueeze(0)
    observed_mask = torch.ones(1, len(vis_np), dtype=torch.bool)  # trajetória limpa, sem buraco real

    model = MotionGRU(hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(MOT17_CKPT, weights_only=True))
    norms = gradient_vanishing_curve(model, boxes, confs, observed_mask)
    horizon = effective_memory_horizon(norms, threshold_frac=0.01)

    print(f"[analítica] modelo final (motion_gru_mot17.pt, treinado com janela T=16), "
          f"trajetória real do MOT17-02, T testado={len(vis_np)}")
    print(f"  horizonte de memória efetivo (gradiente cai abaixo de 1% do valor em k=0): k={horizon}")
    print(f"  (bem menor que o k~30 do GRU na ablação da Parte 3, que foi treinado com T=32 -- "
          f"o horizonte aprendido parece acompanhar a janela de treino, não só o tipo de célula)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(len(norms)), norms, "o-", color="tab:blue", label="MotionGRU final (MOT17-02)")
    ax.axhline(norms[0] * 0.01, color="gray", linestyle=":", label="limiar de 1% de k=0")
    ax.axvline(horizon, color="tab:red", linestyle="--", label=f"horizonte efetivo (k={horizon})")
    ax.set_xlabel("k (quadros pra trás a partir do último passo)")
    ax.set_ylabel("||∂L / ∂h_{t-k}||")
    ax.set_yscale("log")
    ax.set_title("Horizonte de memória efetivo (analítico)\nmodelo final (treinado com T=16), trajetória real do MOT17-02", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "01_horizonte_analitico.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")
    return horizon


# ---------------------------------------------------------------------------
# 2) Empírica -- distribuição real de oclusão no MOT17
# ---------------------------------------------------------------------------
def dataset_occlusion_durations(vis_threshold=VIS_THRESHOLD):
    all_durations = []
    for name, split in SEQUENCES:
        seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
        gt = seq.load_gt()
        by_id = defaultdict(list)
        for f, i, x, y, w, h, vis in gt:
            by_id[i].append((f, vis))
        for rows in by_id.values():
            rows.sort()
            occluded, start = False, None
            for f, vis in rows:
                if vis < vis_threshold and not occluded:
                    occluded, start = True, f
                elif vis >= vis_threshold and occluded:
                    occluded = False
                    all_durations.append(f - start)
            if occluded:
                all_durations.append(rows[-1][0] - start + 1)
    return np.array(all_durations)


# ---------------------------------------------------------------------------
# 2) Empírica -- sobrevivência de identidade vs. duração da oclusão (sintético)
# ---------------------------------------------------------------------------
# Nota importante: a oclusão ROTEIRIZADA do gerador (pa2/synthetic.py) liga
# duração a velocidade relativa por construção (r_safe/half -- já descoberto
# e documentado faz tempo, na investigação do sweep da Parte 0): durações
# CURTAS exigem velocidade de aproximação MAIOR, o que por si só já quebra o
# tracker (achado da Parte 2: velocidade relativa alta é o que derruba a
# associação por IoU). Usar esse mecanismo aqui confundiria "duração do
# buraco" com "velocidade do objeto" -- exatamente o oposto do que
# queremos isolar.
#
# Por isso, aqui a gente FORÇA um buraco de detecção de duração EXATA `d`
# diretamente (apaga as detecções de um objeto que se move devagar o tempo
# todo, por `d` quadros seguidos), desacoplando duração de velocidade.


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def make_forced_gap_trial(d, speed=0.8, seed=0):
    """2 objetos devagar, sem oclusão roteirizada nenhuma (occlusion_duration=0
    -- o gerador nem embaralha depth/posição, os dois só andam devagar o
    vídeo inteiro). Detector quase limpo. Depois, apaga manualmente as
    detecções do objeto de teste (id=1) por exatamente `d` quadros seguidos,
    centrados no meio do vídeo."""
    n_frames = max(60, d + 40)
    gen = SyntheticVideoGenerator(num_objects=2, speed=speed, occlusion_duration=0, seed=seed, n_frames=n_frames)
    _, gt, vis, _ = gen.generate()
    det = simulate_detections(
        gt, visibilities=vis, drop_prob=0.0, occlusion_sensitivity=0.0,
        coord_noise_std=0.5, size_noise_std=0.3, fp_rate=0.0,
        image_size=(gen.width, gen.height), seed=seed,
    )

    t0 = n_frames // 2
    gap_frames = set(range(t0, t0 + d))
    test_id = 1
    gt_test_box = {f: (x, y, w, h) for f, i, x, y, w, h in gt if i == test_id}

    filtered = []
    for frame, x, y, w, h, score in det:
        if frame in gap_frames and frame in gt_test_box and iou_xywh((x, y, w, h), gt_test_box[frame]) > 0.3:
            continue  # essa detecção é do objeto de teste, dentro do buraco forçado -- descarta
        filtered.append((frame, x, y, w, h, score))

    return gt, filtered, gen.width, gen.height, n_frames, t0, test_id


def track_survives_forced_gap(gt, det_list, model, max_age, image_size, t0, d, test_id):
    W, H = image_size
    by_frame = defaultdict(list)
    for frame, x, y, w, h, score in det_list:
        by_frame[int(frame)].append((x, y, w, h, score))

    n_frames = max(f for f, *_ in gt) if gt else 0
    det_by_frame, score_by_frame = {}, {}
    for f in range(1, n_frames + 1):
        v = by_frame.get(f, [])
        det_by_frame[f] = np.array([[x / W, y / H, w / W, h / H] for x, y, w, h, s in v], dtype=np.float32) if v else np.zeros((0, 4), dtype=np.float32)
        score_by_frame[f] = np.array([s for *_, s in v], dtype=np.float32) if v else np.zeros(0, dtype=np.float32)

    tracker = MotionRNNTracker(model, iou_threshold=0.3, max_age=max_age, match_method="hungarian")
    pred_norm = tracker.run(det_by_frame, score_by_frame)
    pred_by_frame = defaultdict(list)
    for f, i, x, y, w, h in pred_norm:
        pred_by_frame[f].append((i, x * W, y * H, w * W, h * H))

    gt_box_by_frame = {f: (x, y, w, h) for f, i, x, y, w, h in gt if i == test_id}

    def matched_id(frame):
        if frame not in gt_box_by_frame or not pred_by_frame.get(frame):
            return None
        gt_box = gt_box_by_frame[frame]
        best_id, best_iou = None, 0.3
        for pid, px, py, pw, ph in pred_by_frame[frame]:
            iou = iou_xywh(gt_box, (px, py, pw, ph))
            if iou > best_iou:
                best_id, best_iou = pid, iou
        return best_id

    id_before = matched_id(t0 - 1)
    id_after = matched_id(t0 + d)
    if id_before is None or id_after is None:
        return False
    return id_before == id_after


def empirical_survival_curve(model, max_age):
    rates = []
    for d in DURATIONS:
        if d == 0:
            rates.append(1.0)
            continue
        survived = 0
        for trial in range(N_TRIALS_PER_DURATION):
            seed = 20000 + d * 1000 + trial
            gt, det, W, H, n_frames, t0, test_id = make_forced_gap_trial(d, speed=0.8, seed=seed)
            if track_survives_forced_gap(gt, det, model, max_age, (W, H), t0, d, test_id):
                survived += 1
        rates.append(survived / N_TRIALS_PER_DURATION)
    return rates


def plot_empirical_horizon(curves_by_max_age, dataset_durations):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax = axes[0]
    colors = {5: "tab:red", 15: "tab:orange", 25: "tab:green"}
    for max_age, rates in curves_by_max_age.items():
        ax.plot(DURATIONS, rates, "o-", color=colors.get(max_age, "tab:blue"), label=f"max_age={max_age}")
    ax.set_ylabel("taxa de sobrevivência da identidade")
    ax.set_title("Empírica: sobrevivência da identidade vs. duração da oclusão (sintético, tracker de verdade)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    bins = np.arange(0, 85, 5)
    frac_beyond_xlim = np.mean(dataset_durations > 80)
    ax2.hist(np.clip(dataset_durations, 0, 82), bins=bins, color="tab:purple", alpha=0.7)
    ax2.axvline(5, color="tab:red", linestyle="--", label="max_age atual = 5")
    frac_above_5 = np.mean(dataset_durations > 5)
    ax2.set_xlabel("duração da oclusão (quadros)")
    ax2.set_ylabel("nº de eventos de oclusão\nno MOT17 (4 sequências)")
    ax2.set_title(f"Distribuição real de duração de oclusão no MOT17\n"
                  f"({frac_above_5:.0%} dos eventos > max_age=5; eixo cortado em 80, "
                  f"{frac_beyond_xlim:.0%} dos eventos são mais longos ainda)", fontsize=10)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 80)

    plt.tight_layout()
    out_path = OUT_DIR / "02_horizonte_empirico.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")


if __name__ == "__main__":
    analytic_memory_horizon()

    print("\n[empírica] medindo distribuição real de oclusão no MOT17...")
    dataset_durations = dataset_occlusion_durations()
    print(f"  {len(dataset_durations)} eventos, mediana={np.median(dataset_durations):.1f} quadros, "
          f"p75={np.percentile(dataset_durations,75):.1f}, "
          f"fração > max_age=5: {np.mean(dataset_durations>5):.2f}")

    print("\n[empírica] medindo sobrevivência vs. duração no sintético (pode levar alguns minutos)...")
    model = MotionGRU(hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(SYNTH_CKPT, weights_only=True))

    curves = {}
    for max_age in MAX_AGES_TO_COMPARE:
        print(f"  max_age={max_age}...")
        curves[max_age] = empirical_survival_curve(model, max_age)
        print(f"    taxas: {[f'{r:.2f}' for r in curves[max_age]]}")

    plot_empirical_horizon(curves, dataset_durations)
