"""Parte 2 — avalia o MotionRNNTracker (Trilha A) nas mesmas 4 sequências e
com a mesma fonte de detecção (SDP) da Parte 1, e compara lado a lado com o
NaiveIoUTracker (baseline). Essa comparação é o que a apresentação precisa
mostrar: "em que aspecto a representação aprendida melhora o fracasso da
Parte 1" (ou não -- reportamos os dois casos, com e sem melhora).
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

from metrics import compute_mot_metrics
from pa2.baseline_tracker import NaiveIoUTracker
from pa2.models.motion_rnn import MotionGRU
from pa2.motion_tracker import MotionRNNTracker
from pa2.mot17 import MOT17Sequence, gt_to_tracks

OUT_DIR = ROOT / "outputs" / "part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = ROOT / "checkpoints" / "motion_gru_mot17.pt"

SEQUENCES = [
    ("MOT17-09", "val", 10.1),
    ("MOT17-11", "test", 10.5),
    ("MOT17-02", "train", 31.0),
    ("MOT17-04", "diagnostic", 45.3),
]
DETECTOR = "SDP"
HIDDEN_SIZE = 64


def dets_by_frame_raw(det_list):
    by_frame = {}
    for frame, x, y, w, h, score in det_list:
        by_frame.setdefault(int(frame), []).append((x, y, w, h, score))
    return by_frame


def run_baseline(seq, gt, det_list):
    by_frame = dets_by_frame_raw(det_list)
    det_by_frame = {f: np.array([[x, y, w, h] for x, y, w, h, s in v]) for f, v in by_frame.items()}
    for f in range(1, seq.n_frames + 1):
        det_by_frame.setdefault(f, np.zeros((0, 4)))
    tracker = NaiveIoUTracker(iou_threshold=0.3, max_age=5, match_method="hungarian")
    pred = tracker.run(det_by_frame)
    pred_tracks = [(f, i, x, y, w, h) for f, i, x, y, w, h in pred]
    return compute_mot_metrics(gt, pred_tracks)


def run_motion(seq, gt, det_list, model):
    W, H = seq.info["imWidth"], seq.info["imHeight"]
    by_frame = dets_by_frame_raw(det_list)

    det_by_frame, score_by_frame = {}, {}
    for f, v in by_frame.items():
        det_by_frame[f] = np.array([[x / W, y / H, w / W, h / H] for x, y, w, h, s in v], dtype=np.float32)
        score_by_frame[f] = np.array([s for *_, s in v], dtype=np.float32)
    for f in range(1, seq.n_frames + 1):
        det_by_frame.setdefault(f, np.zeros((0, 4), dtype=np.float32))
        score_by_frame.setdefault(f, np.zeros(0, dtype=np.float32))

    tracker = MotionRNNTracker(model, iou_threshold=0.3, max_age=5, match_method="hungarian")
    pred_norm = tracker.run(det_by_frame, score_by_frame)
    pred_tracks = [(f, i, x * W, y * H, w * W, h * H) for f, i, x, y, w, h in pred_norm]
    return compute_mot_metrics(gt, pred_tracks)


def displacement_stats(gt):
    """Deslocamento do centro da caixa entre quadros consecutivos,
    relativo à largura da própria caixa -- mede o quanto um objeto
    realmente se move por quadro, na escala que importa pra IoU (se ele se
    move pouco relativo ao próprio tamanho, a hipótese de velocidade zero
    já é quase ótima, e um modelo de movimento tem pouco a ganhar)."""
    by_id = defaultdict(list)
    for f, i, x, y, w, h in gt:
        by_id[i].append((f, x + w / 2, y + h / 2, w))
    rel = []
    for rows in by_id.values():
        rows.sort()
        for (f0, x0, y0, w0), (f1, x1, y1, _w1) in zip(rows, rows[1:]):
            if f1 == f0 + 1:
                rel.append(np.hypot(x1 - x0, y1 - y0) / w0)
    rel = np.array(rel)
    return dict(mean=float(rel.mean()), median=float(np.median(rel)), n=len(rel))


def main():
    model = MotionGRU(hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()
    print(f"[ok] checkpoint carregado de {CKPT_PATH}")

    results = []
    for name, split, dens in SEQUENCES:
        seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
        gt_rows = seq.load_gt()
        gt = gt_to_tracks(gt_rows)
        det_list = seq.load_det(variant=DETECTOR)

        m_base = run_baseline(seq, gt, det_list)
        m_motion = run_motion(seq, gt, det_list, model)
        disp = displacement_stats(gt)

        results.append(dict(name=name, dens=dens, baseline=m_base, motion=m_motion, disp=disp))
        print(
            f"[{name}] dens={dens:5.1f}  desloc.rel.mediano={disp['median']:.3f} (x largura da caixa)  "
            f"IDF1 base={m_base['idf1']:.3f} motion={m_motion['idf1']:.3f}  |  "
            f"switches base={m_base['id_switches']:4d} motion={m_motion['id_switches']:4d}  |  "
            f"frags base={m_base['fragmentations']:4d} motion={m_motion['fragmentations']:4d}  |  "
            f"ratio base={m_base['ratio']:.2f} motion={m_motion['ratio']:.2f}"
        )
    return results


def plot_comparison(results):
    results = sorted(results, key=lambda r: r["dens"])
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    specs = [
        ("idf1", "IDF1 (maior é melhor)", axes[0, 0]),
        ("id_switches", "ID switches (menor é melhor)", axes[0, 1]),
        ("fragmentations", "Fragmentações (menor é melhor)", axes[1, 0]),
        ("ratio", "Razão ids previstas/verdadeiras (quanto mais perto de 1, melhor)", axes[1, 1]),
    ]
    for key, title, ax in specs:
        base_vals = [r["baseline"][key] for r in results]
        motion_vals = [r["motion"][key] for r in results]
        ax.bar(x - width / 2, base_vals, width, label="baseline ingênuo (Parte 1)", color="tab:gray")
        ax.bar(x + width / 2, motion_vals, width, label="MotionGRU (Parte 2)", color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title, fontsize=10)
        if key == "ratio":
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)

    plt.suptitle(f"Parte 2 vs. Parte 1: mesma fonte de detecção ({DETECTOR}), mesmas 4 sequências", fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "04_comparacao_parte1_vs_parte2.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] figura de comparação salva em {out_path}")


def plot_displacement_context(results):
    """Conecta o resultado de volta à Parte 0: a velocidade (deslocamento
    relativo ao tamanho da caixa) foi o eixo que mais quebrava o tracker
    ingênuo no sweep sintético (ver outputs/part0/03_difficulty_sweep.png).
    Medindo esse MESMO eixo nos vídeos do sweep (mesma definição: deslocamento
    do centro entre quadros / largura da caixa), a velocidade mais BAIXA
    testada (0.5 px/quadro, onde o baseline ingênuo já tinha IDF1~0.82, ou
    seja, "fácil") corresponde a um deslocamento relativo mediano de 0.030.
    As barras abaixo mostram que as 4 sequências do MOT17 estão TODAS no
    regime fácil ou mais fácil ainda -- nenhuma chega perto do regime onde o
    sweep mostrou o tracker ingênuo quebrando de verdade (0.15+, speed>=3)."""
    results = sorted(results, key=lambda r: r["dens"])
    names = [r["name"] for r in results]
    medians = [r["disp"]["median"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, medians, color="tab:blue", label="MOT17 (gt real)")
    ax.axhline(0.030, color="tab:green", linestyle="--",
               label="sweep sintético, speed=0.5 px/quadro (\"fácil\", IDF1 ingênuo~0.82)")
    ax.axhline(0.153, color="tab:red", linestyle="--",
               label="sweep sintético, speed=3.0 px/quadro (onde o ingênuo começa a quebrar, IDF1~0.55)")
    ax.set_ylabel("deslocamento mediano por quadro\n(fração da largura da própria caixa)")
    ax.set_title("Por que o MotionGRU ajuda pouco no MOT17: as 4 sequências\nestão no regime 'fácil' do sweep da Parte 0")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = OUT_DIR / "05_deslocamento_relativo_mot17.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] figura de contexto (deslocamento) salva em {out_path}")


if __name__ == "__main__":
    results = main()
    plot_comparison(results)
    plot_displacement_context(results)
