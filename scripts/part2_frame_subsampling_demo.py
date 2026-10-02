"""Parte 2 — demonstração direcionada: em que regime o MotionGRU realmente
ajuda, usando dado real (não só sintético).

Medimos (pergunta anterior) que as 4 sequências do MOT17 vivem no regime
"fácil" do sweep de velocidade da Parte 0 (deslocamento relativo entre
0.010 e 0.037, contra o limiar de quebra do tracker ingênuo em ~0.153) --
isso explica por que a Parte 2 deu uma melhora pequena/nula na taxa nativa.

Esse script empurra o MOT17 real pro regime "difícil" SEM RETREINAR nada:
subamostra os quadros (mantém 1 a cada k, "k" = stride), o que multiplica o
deslocamento real por quadro por ~k (o objeto percorre a mesma distância
física em menos passos). Rodamos os DOIS trackers (ingênuo e MotionGRU,
mesmo checkpoint da Parte 2, treinado a ∆t=1 -- deliberadamente NÃO
alimentamos ∆t, pra testar resiliência crua) na mesma sequência subamostrada
e comparamos.

Nota: isso é conceitualmente a mesma manobra do teste de estresse de queda
de frame rate da Parte 5, mas aqui é só uma demonstração focada (um
experimento pequeno, não o teste de estresse completo da Parte 5 -- que
escolhemos ser sobre degradação de detector, não frame rate).
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
    ("MOT17-09", "val"),
    ("MOT17-11", "test"),
    ("MOT17-02", "train"),
    ("MOT17-04", "diagnostic"),
]
DETECTOR = "SDP"
HIDDEN_SIZE = 64
STRIDES = [1, 2, 3, 4, 5, 8]

# limiares de referência medidos no sweep sintético da Parte 0 (mesma
# definição: deslocamento mediano do centro / largura da caixa)
REF_EASY = 0.030  # speed=0.5 px/quadro, onde o ingênuo já ia bem (IDF1~0.82)
REF_BREAK = 0.153  # speed=3.0 px/quadro, onde o ingênuo começa a quebrar (IDF1~0.55)


def subsample(gt, det_list, stride, n_frames):
    """Mantém 1 a cada `stride` quadros originais e renumera pra 1..N
    consecutivos -- os trackers tratam como se fosse um vídeo gravado a uma
    taxa de quadros `stride` vezes menor, sem saber que é sintético."""
    keep_frames = list(range(1, n_frames + 1, stride))
    remap = {old: new for new, old in enumerate(keep_frames, start=1)}

    new_gt = [(remap[f], i, x, y, w, h) for f, i, x, y, w, h in gt if f in remap]
    new_det = [(remap[f], x, y, w, h, s) for f, x, y, w, h, s in det_list if f in remap]
    return new_gt, new_det, len(keep_frames)


def relative_displacement(gt):
    by_id = defaultdict(list)
    for f, i, x, y, w, h in gt:
        by_id[i].append((f, x + w / 2, y + h / 2, w))
    rel = []
    for rows in by_id.values():
        rows.sort()
        for (f0, x0, y0, w0), (f1, x1, y1, _w1) in zip(rows, rows[1:]):
            if f1 == f0 + 1:
                rel.append(np.hypot(x1 - x0, y1 - y0) / w0)
    return float(np.median(rel)) if rel else float("nan")


def dets_by_frame_raw(det_list):
    by_frame = {}
    for frame, x, y, w, h, score in det_list:
        by_frame.setdefault(int(frame), []).append((x, y, w, h, score))
    return by_frame


def run_baseline(n_frames, gt, det_list):
    by_frame = dets_by_frame_raw(det_list)
    det_by_frame = {f: np.array([[x, y, w, h] for x, y, w, h, s in v]) for f, v in by_frame.items()}
    for f in range(1, n_frames + 1):
        det_by_frame.setdefault(f, np.zeros((0, 4)))
    tracker = NaiveIoUTracker(iou_threshold=0.3, max_age=5, match_method="hungarian")
    pred = tracker.run(det_by_frame)
    pred_tracks = [(f, i, x, y, w, h) for f, i, x, y, w, h in pred]
    return compute_mot_metrics(gt, pred_tracks)


def run_motion(n_frames, image_size, gt, det_list, model):
    W, H = image_size
    by_frame = dets_by_frame_raw(det_list)

    det_by_frame, score_by_frame = {}, {}
    for f, v in by_frame.items():
        det_by_frame[f] = np.array([[x / W, y / H, w / W, h / H] for x, y, w, h, s in v], dtype=np.float32)
        score_by_frame[f] = np.array([s for *_, s in v], dtype=np.float32)
    for f in range(1, n_frames + 1):
        det_by_frame.setdefault(f, np.zeros((0, 4), dtype=np.float32))
        score_by_frame.setdefault(f, np.zeros(0, dtype=np.float32))

    tracker = MotionRNNTracker(model, iou_threshold=0.3, max_age=5, match_method="hungarian")
    pred_norm = tracker.run(det_by_frame, score_by_frame)
    pred_tracks = [(f, i, x * W, y * H, w * W, h * H) for f, i, x, y, w, h in pred_norm]
    return compute_mot_metrics(gt, pred_tracks)


def main():
    model = MotionGRU(hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()
    print(f"[ok] checkpoint carregado de {CKPT_PATH} (treinado a passo=1 quadro, NAO re-treinado aqui)\n")

    results = []
    for name, split in SEQUENCES:
        seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
        gt_full = gt_to_tracks(seq.load_gt())
        det_full = seq.load_det(variant=DETECTOR)
        image_size = (seq.info["imWidth"], seq.info["imHeight"])

        print(f"=== {name} ===")
        for stride in STRIDES:
            gt_s, det_s, n_frames_s = subsample(gt_full, det_full, stride, seq.n_frames)
            disp = relative_displacement(gt_s)

            m_base = run_baseline(n_frames_s, gt_s, det_s)
            m_motion = run_motion(n_frames_s, image_size, gt_s, det_s, model)

            results.append(dict(
                name=name, stride=stride, disp=disp,
                idf1_base=m_base["idf1"], idf1_motion=m_motion["idf1"],
                switches_base=m_base["id_switches"], switches_motion=m_motion["id_switches"],
            ))
            print(f"  stride={stride}  desloc_rel={disp:.3f}  "
                  f"IDF1 base={m_base['idf1']:.3f} motion={m_motion['idf1']:.3f} "
                  f"(delta={m_motion['idf1']-m_base['idf1']:+.3f})  "
                  f"switches base={m_base['id_switches']} motion={m_motion['id_switches']}")
        print()
    return results


def plot_results(results):
    names = [n for n, _ in SEQUENCES]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), sharey=True)

    for ax, name in zip(axes, names):
        rows = sorted([r for r in results if r["name"] == name], key=lambda r: r["stride"])
        disp = [r["disp"] for r in rows]
        base = [r["idf1_base"] for r in rows]
        motion = [r["idf1_motion"] for r in rows]

        ax.plot(disp, base, "o-", color="tab:gray", label="baseline ingênuo")
        ax.plot(disp, motion, "s-", color="tab:blue", label="MotionGRU")
        ax.axvline(REF_EASY, color="tab:green", linestyle=":", linewidth=1, label="limiar fácil (sweep P0)")
        ax.axvline(REF_BREAK, color="tab:red", linestyle=":", linewidth=1, label="limiar de quebra (sweep P0)")
        ax.set_xlabel("deslocamento relativo\n(fração da largura da caixa)")
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("IDF1")
    axes[0].legend(fontsize=7, loc="lower left")
    plt.suptitle("Subamostrando o MOT17 real pro regime 'difícil' do sweep da Parte 0\n"
                  "(mesmo checkpoint da Parte 2, sem retreinar)", fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / "06_subamostragem_frame_rate.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")


if __name__ == "__main__":
    results = main()
    plot_results(results)
