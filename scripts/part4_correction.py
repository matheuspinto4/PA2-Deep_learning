"""Parte 4 — a correção.

Diagnóstico escolhido (dos dois experimentos de horizonte de memória):
max_age=5 é pequeno demais pro MOT17 real -- 78% dos eventos de oclusão
reais duram mais que isso, e a sonda empírica mostrou a sobrevivência de
identidade despencar a 0 exatamente aí. A mudança sugerida: aumentar
max_age. Escolhemos max_age=15 (não 25): a sonda empírica mostrou que
25 mal ajuda mais que 15 (um segundo limite, de deriva da previsão em
free-running, satura por volta dali) e 15 já casa com a mediana real de
duração de oclusão do MOT17.

Antes/depois, MESMO checkpoint (motion_gru_mot17.pt), mesma fonte de
detecção (SDP), mesmas 4 sequências -- só o max_age do tracker muda.
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

from metrics import compute_mot_metrics
from pa2.models.motion_rnn import MotionGRU
from pa2.motion_tracker import MotionRNNTracker
from pa2.mot17 import MOT17Sequence, gt_to_tracks

OUT_DIR = ROOT / "outputs" / "part4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = ROOT / "checkpoints" / "motion_gru_mot17.pt"

SEQUENCES = [("MOT17-09", "val", 10.1), ("MOT17-11", "test", 10.5), ("MOT17-02", "train", 31.0), ("MOT17-04", "diagnostic", 45.3)]
DETECTOR = "SDP"
HIDDEN_SIZE = 64
MAX_AGE_BEFORE = 5
MAX_AGE_AFTER = 15


def dets_by_frame_raw(det_list):
    by_frame = {}
    for frame, x, y, w, h, score in det_list:
        by_frame.setdefault(int(frame), []).append((x, y, w, h, score))
    return by_frame


def run_motion(seq, gt, det_list, model, max_age):
    W, H = seq.info["imWidth"], seq.info["imHeight"]
    by_frame = dets_by_frame_raw(det_list)

    det_by_frame, score_by_frame = {}, {}
    for f, v in by_frame.items():
        det_by_frame[f] = np.array([[x / W, y / H, w / W, h / H] for x, y, w, h, s in v], dtype=np.float32)
        score_by_frame[f] = np.array([s for *_, s in v], dtype=np.float32)
    for f in range(1, seq.n_frames + 1):
        det_by_frame.setdefault(f, np.zeros((0, 4), dtype=np.float32))
        score_by_frame.setdefault(f, np.zeros(0, dtype=np.float32))

    tracker = MotionRNNTracker(model, iou_threshold=0.3, max_age=max_age, match_method="hungarian")
    pred_norm = tracker.run(det_by_frame, score_by_frame)
    pred_tracks = [(f, i, x * W, y * H, w * W, h * H) for f, i, x, y, w, h in pred_norm]
    return compute_mot_metrics(gt, pred_tracks)


def main():
    model = MotionGRU(hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()

    results = []
    for name, split, dens in SEQUENCES:
        seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
        gt = gt_to_tracks(seq.load_gt())
        det_list = seq.load_det(variant=DETECTOR)

        m_before = run_motion(seq, gt, det_list, model, MAX_AGE_BEFORE)
        m_after = run_motion(seq, gt, det_list, model, MAX_AGE_AFTER)
        results.append(dict(name=name, dens=dens, before=m_before, after=m_after))

        print(f"[{name}] max_age={MAX_AGE_BEFORE} -> {MAX_AGE_AFTER}:")
        print(f"    IDF1    {m_before['idf1']:.3f} -> {m_after['idf1']:.3f}  (delta={m_after['idf1']-m_before['idf1']:+.3f})")
        print(f"    switches {m_before['id_switches']:4d} -> {m_after['id_switches']:4d}")
        print(f"    frags    {m_before['fragmentations']:4d} -> {m_after['fragmentations']:4d}")
        print(f"    ratio    {m_before['ratio']:.2f} -> {m_after['ratio']:.2f}")
    return results


def plot_before_after(results):
    results = sorted(results, key=lambda r: r["dens"])
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [
        ("idf1", "IDF1 (maior é melhor)", axes[0, 0]),
        ("id_switches", "ID switches (menor é melhor)", axes[0, 1]),
        ("fragmentations", "Fragmentações (menor é melhor)", axes[1, 0]),
        ("ratio", "Razão ids previstas/verdadeiras (perto de 1 é melhor)", axes[1, 1]),
    ]
    for key, title, ax in specs:
        before = [r["before"][key] for r in results]
        after = [r["after"][key] for r in results]
        ax.bar(x - width / 2, before, width, label=f"antes (max_age={MAX_AGE_BEFORE})", color="tab:gray")
        ax.bar(x + width / 2, after, width, label=f"depois (max_age={MAX_AGE_AFTER})", color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title, fontsize=10)
        if key == "ratio":
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)

    plt.suptitle(f"Parte 4 -- correção: max_age {MAX_AGE_BEFORE} -> {MAX_AGE_AFTER} (mesmo checkpoint, mesmas sequências)", fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "03_correcao_antes_depois.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n[ok] figura salva em {out_path}")


if __name__ == "__main__":
    results = main()
    plot_before_after(results)
