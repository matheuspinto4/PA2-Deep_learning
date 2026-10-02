"""Parte 5 — Teste de estresse: qualidade do detector (escolhido em vez de
queda de frame rate). Feito SEM RETREINAR, em cima do modelo final da
Parte 2 (checkpoints/motion_gru_mot17.pt).

Reaproveita `pa2.synthetic.simulate_detections` (construído na Parte 0 —
o próprio enunciado avisa que esse é "exatamente um experimento da Parte
5") para degradar de propósito as caixas verdadeiras do MOT17 real: em vez
de usar o det.txt público (que tem um nível de ruído fixo, não
controlável), geramos as detecções a partir do gt.txt + visibilidade, com
ruído que NÓS controlamos, em 3 intensidades.

Pergunta do enunciado: o modelo temporal ABSORVE ou AMPLIFICA a falha do
detector? Medimos reportando mAP (qualidade de detecção, igual pras duas
fontes já que ambos tracker consomem a MESMA detecção degradada) e IDF1
(baseline ingênuo vs. MotionGRU) juntos, nas mesmas 3 intensidades.
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
from pa2.baseline_tracker import NaiveIoUTracker
from pa2.detection_metrics import average_precision
from pa2.models.motion_rnn import MotionGRU
from pa2.motion_tracker import MotionRNNTracker
from pa2.mot17 import MOT17Sequence, gt_to_tracks, gt_visibility_map
from pa2.synthetic import simulate_detections

OUT_DIR = ROOT / "outputs" / "part5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = ROOT / "checkpoints" / "motion_gru_mot17.pt"
HIDDEN_SIZE = 64
MAX_AGE = 15  # já usamos o valor corrigido na Parte 4, não o 5 original

SEQUENCES = [("MOT17-09", "val", 10.1), ("MOT17-11", "test", 10.5), ("MOT17-02", "train", 31.0), ("MOT17-04", "diagnostic", 45.3)]
SEEDS = [0, 1, 2]

INTENSITIES = {
    "leve": dict(drop_prob=0.05, occlusion_sensitivity=1.0, coord_noise_std=3.0, size_noise_std=2.0, fp_rate=0.1),
    "média": dict(drop_prob=0.15, occlusion_sensitivity=2.0, coord_noise_std=8.0, size_noise_std=5.0, fp_rate=0.3),
    "severa": dict(drop_prob=0.30, occlusion_sensitivity=3.0, coord_noise_std=15.0, size_noise_std=10.0, fp_rate=0.6),
}
INTENSITY_ORDER = ["limpo", "leve", "média", "severa"]


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
    tracker = NaiveIoUTracker(iou_threshold=0.3, max_age=MAX_AGE, match_method="hungarian")
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

    tracker = MotionRNNTracker(model, iou_threshold=0.3, max_age=MAX_AGE, match_method="hungarian")
    pred_norm = tracker.run(det_by_frame, score_by_frame)
    pred_tracks = [(f, i, x * W, y * H, w * W, h * H) for f, i, x, y, w, h in pred_norm]
    return compute_mot_metrics(gt, pred_tracks)


def main():
    model = MotionGRU(hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()

    results = []  # linhas: seq, intensity, seed, map, idf1_base, idf1_motion
    for name, split, dens in SEQUENCES:
        seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
        gt_rows = seq.load_gt()
        gt = gt_to_tracks(gt_rows)
        vis_map = gt_visibility_map(gt_rows)
        image_size = (seq.info["imWidth"], seq.info["imHeight"])

        print(f"=== {name} (densidade={dens}) ===")
        for intensity_name in INTENSITY_ORDER:
            maps, idf1_bases, idf1_motions = [], [], []
            for seed in SEEDS:
                if intensity_name == "limpo":
                    det_list = [(f, x, y, w, h, 1.0) for f, i, x, y, w, h in gt]  # detector "perfeito" -- sanity check
                else:
                    params = INTENSITIES[intensity_name]
                    det_list = simulate_detections(gt, visibilities=vis_map, image_size=image_size, seed=seed, **params)

                ap = average_precision(gt, det_list, iou_threshold=0.5)["ap"]
                m_base = run_baseline(seq, gt, det_list)
                m_motion = run_motion(seq, gt, det_list, model)

                maps.append(ap)
                idf1_bases.append(m_base["idf1"])
                idf1_motions.append(m_motion["idf1"])

            results.append(dict(
                name=name, dens=dens, intensity=intensity_name,
                map_mean=np.mean(maps), map_std=np.std(maps),
                idf1_base_mean=np.mean(idf1_bases), idf1_base_std=np.std(idf1_bases),
                idf1_motion_mean=np.mean(idf1_motions), idf1_motion_std=np.std(idf1_motions),
            ))
            r = results[-1]
            print(f"  {intensity_name:7s}: mAP={r['map_mean']:.3f}±{r['map_std']:.3f}  "
                  f"IDF1 base={r['idf1_base_mean']:.3f}±{r['idf1_base_std']:.3f}  "
                  f"IDF1 motion={r['idf1_motion_mean']:.3f}±{r['idf1_motion_std']:.3f}")
    return results


def plot_results(results):
    names = [n for n, _, _ in SEQUENCES]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), sharey=True)
    x = np.arange(len(INTENSITY_ORDER))

    for ax, name in zip(axes, names):
        rows = {r["intensity"]: r for r in results if r["name"] == name}
        rows = [rows[i] for i in INTENSITY_ORDER]

        ax.errorbar(x, [r["map_mean"] for r in rows], yerr=[r["map_std"] for r in rows],
                    marker="o", color="tab:green", label="mAP (detecção)", capsize=3)
        ax.errorbar(x, [r["idf1_base_mean"] for r in rows], yerr=[r["idf1_base_std"] for r in rows],
                    marker="s", color="tab:gray", label="IDF1 baseline ingênuo", capsize=3)
        ax.errorbar(x, [r["idf1_motion_mean"] for r in rows], yerr=[r["idf1_motion_std"] for r in rows],
                    marker="^", color="tab:blue", label="IDF1 MotionGRU", capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(INTENSITY_ORDER)
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8, loc="lower left")
    plt.suptitle("Parte 5 — degradação do detector em 3 intensidades (sem retreinar, mesmo checkpoint)", fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "01_degradacao_detector.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n[ok] figura salva em {out_path}")


def plot_amplification(results):
    """absorve/amplifica: quanto do RECUO de mAP (relativo ao 'limpo') vira
    recuo de IDF1, pra cada tracker. Se a razão do MotionGRU for MENOR que
    a do baseline, ele está absorvendo a falha; se for MAIOR, amplificando."""
    names = [n for n, _, _ in SEQUENCES]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.35
    x = np.arange(len(names))

    ratios_base, ratios_motion = [], []
    for name in names:
        rows = {r["intensity"]: r for r in results if r["name"] == name}
        clean = rows["limpo"]
        severe = rows["severa"]
        d_map = clean["map_mean"] - severe["map_mean"]
        d_idf1_base = clean["idf1_base_mean"] - severe["idf1_base_mean"]
        d_idf1_motion = clean["idf1_motion_mean"] - severe["idf1_motion_mean"]
        ratios_base.append(d_idf1_base / d_map if d_map > 1e-6 else float("nan"))
        ratios_motion.append(d_idf1_motion / d_map if d_map > 1e-6 else float("nan"))

    ax.bar(x - width / 2, ratios_base, width, label="baseline ingênuo", color="tab:gray")
    ax.bar(x + width / 2, ratios_motion, width, label="MotionGRU", color="tab:blue")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, label="recuo de IDF1 = recuo de mAP (neutro)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("(queda de IDF1) / (queda de mAP), limpo -> severa")
    ax.set_title("Absorve (< 1) ou amplifica (> 1) a falha do detector?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "02_absorve_ou_amplifica.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")


if __name__ == "__main__":
    results = main()
    plot_results(results)
    plot_amplification(results)
