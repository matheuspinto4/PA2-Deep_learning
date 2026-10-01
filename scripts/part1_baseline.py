"""Parte 1 — baseline por quadro no MOT17 real.

1. Compara as 3 variantes de detecção pública (DPM/FRCNN/SDP) via AP contra
   o gt, nas 4 sequências baixadas, e escolhe a de melhor qualidade média
   como fonte padrão do resto do PA.
2. Roda a associação ingênua (pa2.baseline_tracker, a mesma da Parte 0 item 4)
   usando essa fonte padrão.
3. Avalia como trajetórias: IDF1, ID switches, fragmentações, erro de
   contagem de identidades (metrics.py).
4. Produz o gráfico obrigatório (descolamento): em cima mAP e IDF1 por
   sequência; embaixo razão de identidades previstas/verdadeiras e ID
   switches por identidade verdadeira. Sequências ordenadas por densidade
   (o eixo de dificuldade escolhido — ver justificativa no README/AI_LOG).
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

from metrics import compute_mot_metrics
from pa2.baseline_tracker import NaiveIoUTracker
from pa2.detection_metrics import average_precision
from pa2.mot17 import MOT17Sequence, gt_to_tracks

OUT_DIR = ROOT / "outputs" / "part1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCES = [
    # (nome, split, câmera, densidade média já medida na Parte 1 prep / AI_LOG)
    ("MOT17-02", "train", "parada", 31.0),
    ("MOT17-09", "val", "parada", 10.1),
    ("MOT17-04", "diagnostic", "parada", 45.3),
    ("MOT17-11", "test", "em movimento (held-out)", 10.5),
]
VARIANTS = ["DPM", "FRCNN", "SDP"]


def detections_by_frame(det_list):
    by_frame: dict[int, list] = {}
    for frame, x, y, w, h, score in det_list:
        by_frame.setdefault(int(frame), []).append((x, y, w, h))
    return {f: np.array(v) for f, v in by_frame.items()}


def choose_default_detector():
    print("=" * 70)
    print("Item 1 — comparando DPM / FRCNN / SDP via AP contra o gt (IoU>=0.5)")
    print("=" * 70)
    ap_table = {v: [] for v in VARIANTS}
    for name, split, _cam, _dens in SEQUENCES:
        seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
        gt = gt_to_tracks(seq.load_gt())
        row = []
        for v in VARIANTS:
            det = seq.load_det(variant=v)
            r = average_precision(gt, det, iou_threshold=0.5)
            ap_table[v].append(r["ap"])
            row.append(f"{v}={r['ap']:.3f}")
        print(f"  {name:10s} (dens={_dens:5.1f}): " + "  ".join(row))

    means = {v: float(np.mean(aps)) for v, aps in ap_table.items()}
    print("\n  AP médio across sequências:", {v: round(m, 3) for v, m in means.items()})
    best = max(means, key=means.get)
    print(f"  -> detector padrão escolhido: {best} (maior AP médio nas 4 sequências)\n")
    return best, means


def run_part1(default_detector: str):
    results = []
    for name, split, cam, dens in SEQUENCES:
        seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
        gt_rows = seq.load_gt()
        gt = gt_to_tracks(gt_rows)
        det = seq.load_det(variant=default_detector)

        ap_r = average_precision(gt, det, iou_threshold=0.5)

        det_by_frame = detections_by_frame(det)
        for f in range(1, seq.n_frames + 1):
            det_by_frame.setdefault(f, np.zeros((0, 4)))

        tracker = NaiveIoUTracker(iou_threshold=0.3, max_age=5, match_method="hungarian")
        pred = tracker.run(det_by_frame)
        pred_tracks = [(f, i, x, y, w, h) for f, i, x, y, w, h in pred]

        m = compute_mot_metrics(gt, pred_tracks)
        switches_per_id = m["id_switches"] / m["n_gt_ids"] if m["n_gt_ids"] else 0.0

        results.append(
            dict(
                name=name, cam=cam, densidade=dens, ap=ap_r["ap"],
                idf1=m["idf1"], idp=m["idp"], idr=m["idr"],
                id_switches=m["id_switches"], fragmentations=m["fragmentations"],
                n_gt_ids=m["n_gt_ids"], n_pred_ids=m["n_pred_ids"], ratio=m["ratio"],
                switches_per_id=switches_per_id,
            )
        )
        print(
            f"[{name}] cam={cam:24s} dens={dens:5.1f}  AP={ap_r['ap']:.3f}  IDF1={m['idf1']:.3f}  "
            f"switches={m['id_switches']:3d}  frags={m['fragmentations']:3d}  "
            f"ids(pred/gt)={m['n_pred_ids']}/{m['n_gt_ids']} (ratio={m['ratio']:.2f})"
        )
    return results


def plot_descolamento(results, detector_name):
    results = sorted(results, key=lambda r: r["densidade"])
    names = [r["name"] for r in results]
    cams = [r["cam"] for r in results]
    x = np.arange(len(results))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(x, [r["ap"] for r in results], "o-", color="tab:green", label="mAP (detecção, por quadro)")
    ax.plot(x, [r["idf1"] for r in results], "s-", color="tab:blue", label="IDF1 (identidade, trajetória)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"Descolamento entre qualidade de detecção e qualidade de identidade (detector={detector_name})")

    ax2 = axes[1]
    ax2.plot(x, [r["ratio"] for r in results], "o-", color="tab:purple", label="ids previstas / ids verdadeiras")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("razão de identidades", color="tab:purple")
    ax2b = ax2.twinx()
    ax2b.plot(x, [r["switches_per_id"] for r in results], "s--", color="tab:red", label="ID switches / identidade verdadeira")
    ax2b.set_ylabel("switches por identidade", color="tab:red")

    labels = [f"{n}\n({c})\ndens={d:.0f}" for n, c, d in zip(names, cams, [r["densidade"] for r in results])]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_xlabel("sequência, ordenada por densidade (eixo de dificuldade escolhido)")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_descolamento.png", dpi=130)
    plt.close(fig)
    print(f"\n[ok] gráfico salvo em {OUT_DIR/'01_descolamento.png'}")


if __name__ == "__main__":
    default_detector, ap_means = choose_default_detector()
    results = run_part1(default_detector)
    plot_descolamento(results, default_detector)
