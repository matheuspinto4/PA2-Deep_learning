"""Parte 1 — figuras-resumo pra apresentação (gráficos de barra), além das
já geradas por part1_baseline.py e part1_torchvision_detector.py:

  1. AP de DPM/FRCNN/SDP por sequência -- visualiza a escolha do detector
     público padrão (item 1).
  2. AP torchvision vs. SDP por sequência -- visualiza quando o detector
     genérico do torchvision ganha, empata ou perde do detector nativo do
     dataset (item 1, segunda fonte). MOT17-11 ainda pendente (sem imagem
     baixada o suficiente ainda); o gráfico mostra isso explicitamente.
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

from pa2.detection_metrics import average_precision
from pa2.mot17 import MOT17Sequence, gt_to_tracks

OUT_DIR = ROOT / "outputs" / "part1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCES = [
    ("MOT17-09", "val", 10.1),
    ("MOT17-11", "test", 10.5),
    ("MOT17-02", "train", 31.0),
    ("MOT17-04", "diagnostic", 45.3),
]
VARIANTS = ["DPM", "FRCNN", "SDP"]

# resultados da comparação torchvision (scripts/part1_torchvision_detector.py),
# 40 quadros amostrados por sequência -- ver outputs/part1/02_torchvision_vs_publico_*.png
TORCHVISION_AP = {
    "MOT17-02": 0.398,
    "MOT17-09": 0.704,
    "MOT17-04": 0.553,
    "MOT17-11": 0.657,
}
SDP_AP_NA_AMOSTRA_TORCHVISION = {  # AP do SDP calculado nos MESMOS 40 quadros (maçã com maçã)
    "MOT17-02": 0.407,
    "MOT17-09": 0.656,
    "MOT17-04": 0.748,
    "MOT17-11": 0.747,
}


def plot_public_detector_comparison():
    ap_table = {v: [] for v in VARIANTS}
    names = []
    for name, split, _dens in SEQUENCES:
        seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
        gt = gt_to_tracks(seq.load_gt())
        names.append(name)
        for v in VARIANTS:
            det = seq.load_det(variant=v)
            r = average_precision(gt, det, iou_threshold=0.5)
            ap_table[v].append(r["ap"])

    x = np.arange(len(names))
    width = 0.25
    colors = {"DPM": "tab:gray", "FRCNN": "tab:cyan", "SDP": "tab:green"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, v in enumerate(VARIANTS):
        bars = ax.bar(x + (i - 1) * width, ap_table[v], width, label=v, color=colors[v])
        if v == "SDP":
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.8)

    means = {v: np.mean(ap_table[v]) for v in VARIANTS}
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("AP (IoU ≥ 0.5)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Escolha do detector público padrão: SDP vence nas 4 sequências\n"
        f"(AP médio: DPM={means['DPM']:.3f}  FRCNN={means['FRCNN']:.3f}  SDP={means['SDP']:.3f})"
    )
    ax.legend(title="detector público")
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out_path = OUT_DIR / "03_detector_publico_comparacao.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")


def plot_torchvision_vs_sdp_summary():
    names = [n for n, _, _ in SEQUENCES]
    x = np.arange(len(names))
    width = 0.32

    tv_vals, tv_missing = [], []
    sdp_vals = []
    for n in names:
        if n in TORCHVISION_AP:
            tv_vals.append(TORCHVISION_AP[n])
            tv_missing.append(False)
            sdp_vals.append(SDP_AP_NA_AMOSTRA_TORCHVISION[n])
        else:
            tv_vals.append(0.0)
            tv_missing.append(True)
            sdp_vals.append(0.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars_sdp = ax.bar(x - width / 2, sdp_vals, width, label="SDP (público, fonte padrão)", color="tab:green")
    bars_tv = ax.bar(x + width / 2, tv_vals, width, label="torchvision Faster R-CNN (genérico)", color="tab:cyan")

    for i, missing in enumerate(tv_missing):
        if missing:
            ax.text(x[i], 0.05, "imagem\npendente", ha="center", va="bottom", fontsize=9,
                    color="firebrick", fontstyle="italic")
            bars_sdp[i].set_alpha(0.15)
            bars_tv[i].set_alpha(0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("AP (IoU ≥ 0.5), nos mesmos 40 quadros amostrados")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Detector genérico (torchvision) vs. detector nativo do dataset (SDP)\n"
        "o resultado depende da cena: empata, perde e até ganha"
    )
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out_path = OUT_DIR / "04_torchvision_vs_sdp_resumo.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")


if __name__ == "__main__":
    plot_public_detector_comparison()
    plot_torchvision_vs_sdp_summary()
