"""Parte 1, item 1 — segunda fonte de detecção: detector pré-treinado do
torchvision (Faster R-CNN, classe person do COCO), comparado contra a fonte
pública padrão escolhida (SDP, ver scripts/part1_baseline.py).

Em CPU, uma imagem 1920x1080 leva ~5-6s no Faster R-CNN -- rodar nas
~600-1050 imagens de cada sequência levaria horas por sequência. Como o
enunciado só exige "usar as duas fontes" (a pública continua sendo a fonte
padrão do resto do PA, isso já foi decidido), amostramos N quadros
igualmente espaçados por sequência para uma comparação estatisticamente
razoável sem inviabilizar o tempo de execução. A comparação de AP é feita
SÓ nos quadros amostrados, para as duas fontes, para ser justa (maçã com
maçã).

Roda automaticamente só nas sequências cujas imagens já estão disponíveis
localmente (dá pra rodar de novo depois que o download terminar).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pa2.detection_metrics import average_precision
from pa2.mot17 import MOT17Sequence, gt_to_tracks
from pa2.torch_detector import detect_persons, load_person_detector

OUT_DIR = ROOT / "outputs" / "part1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCES = [
    ("MOT17-02", "train"),
    ("MOT17-09", "val"),
    ("MOT17-04", "diagnostic"),
    ("MOT17-11", "test"),
]
DEFAULT_PUBLIC_DETECTOR = "SDP"
N_SAMPLE_FRAMES = 40


def available_frames(seq: MOT17Sequence):
    img_dir = seq.seq_dir / seq.info["imDir"]
    have = sorted(int(p.stem) for p in img_dir.glob(f"*{seq.info['imExt']}"))
    return have


def sample_frames(have_frames, n):
    if len(have_frames) <= n:
        return have_frames
    idx = np.linspace(0, len(have_frames) - 1, n).astype(int)
    return sorted(set(have_frames[i] for i in idx))


def run_for_sequence(name, split, model, preprocess):
    seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
    have = available_frames(seq)
    if len(have) < 10:
        print(f"[{name}] só {len(have)} imagens disponíveis ainda, pulando (rode de novo depois).")
        return None

    frames = sample_frames(have, N_SAMPLE_FRAMES)
    print(f"[{name}] {len(have)}/{seq.n_frames} imagens no disco, amostrando {len(frames)} quadros...")

    torch_dets = []
    t0 = time.time()
    for f in frames:
        img = np.array(Image.open(seq.frame_path(f)).convert("RGB"))
        dets = detect_persons(model, preprocess, img, score_threshold=0.5)
        for x, y, w, h, score in dets:
            torch_dets.append((f, x, y, w, h, score))
    dt = time.time() - t0
    print(f"    inferência torchvision: {dt:.1f}s para {len(frames)} quadros ({dt/len(frames):.2f}s/quadro)")

    gt_all = gt_to_tracks(seq.load_gt())
    gt_sample = [d for d in gt_all if d[0] in frames]

    public_det_all = seq.load_det(variant=DEFAULT_PUBLIC_DETECTOR)
    public_sample = [d for d in public_det_all if int(d[0]) in frames]

    ap_torch = average_precision(gt_sample, torch_dets, iou_threshold=0.5)
    ap_public = average_precision(gt_sample, public_sample, iou_threshold=0.5)

    print(f"    AP (nos {len(frames)} quadros amostrados): torchvision/FasterRCNN={ap_torch['ap']:.3f}  "
          f"{DEFAULT_PUBLIC_DETECTOR}={ap_public['ap']:.3f}  (n_gt={ap_torch['n_gt']})")

    return dict(
        name=name, frames=frames, gt_sample=gt_sample, torch_dets=torch_dets, public_sample=public_sample,
        ap_torch=ap_torch["ap"], ap_public=ap_public["ap"], seq=seq,
    )


def plot_comparison(result):
    seq = result["seq"]
    frame = result["frames"][len(result["frames"]) // 2]
    img = np.array(Image.open(seq.frame_path(frame)).convert("RGB"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    titles = ["Ground truth", f"Público ({DEFAULT_PUBLIC_DETECTOR})", "torchvision Faster R-CNN"]
    sources = [
        [(x, y, w, h) for f, i, x, y, w, h in result["gt_sample"] if f == frame],
        [(x, y, w, h) for f, x, y, w, h, s in result["public_sample"] if f == frame],
        [(x, y, w, h) for f, x, y, w, h, s in result["torch_dets"] if f == frame],
    ]
    colors = ["lime", "orange", "cyan"]
    for ax, title, boxes, color in zip(axes, titles, sources, colors):
        ax.imshow(img)
        ax.set_title(f"{title} ({len(boxes)} caixas)")
        ax.axis("off")
        for x, y, w, h in boxes:
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.5))

    plt.suptitle(f"{result['name']}, quadro {frame} — AP@0.5 na amostra: {DEFAULT_PUBLIC_DETECTOR}="
                 f"{result['ap_public']:.3f}  torchvision={result['ap_torch']:.3f}")
    plt.tight_layout()
    out_path = OUT_DIR / f"02_torchvision_vs_publico_{result['name']}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"    figura salva em {out_path}")


if __name__ == "__main__":
    print("Carregando Faster R-CNN (torchvision, pré-treinado, classe person do COCO)...")
    model, preprocess = load_person_detector()

    all_results = []
    for name, split in SEQUENCES:
        r = run_for_sequence(name, split, model, preprocess)
        if r is not None:
            plot_comparison(r)
            all_results.append(r)

    if all_results:
        print("\nResumo:")
        for r in all_results:
            print(f"  {r['name']:10s} AP torchvision={r['ap_torch']:.3f}  AP {DEFAULT_PUBLIC_DETECTOR}={r['ap_public']:.3f}")
    else:
        print("\nNenhuma sequência com imagens suficientes ainda. Rode de novo depois que o download terminar.")
