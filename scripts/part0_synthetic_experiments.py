"""Parte 0 — gera todos os artefatos exigidos a partir do ambiente sintético:

  1. Figura de oclusão verificável (uma trajetória que some por N quadros e volta).
  2. Validação visual do simulador de detector (gt vs detecções estragadas).
  3. Reexecução programática dos 3 casos de métrica construídos à mão (o
     teste "de verdade" está em tests/test_metrics.py; aqui só imprimimos os
     números para referência rápida / apresentação).
  4. Baseline no piso fácil: poucas elipses, lentas, sem oclusão -> IDF1 ~ 1.
  5. Sweep dos parâmetros do gerador (nº objetos, velocidade, duração de
     oclusão) mostrando onde o baseline da Parte 1 começa a quebrar.

Saídas em outputs/part0/. Rodar com: python scripts/part0_synthetic_experiments.py
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
from pa2.synthetic import SyntheticVideoGenerator, simulate_detections

OUT_DIR = ROOT / "outputs" / "part0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def detections_by_frame_from_sim(sim_dets):
    by_frame: dict[int, list] = {}
    for frame, x, y, w, h, score in sim_dets:
        by_frame.setdefault(frame, []).append((x, y, w, h))
    return {f: np.array(v) if v else np.zeros((0, 4)) for f, v in by_frame.items()}


def run_baseline(gt, visibilities, image_size, detector_kwargs, tracker_kwargs, seed=0):
    sim_dets = simulate_detections(gt, visibilities=visibilities, image_size=image_size, seed=seed, **detector_kwargs)
    det_by_frame = detections_by_frame_from_sim(sim_dets)
    all_frames = sorted(set(d[0] for d in gt))
    for f in all_frames:
        det_by_frame.setdefault(f, np.zeros((0, 4)))

    tracker = NaiveIoUTracker(**tracker_kwargs)
    pred = tracker.run(det_by_frame)
    pred_xyhw = [(f, i, x, y, w, h) for f, i, x, y, w, h in pred]
    m = compute_mot_metrics(gt, pred_xyhw)
    return m, sim_dets


# ---------------------------------------------------------------------------
# 1) Figura de oclusão verificável
# ---------------------------------------------------------------------------
def demo_occlusion_figure():
    gen = SyntheticVideoGenerator(num_objects=8, speed=2.0, occlusion_duration=14, seed=42, n_frames=50)
    frames, gt, vis, meta = gen.generate()

    occ_id = meta["occluded_id"]
    t_mid = meta["t_mid"]
    w0, w1 = meta["window"]
    print(f"[oclusao] occluded_id={occ_id} occluder_id={meta['occluder_id']} janela={meta['window']} t_mid={t_mid}")

    # curva de visibilidade do objeto ocluído ao longo do tempo
    n_frames = frames.shape[0]
    vis_curve = [vis.get((t, occ_id), 1.0) for t in range(n_frames)]

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(2, 7, height_ratios=[1.0, 0.8])

    show_frames = np.linspace(max(0, t_mid - 18), min(n_frames - 1, t_mid + 18), 7).astype(int)
    for j, t in enumerate(show_frames):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(frames[t], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"t={t}" + ("  (ocluso)" if w0 <= t <= w1 else ""), fontsize=9)
        ax.axis("off")
        box = next((d for d in gt if d[0] == t and d[1] == occ_id), None)
        if box is not None:
            _, _, x, y, w, h = box
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1.5)
            ax.add_patch(rect)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(range(n_frames), vis_curve, color="tab:red", linewidth=2)
    ax2.axvspan(w0, w1, color="gray", alpha=0.3, label=f"janela roteirizada ({w1 - w0} quadros)")
    ax2.set_xlabel("quadro")
    ax2.set_ylabel(f"visibilidade do objeto {occ_id}")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.set_title("Visibilidade cai a ~0 durante a janela roteirizada (cinza); outras quedas são oclusões incidentais com outros objetos")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_occlusion_demo.png", dpi=130)
    plt.close(fig)
    print(f"[ok] figura salva em {OUT_DIR/'01_occlusion_demo.png'}")

    min_vis_in_window = min(vis_curve[w0 : w1 + 1]) if w1 >= w0 else 1.0
    assert min_vis_in_window < 0.05, f"janela deveria ter oclusao quase total, mas vis min={min_vis_in_window}"
    print(f"[ok] requisito verificavel: visibilidade minima na janela = {min_vis_in_window:.4f} (<0.05)")


# ---------------------------------------------------------------------------
# 2) Validação visual do simulador de detector
# ---------------------------------------------------------------------------
def demo_detector_sim():
    gen = SyntheticVideoGenerator(num_objects=6, speed=1.5, occlusion_duration=0, seed=7, n_frames=30)
    frames, gt, vis, _ = gen.generate()

    sim_dets = simulate_detections(
        gt, visibilities=vis, drop_prob=0.15, coord_noise_std=2.5, size_noise_std=1.5, fp_rate=0.5, seed=1
    )

    t = 10
    gt_t = [d for d in gt if d[0] == t]
    det_t = [d for d in sim_dets if d[0] == t]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, title in zip(axes, ["Ground truth", "Detecções simuladas (ruidosas)"]):
        ax.imshow(frames[t], cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    for _, oid, x, y, w, h in gt_t:
        axes[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.5))
        axes[0].text(x, y - 2, str(oid), color="lime", fontsize=8)

    for _, x, y, w, h, score in det_t:
        axes[1].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="orange", linewidth=1.5))
        axes[1].text(x, y - 2, f"{score:.2f}", color="orange", fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_detector_sim_demo.png", dpi=130)
    plt.close(fig)

    n_gt_total = len(gt)
    n_det_total = len(sim_dets)
    print(f"[ok] simulador de detector: {n_gt_total} caixas gt -> {n_det_total} deteccoes simuladas "
          f"(figura em {OUT_DIR/'02_detector_sim_demo.png'})")


# ---------------------------------------------------------------------------
# 3) Casos de métrica (resumo em texto; os testes de verdade estão em tests/)
# ---------------------------------------------------------------------------
def print_metric_cases_summary():
    sys.path.insert(0, str(ROOT / "tests"))
    import test_metrics as tm

    gt_simple = tm.make_track(1, tm.BOX_A, range(10)) + tm.make_track(2, tm.BOX_B, range(10))
    pred_identical = tm.make_track(1, tm.BOX_A, range(10)) + tm.make_track(2, tm.BOX_B, range(10))
    m_a = compute_mot_metrics(gt_simple, pred_identical)

    pred_swap = (
        tm.make_track(10, tm.BOX_A, range(0, 5)) + tm.make_track(10, tm.BOX_B, range(5, 10))
        + tm.make_track(20, tm.BOX_B, range(0, 5)) + tm.make_track(20, tm.BOX_A, range(5, 10))
    )
    m_b = compute_mot_metrics(gt_simple, pred_swap)

    gt_single = tm.make_track(1, tm.BOX_A, range(10))
    pred_split = tm.make_track(10, tm.BOX_A, range(0, 5)) + tm.make_track(11, tm.BOX_A, range(7, 10))
    m_c = compute_mot_metrics(gt_single, pred_split)

    print("\n[Parte 0.3] Casos de metrica construidos a mao:")
    print(f"  (a) pred==gt           -> IDF1={m_a['idf1']:.3f} switches={m_a['id_switches']} frags={m_a['fragmentations']}")
    print(f"  (b) swap no quadro k=5  -> IDF1={m_b['idf1']:.3f} switches={m_b['id_switches']} frags={m_b['fragmentations']}")
    print(f"  (c) track partida no meio-> IDF1={m_c['idf1']:.3f} switches={m_c['id_switches']} frags={m_c['fragmentations']}")


# ---------------------------------------------------------------------------
# 4) Baseline no piso fácil
# ---------------------------------------------------------------------------
def demo_easy_floor():
    gen = SyntheticVideoGenerator(num_objects=4, speed=0.8, occlusion_duration=0, seed=123, n_frames=40)
    _, gt, vis, _ = gen.generate()

    # variante 1: deteccao quase perfeita -> sanity check de que a metrica/tracker
    # nao tem bug nenhum escondido (deve dar praticamente IDF1=1.0 exato).
    m_clean, _ = run_baseline(
        gt,
        vis,
        (gen.width, gen.height),
        detector_kwargs=dict(drop_prob=0.0, occlusion_sensitivity=0.0, coord_noise_std=0.5, size_noise_std=0.3, fp_rate=0.0),
        tracker_kwargs=dict(iou_threshold=0.3, max_age=5, match_method="hungarian"),
        seed=0,
    )
    print(f"\n[Parte 0.4] Piso facil, deteccao quase perfeita: IDF1={m_clean['idf1']:.3f} "
          f"switches={m_clean['id_switches']} frags={m_clean['fragmentations']}")
    assert m_clean["idf1"] > 0.95, f"IDF1 deveria ficar muito perto de 1 no piso facil, obtido {m_clean['idf1']}"

    # variante 2: mesmo cenario facil (poucas elipses, lentas, sem oclusao), mas com
    # o MESMO ruido de detector moderado usado no sweep de dificuldade -- mostra que
    # a associacao ingenua em si e robusta quando o problema de movimento é trivial,
    # e so degrada quando a dinamica (velocidade/oclusao/densidade) fica dificil.
    m_noisy, _ = run_baseline(
        gt,
        vis,
        (gen.width, gen.height),
        detector_kwargs=dict(drop_prob=0.05, occlusion_sensitivity=2.0, coord_noise_std=1.5, size_noise_std=1.0, fp_rate=0.2),
        tracker_kwargs=dict(iou_threshold=0.3, max_age=5, match_method="hungarian"),
        seed=0,
    )
    print(f"[Parte 0.4] Piso facil, deteccao com ruido moderado (mesmo do sweep): IDF1={m_noisy['idf1']:.3f} "
          f"switches={m_noisy['id_switches']} frags={m_noisy['fragmentations']}")
    assert m_noisy["idf1"] > 0.85, f"IDF1 deveria continuar alto mesmo com ruido moderado, obtido {m_noisy['idf1']}"
    print("[ok] IDF1 > 0.95 (deteccao limpa) e > 0.85 (deteccao com ruido moderado) no piso facil")
    return m_clean, m_noisy


# ---------------------------------------------------------------------------
# 5) Sweep de dificuldade -> onde o baseline quebra
# ---------------------------------------------------------------------------
def demo_difficulty_sweep():
    base_detector = dict(drop_prob=0.05, occlusion_sensitivity=2.0, coord_noise_std=1.5, size_noise_std=1.0, fp_rate=0.2)
    base_tracker = dict(iou_threshold=0.3, max_age=5, match_method="hungarian")

    results = []

    # eixo 1: numero de objetos
    for n_obj in [2, 4, 6, 8, 10, 13, 15]:
        gen = SyntheticVideoGenerator(num_objects=n_obj, speed=2.0, occlusion_duration=8, seed=10, n_frames=45)
        _, gt, vis, _ = gen.generate()
        m, _ = run_baseline(gt, vis, (gen.width, gen.height), base_detector, base_tracker, seed=10)
        results.append(dict(eixo="num_objects", valor=n_obj, **m))

    # eixo 2: velocidade
    for speed in [0.5, 1.5, 3.0, 5.0, 8.0, 12.0]:
        gen = SyntheticVideoGenerator(num_objects=8, speed=speed, occlusion_duration=8, seed=11, n_frames=45)
        _, gt, vis, _ = gen.generate()
        m, _ = run_baseline(gt, vis, (gen.width, gen.height), base_detector, base_tracker, seed=11)
        results.append(dict(eixo="speed", valor=speed, **m))

    # eixo 3: duracao da oclusao
    for occ_dur in [0, 4, 8, 12, 18, 24]:
        gen = SyntheticVideoGenerator(num_objects=8, speed=2.0, occlusion_duration=occ_dur, seed=12, n_frames=45)
        _, gt, vis, _ = gen.generate()
        m, _ = run_baseline(gt, vis, (gen.width, gen.height), base_detector, base_tracker, seed=12)
        results.append(dict(eixo="occlusion_duration", valor=occ_dur, **m))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    titles = dict(num_objects="Numero de objetos", speed="Velocidade tipica (px/quadro)", occlusion_duration="Duracao da oclusao (quadros)")
    for ax, eixo in zip(axes, ["num_objects", "speed", "occlusion_duration"]):
        xs = [r["valor"] for r in results if r["eixo"] == eixo]
        idf1s = [r["idf1"] for r in results if r["eixo"] == eixo]
        switches = [r["id_switches"] for r in results if r["eixo"] == eixo]
        ax.plot(xs, idf1s, "o-", color="tab:blue", label="IDF1")
        ax.set_xlabel(titles[eixo])
        ax.set_ylabel("IDF1", color="tab:blue")
        ax.set_ylim(0, 1.05)
        ax2 = ax.twinx()
        ax2.plot(xs, switches, "s--", color="tab:red", label="ID switches")
        ax2.set_ylabel("ID switches", color="tab:red")
        ax.set_title(titles[eixo])

    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_difficulty_sweep.png", dpi=130)
    plt.close(fig)
    print(f"\n[Parte 0.5] Sweep de dificuldade salvo em {OUT_DIR/'03_difficulty_sweep.png'}")
    for r in results:
        print(f"  {r['eixo']:20s} = {r['valor']:>6} -> IDF1={r['idf1']:.3f} switches={r['id_switches']:>3} "
              f"frags={r['fragmentations']:>3} idratio={r['ratio']:.2f}")


if __name__ == "__main__":
    demo_occlusion_figure()
    demo_detector_sim()
    print_metric_cases_summary()
    demo_easy_floor()
    demo_difficulty_sweep()
    print("\n[Parte 0] Todos os artefatos gerados em outputs/part0/.")
