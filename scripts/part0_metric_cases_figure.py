"""Parte 0, item 3 — figura quadro-a-quadro dos 3 casos de métrica
construídos à mão (ver tests/test_metrics.py para a implementação e os
valores exatos derivados analiticamente).

Os cenários dos testes são abstratos (duas caixas em posições fixas, só os
RÓTULOS previstos mudam ao longo do tempo) — não são vídeo de verdade. Esta
figura desenha essas cenas quadro a quadro (mesmo espírito da figura de
oclusão da Parte 0 item 1), pra tornar visível o que "troca de identidade"
e "track partida" realmente significam, em vez de só números.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import compute_mot_metrics

OUT_DIR = ROOT / "outputs" / "part0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [0, 3, 4, 5, 6, 7, 9]

# posições fixas (normalizadas 0-1) de cada "pessoa real" na cena de brinquedo
POS_A = dict(x=0.06, y=0.25, w=0.34, h=0.5, label="pessoa A")
POS_B = dict(x=0.60, y=0.25, w=0.34, h=0.5, label="pessoa B")
POS_SINGLE = dict(x=0.33, y=0.25, w=0.34, h=0.5, label="pessoa real (única)")

ID_COLOR = {10: "tab:blue", 20: "tab:orange", 11: "tab:red", 1: "tab:blue", 2: "tab:orange"}


def case_a_state(frame):
    # pred == gt: ID 1 sempre em A, ID 2 sempre em B.
    return {1: POS_A, 2: POS_B}


def case_b_state(frame):
    # swap no quadro 5: antes, ID10->A e ID20->B; a partir do quadro 5, invertido.
    if frame < 5:
        return {10: POS_A, 20: POS_B}
    return {10: POS_B, 20: POS_A}


def case_c_state(frame):
    # ID10 nos quadros 0-4, buraco real nos quadros 5-6, ID11 nos quadros 7-9.
    if frame <= 4:
        return {10: POS_SINGLE}
    if frame <= 6:
        return {}
    return {11: POS_SINGLE}


CASES = [
    dict(
        key="a", fn=case_a_state,
        title="Caso (a): predição = verdade (idêntica)",
        gt_positions=[POS_A, POS_B],
    ),
    dict(
        key="b", fn=case_b_state,
        title="Caso (b): duas identidades trocadas a partir do quadro 5",
        gt_positions=[POS_A, POS_B],
    ),
    dict(
        key="c", fn=case_c_state,
        title="Caso (c): track partida no meio (buraco real de detecção)",
        gt_positions=[POS_SINGLE],
    ),
]


# --- cálculo real das métricas, reaproveitando as mesmas construções de tests/test_metrics.py ---
sys.path.insert(0, str(ROOT / "tests"))
import test_metrics as tm  # noqa: E402


def compute_all_metrics():
    gt_ab = tm.make_track(1, tm.BOX_A, range(10)) + tm.make_track(2, tm.BOX_B, range(10))

    pred_a = tm.make_track(1, tm.BOX_A, range(10)) + tm.make_track(2, tm.BOX_B, range(10))
    m_a = compute_mot_metrics(gt_ab, pred_a)

    pred_b = (
        tm.make_track(10, tm.BOX_A, range(0, 5)) + tm.make_track(10, tm.BOX_B, range(5, 10))
        + tm.make_track(20, tm.BOX_B, range(0, 5)) + tm.make_track(20, tm.BOX_A, range(5, 10))
    )
    m_b = compute_mot_metrics(gt_ab, pred_b)

    gt_single = tm.make_track(1, tm.BOX_A, range(10))
    pred_c = tm.make_track(10, tm.BOX_A, range(0, 5)) + tm.make_track(11, tm.BOX_A, range(7, 10))
    m_c = compute_mot_metrics(gt_single, pred_c)

    return m_a, m_b, m_c


def draw_frame(ax, frame, state, gt_positions):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("lightgray")

    # posição real (tracejado cinza) -- sempre presente, independente de ter sido detectada
    for pos in gt_positions:
        ax.add_patch(
            plt.Rectangle((pos["x"], pos["y"]), pos["w"], pos["h"], fill=False,
                          edgecolor="gray", linestyle="--", linewidth=1.3)
        )
        ax.text(pos["x"] + pos["w"] / 2, pos["y"] - 0.07, pos["label"], ha="center",
                fontsize=9, color="gray", style="italic")

    # caixa prevista (sólida, colorida por ID) -- só existe se o id foi "detectado" neste quadro
    if not state:
        ax.text(0.5, 0.5, "SEM\nDETECÇÃO", ha="center", va="center", fontsize=11,
                color="firebrick", fontweight="bold")
    else:
        for pred_id, pos in state.items():
            color = ID_COLOR.get(pred_id, "black")
            inset = 0.025
            ax.add_patch(
                plt.Rectangle((pos["x"] + inset, pos["y"] + inset), pos["w"] - 2 * inset, pos["h"] - 2 * inset,
                              fill=True, facecolor=color, alpha=0.35, edgecolor=color, linewidth=2)
            )
            ax.text(pos["x"] + pos["w"] / 2, pos["y"] + pos["h"] / 2, f"ID {pred_id}",
                    ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    ax.set_title(f"quadro {frame}", fontsize=11)


def build_case_figure(case, m, out_name):
    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 3.6))

    for col, frame in enumerate(COLUMNS):
        ax = axes[col]
        state = case["fn"](frame)
        draw_frame(ax, frame, state, case["gt_positions"])
        if frame == 5:  # marca visualmente onde a troca/buraco começa
            ax.spines["left"].set_color("black")
            ax.spines["left"].set_linewidth(2.5)

    fig.suptitle(
        f"{case['title']}\n"
        f"IDF1={m['idf1']:.3f}    ID switches={m['id_switches']}    fragmentações={m['fragmentations']}",
        fontsize=13,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.86])

    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")


def build_figures():
    m_a, m_b, m_c = compute_all_metrics()
    metrics_by_case = {"a": m_a, "b": m_b, "c": m_c}
    out_names = {"a": "04a_metric_case_a.png", "b": "04b_metric_case_b.png", "c": "04c_metric_case_c.png"}

    for case in CASES:
        build_case_figure(case, metrics_by_case[case["key"]], out_names[case["key"]])


if __name__ == "__main__":
    build_figures()
