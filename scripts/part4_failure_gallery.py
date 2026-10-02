"""Parte 4 — galeria de falhas: 3 trechos onde o modelo final erra feio.

Cada falha: a figura (tira de quadros com gt e previsão coloridos por
identidade, mais o mapa intermediário -- a caixa prevista PELA RECORRÊNCIA,
antes de casar com qualquer detecção) e um diagnóstico de 1-2 frases
conectado ao horizonte de memória medido na Parte 4.

Minerado automaticamente: roda o MotionRNNTracker (max_age=5, a config
"antes" da correção) numa sequência real, e reaproveita a lógica interna de
casamento quadro-a-quadro do metrics.py (_match_frame_with_continuity) pra
achar EXATAMENTE em que quadro e entre quais identidades cada troca
acontece -- não "olhando a olho".
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
from PIL import Image

from metrics import _frames_index, _match_frame_with_continuity
from pa2.models.motion_rnn import MotionGRU
from pa2.motion_tracker import MotionRNNTracker
from pa2.mot17 import MOT17Sequence, gt_to_tracks

OUT_DIR = ROOT / "outputs" / "part4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = ROOT / "checkpoints" / "motion_gru_mot17.pt"
HIDDEN_SIZE = 64
MAX_AGE = 5  # config "antes" da correção -- é essa que queremos diagnosticar


def dets_by_frame_raw(det_list):
    by_frame = {}
    for frame, x, y, w, h, score in det_list:
        by_frame.setdefault(int(frame), []).append((x, y, w, h, score))
    return by_frame


def run_tracker_with_predicted_boxes(seq, det_list, model):
    """Como pa2.motion_tracker.MotionRNNTracker, mas também devolve, por
    quadro, a caixa PREVISTA pela recorrência pra cada track viva (antes do
    casamento) -- o "mapa intermediário" que a Parte 4 pede pra mostrar."""
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
    tracker.reset()

    pred_tracks = []  # (frame, id, x, y, w, h) em pixel
    predicted_box_by_frame = defaultdict(dict)  # frame -> {track_id: (x,y,w,h) em pixel, ANTES do casamento}

    for frame in sorted(det_by_frame):
        for tid, tr in tracker.tracks.items():
            x, y, w, h = tr.pred_box.numpy()
            predicted_box_by_frame[frame][tid] = (x * W, y * H, w * W, h * H)
        out = tracker.step(det_by_frame[frame], score_by_frame[frame])
        for tid, x, y, w, h in out:
            pred_tracks.append((frame, tid, x * W, y * H, w * W, h * H))

    return pred_tracks, predicted_box_by_frame


def mine_switch_events(gt, pred_tracks, iou_threshold=0.3):
    """Reaproveita a mesma lógica de casamento do metrics.py pra achar,
    quadro a quadro, exatamente onde um ID switch acontece (o pred_id de um
    gt_id muda) -- com o "gap" (nº de quadros desde a última vez que esse
    gt_id esteve com observação) e se foi uma TROCA SIMULTÂNEA com outro
    gt_id (duas identidades trocando de rótulo ao mesmo tempo -- proximidade)
    ou uma troca isolada depois de um buraco (oclusão longa)."""
    gt_by_frame = _frames_index(gt)
    pred_by_frame = _frames_index(pred_tracks)
    all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    last_match, was_tracked, since_tracked = {}, defaultdict(bool), defaultdict(int)
    events = []

    for frame in all_frames:
        gt_boxes = gt_by_frame.get(frame, {})
        pred_boxes = pred_by_frame.get(frame, {})
        frame_matches = _match_frame_with_continuity(gt_boxes, pred_boxes, last_match, iou_threshold)

        for gt_id in gt_boxes:
            if gt_id in frame_matches:
                pred_now = frame_matches[gt_id]
                prev = last_match.get(gt_id)
                if prev is not None and prev != pred_now:
                    events.append(dict(
                        frame=frame, gt_id=gt_id, old_pred=prev, new_pred=pred_now,
                        gap=since_tracked[gt_id], was_fragmentation=not was_tracked[gt_id],
                    ))
                was_tracked[gt_id] = True
                last_match[gt_id] = pred_now
                since_tracked[gt_id] = 0
            else:
                was_tracked[gt_id] = False
                since_tracked[gt_id] += 1
                last_match.setdefault(gt_id, None)

    return events


def find_simultaneous_swaps(events):
    """Dois eventos no MESMO quadro onde gt_a passa a usar o pred que gt_b
    tinha, e vice-versa -- a assinatura de uma troca por proximidade."""
    by_frame = defaultdict(list)
    for e in events:
        by_frame[e["frame"]].append(e)
    swaps = []
    for frame, evs in by_frame.items():
        for i, e1 in enumerate(evs):
            for e2 in evs[i + 1 :]:
                if e1["new_pred"] == e2["old_pred"] and e2["new_pred"] == e1["old_pred"]:
                    swaps.append((e1, e2))
    return swaps


def render_clip(seq, gt, pred_tracks, predicted_box_by_frame, center_frame, gt_ids_of_interest, pred_ids_of_interest, out_name, title, half_window=4):
    W, H = seq.info["imWidth"], seq.info["imHeight"]
    frames = list(range(max(1, center_frame - half_window), min(seq.n_frames, center_frame + half_window) + 1))
    gt_by_frame = defaultdict(dict)
    for f, i, x, y, w, h in gt:
        if f in frames:
            gt_by_frame[f][i] = (x, y, w, h)
    pred_by_frame = defaultdict(dict)
    for f, i, x, y, w, h in pred_tracks:
        if f in frames:
            pred_by_frame[f][i] = (x, y, w, h)

    gt_colors = {gid: c for gid, c in zip(gt_ids_of_interest, ["lime", "cyan", "yellow"])}

    # recorte (zoom) ao redor dos objetos de interesse -- pega a união das
    # caixas gt deles em TODOS os quadros da janela, com uma margem, pra não
    # mostrar a cena inteira (ilegível em sequências densas tipo MOT17-02)
    xs0, ys0, xs1, ys1 = [], [], [], []
    for f in frames:
        for gid in gt_ids_of_interest:
            if gid in gt_by_frame.get(f, {}):
                x, y, w, h = gt_by_frame[f][gid]
                xs0.append(x); ys0.append(y); xs1.append(x + w); ys1.append(y + h)
    if not xs0:
        print(f"[aviso] nenhuma caixa gt encontrada pros ids {gt_ids_of_interest} na janela, pulando {out_name}")
        return
    margin = 60
    crop_x0 = max(0, min(xs0) - margin)
    crop_y0 = max(0, min(ys0) - margin)
    crop_x1 = min(W, max(xs1) + margin)
    crop_y1 = min(H, max(ys1) + margin)

    def pred_ids_relevant(f):
        """Só desenha as caixas previstas cujo ID é EXATAMENTE um dos
        envolvidos no evento minerado (não qualquer coisa com IoU alto --
        numa cena densa tipo MOT17-02 isso pegaria tracks vizinhas sem
        relação nenhuma com o evento)."""
        return {pid: box for pid, box in pred_by_frame.get(f, {}).items() if pid in pred_ids_of_interest}

    pred_ids_seen = list(pred_ids_of_interest)
    cmap = plt.get_cmap("tab10")
    pred_colors = {pid: cmap(i % 10) for i, pid in enumerate(pred_ids_seen)}

    n = len(frames)
    nrows = 3
    ncols = int(np.ceil(n / nrows))
    crop_h, crop_w = crop_y1 - crop_y0, crop_x1 - crop_x0
    panel_w = 2.8
    aspect = (crop_h / crop_w) if crop_w > 0 else 1.3
    aspect = min(aspect, 1.6)  # pessoas em pé dão caixas bem altas/finas -- sem isso a figura
    # inteira (3 linhas) fica gigantesca e esticada verticalmente pra caber a proporção exata
    panel_h = panel_w * aspect
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)
    for ax in axes.flat:  # apaga painéis sobrando (quando n não é múltiplo de 3)
        ax.axis("off")

    for idx, f in enumerate(frames):
        ax = axes[idx // ncols, idx % ncols]
        img = np.array(Image.open(seq.frame_path(f)).convert("RGB"))
        crop = img[int(crop_y0):int(crop_y1), int(crop_x0):int(crop_x1)]
        ax.imshow(crop, aspect="auto")  # preenche o painel (que pode ter sido limitado em altura/largura acima)
        ax.set_title(f"quadro {f}", fontsize=10)
        ax.axis("off")

        def to_crop(x, y):
            return x - crop_x0, y - crop_y0

        for gid in gt_ids_of_interest:
            if gid in gt_by_frame.get(f, {}):
                x, y, w, h = gt_by_frame[f][gid]
                cx, cy = to_crop(x, y)
                ax.add_patch(plt.Rectangle((cx, cy), w, h, fill=False, edgecolor=gt_colors[gid], linewidth=2.5, linestyle="--"))
                ax.text(cx, max(0, cy - 8), f"gt {gid}", color=gt_colors[gid], fontsize=9, fontweight="bold")

        for pid, (x, y, w, h) in pred_ids_relevant(f).items():
            color = pred_colors.get(pid, "red")
            cx, cy = to_crop(x, y)
            ax.add_patch(plt.Rectangle((cx, cy), w, h, fill=False, edgecolor=color, linewidth=2))
            ax.text(cx, cy + h + 14, f"pred {pid}", color=color, fontsize=9)

        # mapa intermediário: caixa PREVISTA pela recorrência (antes de casar)
        for pid in pred_colors:
            box = predicted_box_by_frame.get(f, {}).get(pid)
            if box is not None:
                x, y, w, h = box
                cx, cy = to_crop(x, y)
                ax.add_patch(plt.Rectangle((cx, cy), w, h, fill=False, edgecolor=pred_colors[pid], linewidth=1.3, linestyle=":"))

    fig.suptitle(title, fontsize=11, wrap=True)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.04, hspace=0.12)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ok] figura salva em {out_path}")


if __name__ == "__main__":
    model = MotionGRU(hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()

    DETECTOR = "SDP"
    clips_made = 0
    diagnoses = []

    for name, split in [("MOT17-02", "train"), ("MOT17-11", "test"), ("MOT17-09", "val")]:
        seq = MOT17Sequence(ROOT / "data" / "MOT17" / split / name)
        gt = gt_to_tracks(seq.load_gt())
        det_list = seq.load_det(variant=DETECTOR)

        pred_tracks, predicted_box_by_frame = run_tracker_with_predicted_boxes(seq, det_list, model)
        events = mine_switch_events(gt, pred_tracks)
        print(f"[{name}] {len(events)} eventos de ID switch encontrados")

        # só eventos onde o novo pred_id é MAIOR que o antigo -- garante que é
        # uma track genuinamente NOVA nascendo depois do buraco, não uma
        # track antiga e já existente "roubando" a identidade por proximidade
        # (esse segundo mecanismo já é especificamente a Falha 2).
        def is_genuine_rebirth(e):
            return e["new_pred"] > e["old_pred"]

        if clips_made == 0:
            # Clip 1: maior gap antes de um switch isolado (oclusão longa) -> liga direto ao max_age=5
            isolated = [e for e in events if e["was_fragmentation"] and is_genuine_rebirth(e)]
            if isolated:
                e = max(isolated, key=lambda e: e["gap"])
                render_clip(
                    seq, gt, pred_tracks, predicted_box_by_frame, e["frame"], [e["gt_id"]],
                    [e["old_pred"], e["new_pred"]],
                    "04a_falha_oclusao_longa.png",
                    f"Falha 1 ({name}): oclusão de {e['gap']} quadros > max_age={MAX_AGE} -> "
                    f"ID {e['old_pred']} morre, nasce ID {e['new_pred']} (quadro {e['frame']})",
                )
                diagnoses.append(dict(
                    name="Falha 1 - oclusão mais longa que a memória do tracker", seq=name, frame=e["frame"],
                    text=f"O objeto gt={e['gt_id']} fica sem detecção por {e['gap']} quadros seguidos -- mais que "
                         f"o max_age={MAX_AGE} do tracker -- então a track (pred {e['old_pred']}) morre por regra de "
                         f"nascimento/morte antes mesmo do horizonte de memória do modelo (k~7, medido analiticamente "
                         f"nesta mesma Parte 4) virar o fator limitante; quando o objeto reaparece, vira um ID novo "
                         f"({e['new_pred']}).",
                ))
                clips_made += 1

        swaps = find_simultaneous_swaps(events)
        if clips_made == 1 and swaps:
            e1, e2 = swaps[0]
            render_clip(
                seq, gt, pred_tracks, predicted_box_by_frame, e1["frame"], [e1["gt_id"], e2["gt_id"]],
                [e1["old_pred"], e1["new_pred"], e2["old_pred"], e2["new_pred"]],
                "04b_falha_troca_proximidade.png",
                f"Falha 2 ({name}): troca por proximidade entre gt {e1['gt_id']} e gt {e2['gt_id']} "
                f"no quadro {e1['frame']} (pred {e1['old_pred']}<->{e2['old_pred']})",
            )
            diagnoses.append(dict(
                name="Falha 2 - troca por proximidade (identidade, não memória)", seq=name, frame=e1["frame"],
                text=f"Dois objetos reais (gt {e1['gt_id']} e gt {e2['gt_id']}) passam perto o suficiente um do "
                     f"outro que suas caixas PREVISTAS ficam ambíguas -- o casamento por IoU troca os rótulos "
                     f"(pred {e1['old_pred']}<->{e2['old_pred']}) no mesmo quadro, sem nenhum buraco de detecção "
                     f"envolvido. Isso não é um problema de horizonte de memória -- é o gargalo de identidade em "
                     f"cena densa que a Parte 2 já tinha diagnosticado, e que um modelo de só-movimento não resolve.",
            ))
            clips_made += 1

        if clips_made == 2:
            # Clip 3: outro evento isolado (idealmente de uma sequência diferente), pra variedade
            isolated = [e for e in events if e["was_fragmentation"] and e["gap"] < 15 and is_genuine_rebirth(e)]
            if isolated:
                e = max(isolated, key=lambda e: e["gap"])
                render_clip(
                    seq, gt, pred_tracks, predicted_box_by_frame, e["frame"], [e["gt_id"]],
                    [e["old_pred"], e["new_pred"]],
                    "04c_falha_oclusao_curta.png",
                    f"Falha 3 ({name}): oclusão de só {e['gap']} quadros ainda troca o ID "
                    f"(quadro {e['frame']})",
                )
                diagnoses.append(dict(
                    name="Falha 3 - mesmo oclusões curtas custam caro perto do horizonte atual", seq=name, frame=e["frame"],
                    text=f"Aqui o buraco foi de só {e['gap']} quadros (bem menor que a mediana real do dataset, "
                         f"15 quadros) e ainda assim o objeto gt={e['gt_id']} trocou de ID -- com max_age={MAX_AGE} "
                         f"e horizonte analítico k~7, qualquer buraco de poucos quadros já consome boa parte da "
                         f"margem disponível, mesmo sem ser um caso extremo.",
                ))
                clips_made += 1

        if clips_made >= 3:
            break

    print(f"\n{clips_made}/3 falhas renderizadas.")
    for d in diagnoses:
        print(f"\n=== {d['name']} ({d['seq']}, quadro {d['frame']}) ===")
        print(d["text"])
