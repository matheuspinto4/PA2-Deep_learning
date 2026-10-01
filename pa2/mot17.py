"""Leitura dos arquivos do MOT17 (gt.txt, det.txt, seqinfo.ini).

Formato do gt.txt (MOTChallenge): frame, id, bb_left, bb_top, bb_width,
bb_height, conf, class, visibility. `conf` aqui não é confiança de detecção:
é uma flag 0/1 que indica se a anotação deve ser considerada na avaliação
(0 = região de distração/ignorar). `class` segue a convenção do devkit do
MOT17 (1 = pedestrian, o que avaliamos; demais classes são distratores como
pessoa estática, reflexo, veículo etc. e não contam como alvo).

Formato do det.txt: frame, -1, bb_left, bb_top, bb_width, bb_height, conf,
-1, -1, -1. O id é sempre -1 (detecção não tem identidade); `conf` aqui É a
confiança do detector.
"""

from __future__ import annotations

import configparser
from pathlib import Path


def load_gt(path, pedestrian_class: int = 1, require_considered: bool = True, min_visibility: float = 0.0):
    """Lê gt.txt. Retorna lista de (frame, id, x, y, w, h, visibility)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(parts[0])
            obj_id = int(parts[1])
            x, y, w, h = (float(v) for v in parts[2:6])
            conf = int(float(parts[6]))
            cls = int(float(parts[7])) if len(parts) > 7 else pedestrian_class
            vis = float(parts[8]) if len(parts) > 8 else 1.0

            if require_considered and conf != 1:
                continue
            if cls != pedestrian_class:
                continue
            if vis < min_visibility:
                continue
            out.append((frame, obj_id, x, y, w, h, vis))
    return out


def load_det(path):
    """Lê det.txt (detecções públicas). Retorna lista de (frame, x, y, w, h, score)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(parts[0])
            x, y, w, h = (float(v) for v in parts[2:6])
            score = float(parts[6])
            out.append((frame, x, y, w, h, score))
    return out


def load_seqinfo(path) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    s = cfg["Sequence"]
    return dict(
        name=s.get("name"),
        imDir=s.get("imDir"),
        frameRate=int(s.get("frameRate")),
        seqLength=int(s.get("seqLength")),
        imWidth=int(s.get("imWidth")),
        imHeight=int(s.get("imHeight")),
        imExt=s.get("imExt"),
    )


def gt_to_tracks(gt_rows):
    """(frame,id,x,y,w,h,vis) -> (frame,id,x,y,w,h) para uso direto em metrics.py."""
    return [(f, i, x, y, w, h) for f, i, x, y, w, h, _ in gt_rows]


def gt_visibility_map(gt_rows):
    return {(f, i): vis for f, i, _, _, _, _, vis in gt_rows}


class MOT17Sequence:
    """Agrupa os arquivos de uma sequência (ex.: MOT17-02-FRCNN) já baixados localmente."""

    def __init__(self, seq_dir):
        self.seq_dir = Path(seq_dir)
        self.info = load_seqinfo(self.seq_dir / "seqinfo.ini")

    @property
    def name(self):
        return self.info["name"]

    @property
    def n_frames(self):
        return self.info["seqLength"]

    def frame_path(self, frame: int) -> Path:
        return self.seq_dir / self.info["imDir"] / f"{frame:06d}{self.info['imExt']}"

    def load_gt(self, **kwargs):
        return load_gt(self.seq_dir / "gt" / "gt.txt", **kwargs)

    def load_det(self, variant: str = "FRCNN"):
        """variant: DPM, FRCNN ou SDP (ver pa2/mot17.py docstring do módulo
        e scripts/download_mot17_subset.py -- baixamos as 3 variantes de
        detecção pública por sequência, já que os arquivos são pequenos)."""
        path = self.seq_dir / "det" / f"det_{variant}.txt"
        if not path.exists():
            path = self.seq_dir / "det" / "det.txt"  # fallback p/ layout oficial do MOT17.zip
        return load_det(path)
