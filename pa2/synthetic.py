"""Parte 0, itens 1 e 2: gerador de vídeos sintéticos com oclusão real e
simulador de detector.

O gerador produz vídeos 128x128 com elipses em movimento, desenhadas com
ordem de profundidade (depth/z-order) explícita: uma elipse de depth maior é
desenhada por cima e realmente apaga (oculta) o que está atrás dela no
quadro, e isso é exatamente o que medimos via `visibility` (fração de área
não coberta). Um par de elipses é sempre roteirizado analiticamente para
garantir N quadros de oclusão total controlada (parâmetro
`occlusion_duration`), satisfazendo o requisito verificável do enunciado.

O simulador de detector é uma função pura (gt -> detecções ruidosas) que
descarta, com probabilidade maior quanto menor a visibilidade do alvo,
adiciona ruído gaussiano às coordenadas, e injeta falsos positivos — usado
para validar a Parte 1 inteiramente no sintético, e reaproveitado como o
próprio experimento de degradação de detector da Parte 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class _Obj:
    obj_id: int
    pos0: np.ndarray  # posição no frame 0 (x,y)
    vel: np.ndarray  # px/frame (vx,vy), constante
    rx: float
    ry: float
    depth: int
    intensity: float  # cinza de preenchimento, 0-255
    bounce: bool = True  # se True, quica nas paredes; se False, trajetória linear livre (pares roteirizados)

    def center_at(self, t: int, width: int, height: int) -> np.ndarray:
        if not self.bounce:
            return self.pos0 + self.vel * t
        # quicar nas paredes: reflete a posição "desdobrada" (triangle wave),
        # equivalente a inverter a velocidade a cada colisão, mas sem laço.
        x = self.pos0[0] + self.vel[0] * t
        y = self.pos0[1] + self.vel[1] * t
        x = _reflect(x, self.rx, width - self.rx)
        y = _reflect(y, self.ry, height - self.ry)
        return np.array([x, y])


def _reflect(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return (lo + hi) / 2
    span = hi - lo
    v = (value - lo) % (2 * span)
    if v < 0:
        v += 2 * span
    return lo + (v if v <= span else 2 * span - v)


@dataclass
class SyntheticVideoGenerator:
    width: int = 128
    height: int = 128
    n_frames: int | None = None  # se None, sorteia em [30,60]
    num_objects: int = 10
    speed: float = 2.5  # px/quadro "típico"
    occlusion_duration: int = 10  # quadros de oclusão total roteirizada
    noise_std: float = 12.0  # ruído do fundo
    size_range: tuple = (5.0, 14.0)
    seed: int | None = None

    def generate(self):
        rng = np.random.default_rng(self.seed)
        n_frames = self.n_frames if self.n_frames is not None else int(rng.integers(30, 61))

        objects = self._build_objects(rng, n_frames)

        frames = np.zeros((n_frames, self.height, self.width), dtype=np.uint8)
        gt_detections = []  # (frame, id, x, y, w, h)
        visibilities = {}  # (frame, id) -> float em [0,1]

        objs_by_depth = sorted(objects, key=lambda o: o.depth)

        for t in range(n_frames):
            centers = {o.obj_id: o.center_at(t, self.width, self.height) for o in objects}
            masks = {o.obj_id: self._ellipse_mask(centers[o.obj_id], o.rx, o.ry) for o in objects}

            # visibilidade: área do objeto não coberta por nenhum objeto de depth maior
            # (O(n^2) em máscaras, n pequeno o suficiente para não importar).
            for o in objects:
                union_above = np.zeros((self.height, self.width), dtype=bool)
                for o2 in objects:
                    if o2.depth > o.depth:
                        union_above |= masks[o2.obj_id]
                own = masks[o.obj_id]
                own_area = own.sum()
                if own_area == 0:
                    vis = 0.0
                else:
                    vis = float((own & ~union_above).sum()) / float(own_area)
                visibilities[(t, o.obj_id)] = vis

                x, y, w, h = self._bbox_from_center(centers[o.obj_id], o.rx, o.ry)
                x, y, w, h = self._clip_box(x, y, w, h)
                if w > 0 and h > 0:
                    gt_detections.append((t, o.obj_id, x, y, w, h))

            frames[t] = self._render_frame(objs_by_depth, centers, rng)

        meta = self._occlusion_meta
        return frames, gt_detections, visibilities, meta

    # ---- construção dos objetos ----

    def _build_objects(self, rng, n_frames):
        objects = []
        depths = rng.permutation(self.num_objects)
        for i in range(self.num_objects):
            rx = rng.uniform(*self.size_range)
            ry = rng.uniform(*self.size_range) * rng.uniform(0.7, 1.3)
            margin = max(rx, ry) + 2
            pos0 = rng.uniform([margin, margin], [self.width - margin, self.height - margin])
            angle = rng.uniform(0, 2 * np.pi)
            spd = self.speed * rng.uniform(0.6, 1.4)
            vel = spd * np.array([np.cos(angle), np.sin(angle)])
            intensity = rng.uniform(60, 230)
            objects.append(_Obj(i, pos0, vel, rx, ry, int(depths[i]), intensity, bounce=True))

        self._occlusion_meta = None
        if self.num_objects >= 2 and self.occlusion_duration > 0:
            self._script_occlusion(objects, rng, n_frames)
        return objects

    def _script_occlusion(self, objects, rng, n_frames):
        """Sobrescreve a trajetória de um par (occluder, occluded) para garantir
        exatamente `occlusion_duration` quadros de oclusão total, centrados no
        meio do vídeo (ver docstring do módulo para a dedução analítica)."""
        occluder, occluded = objects[0], objects[1]

        occluder.rx = occluded.rx + rng.uniform(6.0, 10.0)
        occluder.ry = occluder.rx * rng.uniform(0.9, 1.1)
        occluder.depth = max(o.depth for o in objects) + 1  # garante que fica por cima de todos
        occluded.depth = min(o.depth for o in objects) - 1  # garante que fica atrás de todos

        r_occluder = (occluder.rx + occluder.ry) / 2
        r_occluded = (occluded.rx + occluded.ry) / 2
        r_safe = max(r_occluder - r_occluded, 3.0)

        t_mid = n_frames // 2
        D = max(2, min(self.occlusion_duration, n_frames - 4))
        half = D / 2.0
        rel_speed = r_safe / half  # |P_hid(t) - P_occ(t)| = rel_speed * |t - t_mid|

        center = np.array([self.width / 2.0, self.height / 2.0])
        occluder.pos0 = center.copy()
        occluder.vel = np.zeros(2)
        occluder.bounce = False

        direction = rng.uniform(-1, 1, size=2)
        direction /= np.linalg.norm(direction) + 1e-9

        occluded.vel = rel_speed * direction
        occluded.pos0 = center - occluded.vel * t_mid
        occluded.bounce = False

        self._occlusion_meta = dict(
            occluder_id=occluder.obj_id,
            occluded_id=occluded.obj_id,
            t_mid=t_mid,
            window=(int(round(t_mid - half)), int(round(t_mid + half))),
            r_safe=r_safe,
        )

    # ---- geometria / render ----

    def _ellipse_mask(self, center, rx, ry) -> np.ndarray:
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        cx, cy = center
        return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

    @staticmethod
    def _bbox_from_center(center, rx, ry):
        cx, cy = center
        return cx - rx, cy - ry, 2 * rx, 2 * ry

    def _clip_box(self, x, y, w, h):
        x2, y2 = x + w, y + h
        x = max(0.0, min(x, self.width))
        y = max(0.0, min(y, self.height))
        x2 = max(0.0, min(x2, self.width))
        y2 = max(0.0, min(y2, self.height))
        return x, y, max(0.0, x2 - x), max(0.0, y2 - y)

    def _render_frame(self, objs_by_depth, centers, rng) -> np.ndarray:
        bg = rng.normal(loc=30.0, scale=self.noise_std, size=(self.height, self.width))
        img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), mode="L")
        draw = ImageDraw.Draw(img)
        for o in objs_by_depth:  # depth crescente: quem tem depth maior é desenhado por último (por cima)
            cx, cy = centers[o.obj_id]
            draw.ellipse([cx - o.rx, cy - o.ry, cx + o.rx, cy + o.ry], fill=int(o.intensity))
        arr = np.array(img, dtype=np.float64)
        arr += rng.normal(0, self.noise_std * 0.3, size=arr.shape)  # ruído de captura por cima do desenho
        return np.clip(arr, 0, 255).astype(np.uint8)


def simulate_detections(
    gt_detections,
    visibilities: dict | None = None,
    drop_prob: float = 0.1,
    occlusion_sensitivity: float = 2.0,
    coord_noise_std: float = 2.0,
    size_noise_std: float = 1.5,
    fp_rate: float = 0.1,
    fp_size_range: tuple = (5.0, 20.0),
    image_size: tuple = (128, 128),
    seed: int | None = None,
):
    """Estraga caixas verdadeiras de propósito, simulando um detector real.

    gt_detections: lista de (frame, id, x, y, w, h). Note que a identidade é
    deliberadamente descartada na saída: um detector não produz identidade,
    só caixas (+ score), que é exatamente o motivo do PA existir.

    Para cada gt, a probabilidade efetiva de descarte cresce quando a
    visibilidade é baixa (`occlusion_sensitivity` controla o quanto): um
    detector real erra mais em alvos parcialmente ocluídos.

    Retorna lista de (frame, x, y, w, h, score).
    """
    rng = np.random.default_rng(seed)
    width, height = image_size
    out = []

    frames_present = sorted(set(d[0] for d in gt_detections))

    for frame, obj_id, x, y, w, h in gt_detections:
        vis = 1.0
        if visibilities is not None:
            vis = visibilities.get((frame, obj_id), 1.0)
        p_drop = drop_prob + (1 - drop_prob) * occlusion_sensitivity * (1 - vis)
        p_drop = min(1.0, max(0.0, p_drop)) if occlusion_sensitivity > 0 else drop_prob
        if rng.uniform() < p_drop:
            continue

        cx, cy = x + w / 2, y + h / 2
        cx += rng.normal(0, coord_noise_std)
        cy += rng.normal(0, coord_noise_std)
        w2 = max(1.0, w + rng.normal(0, size_noise_std))
        h2 = max(1.0, h + rng.normal(0, size_noise_std))
        nx, ny = cx - w2 / 2, cy - h2 / 2
        score = float(np.clip(vis * rng.uniform(0.7, 1.0) + rng.uniform(-0.05, 0.05), 0.05, 1.0))
        out.append((frame, nx, ny, w2, h2, score))

    for frame in frames_present:
        n_fp = rng.poisson(fp_rate)
        for _ in range(n_fp):
            w = rng.uniform(*fp_size_range)
            h = rng.uniform(*fp_size_range)
            x = rng.uniform(0, max(1.0, width - w))
            y = rng.uniform(0, max(1.0, height - h))
            score = float(rng.uniform(0.05, 0.6))  # FPs tendem a score mais baixo, mas não sempre
            out.append((frame, x, y, w, h, score))

    out.sort(key=lambda d: d[0])
    return out
