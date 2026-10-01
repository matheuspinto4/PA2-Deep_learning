import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa2.trajectories import extract_contiguous_trajectories, make_windows  # noqa: E402

BOX = (10.0, 10.0, 4.0, 4.0)


def make_dets(obj_id, frames, box=BOX):
    return [(f, obj_id, *box) for f in frames]


def test_quebra_trajetoria_em_buraco_real():
    dets = make_dets(1, [0, 1, 2, 5, 6, 7, 8])  # buraco entre 2 e 5
    trajs = extract_contiguous_trajectories(dets, image_size=(100, 100), min_length=2)
    lengths = sorted(len(t) for t in trajs)
    assert lengths == [3, 4], lengths


def test_descarta_trecho_curto_demais():
    # um único quadro isolado no meio -> trecho de tamanho 1, descartado com min_length=2
    dets = make_dets(1, [0, 1, 5, 10, 11, 12])
    trajs = extract_contiguous_trajectories(dets, image_size=(100, 100), min_length=2)
    lengths = sorted(len(t) for t in trajs)
    assert lengths == [2, 3], lengths


def test_normaliza_pelo_tamanho_da_imagem():
    dets = make_dets(1, [0, 1], box=(50.0, 25.0, 10.0, 5.0))
    trajs = extract_contiguous_trajectories(dets, image_size=(100, 50), min_length=2)
    assert len(trajs) == 1
    x, y, w, h, vis = trajs[0][0]
    assert np.isclose(x, 0.5) and np.isclose(y, 0.5) and np.isclose(w, 0.1) and np.isclose(h, 0.1)


def test_janelas_tamanho_fixo_com_passo():
    traj = np.zeros((10, 5), dtype=np.float32)
    windows = make_windows([traj], window=4, stride=2)
    # starts: 0,2,4,6 (10-4+1=7, range(0,7,2) -> 0,2,4,6)
    assert len(windows) == 4
    assert all(w.shape == (4, 5) for w in windows)


def test_trajetoria_curta_demais_nao_gera_janela():
    traj = np.zeros((3, 5), dtype=np.float32)
    windows = make_windows([traj], window=4, stride=2)
    assert windows == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
