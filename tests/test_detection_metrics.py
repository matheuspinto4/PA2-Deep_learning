import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa2.detection_metrics import average_precision  # noqa: E402

BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (50.0, 50.0, 10.0, 10.0)


def test_ap_perfeito():
    gt = [(0, 1, *BOX_A), (0, 2, *BOX_B)]
    det = [(0, *BOX_A, 0.9), (0, *BOX_B, 0.8)]
    r = average_precision(gt, det)
    assert math.isclose(r["ap"], 1.0, rel_tol=1e-9), r


def test_ap_com_falso_positivo_de_score_baixo_nao_derruba_muito():
    gt = [(0, 1, *BOX_A)]
    det = [(0, *BOX_A, 0.9), (0, 20.0, 20.0, 5.0, 5.0, 0.1)]  # FP sem overlap, score baixo
    r = average_precision(gt, det)
    # recall chega a 1 antes do FP entrar (FP tem score menor, vem depois na
    # ordenação) -> AP ainda deve ficar bem alto, mas não necessariamente 1.0
    # pois o ponto de recall=1 inclui o FP na precisão acumulada.
    assert r["ap"] > 0.9, r


def test_ap_sem_deteccao_nenhuma():
    gt = [(0, 1, *BOX_A)]
    det = []
    r = average_precision(gt, det)
    assert r["ap"] == 0.0


def test_ap_deteccao_errada_zero():
    gt = [(0, 1, *BOX_A)]
    det = [(0, *BOX_B, 0.9)]  # não sobrepõe nada
    r = average_precision(gt, det)
    assert r["ap"] == 0.0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
