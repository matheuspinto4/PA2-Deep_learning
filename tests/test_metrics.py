"""Testes de metrics.py com casos construídos à mão (valores derivados analiticamente,
ver AI_LOG.md / notas da Parte 0 para a dedução de cada número esperado).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import compute_mot_metrics, iou_xywh  # noqa: E402

BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (100.0, 100.0, 10.0, 10.0)  # bem longe de A, IoU(A,B)=0


def make_track(obj_id, box, frames):
    return [(f, obj_id, *box) for f in frames]


def test_iou_basic():
    assert iou_xywh(BOX_A, BOX_A) == 1.0
    assert iou_xywh(BOX_A, BOX_B) == 0.0
    # metade sobreposta: caixa 10x10 deslocada 5 em x -> inter=5x10=50, union=200-50=150
    half = (5.0, 0.0, 10.0, 10.0)
    assert math.isclose(iou_xywh(BOX_A, half), 50.0 / 150.0, rel_tol=1e-9)


def test_caso_a_predicao_igual_ground_truth():
    """pred == gt inteiro -> IDF1=1, 0 switches, 0 fragmentações."""
    frames = range(10)
    gt = make_track(1, BOX_A, frames) + make_track(2, BOX_B, frames)
    pred = make_track(1, BOX_A, frames) + make_track(2, BOX_B, frames)

    m = compute_mot_metrics(gt, pred)
    assert math.isclose(m["idf1"], 1.0, rel_tol=1e-9)
    assert m["id_switches"] == 0
    assert m["fragmentations"] == 0
    assert m["idfn"] == 0 and m["idfp"] == 0


def test_caso_b_duas_identidades_trocadas_a_partir_do_quadro_k():
    """Dois trackers (ids 10,20) trocam de alvo no quadro 5 (de 10 quadros totais),
    sem nenhum gap: tracker 10 segue A nos quadros 0-4 e passa a seguir B nos
    quadros 5-9; tracker 20 faz o inverso. Cada caixa prevista sempre coincide
    perfeitamente com a posição verdadeira de algum alvo (IoU=1), então não há
    nenhum FN/FP "espacial" em quadro nenhum — o único erro é de identidade.

    Esperado (deduzido à mão, ver metrics.py/AI_LOG):
      - 2 ID switches (um para a identidade A, um para a identidade B, ambos no
        quadro em que o casamento espacial deixa de ser o tracker anterior);
      - 0 fragmentações (nunca há quadro em que A ou B fiquem sem correspondência);
      - IDF1 = 0.5 (o assignment global de identidade só pode escolher UM tracker
        por alvo verdadeiro; qualquer escolha cobre exatamente metade dos quadros
        de cada alvo, então idtp=10, idfn=idfp=10 -> IDF1 = 2*10/(20+10+10) = 0.5).
    """
    k = 5
    frames_before = range(0, k)
    frames_after = range(k, 10)

    gt = make_track(1, BOX_A, range(10)) + make_track(2, BOX_B, range(10))
    pred = (
        make_track(10, BOX_A, frames_before)
        + make_track(10, BOX_B, frames_after)
        + make_track(20, BOX_B, frames_before)
        + make_track(20, BOX_A, frames_after)
    )

    m = compute_mot_metrics(gt, pred)
    assert m["id_switches"] == 2, m
    assert m["fragmentations"] == 0, m
    assert math.isclose(m["idf1"], 0.5, rel_tol=1e-9), m
    assert m["idtp"] == 10 and m["idfn"] == 10 and m["idfp"] == 10


def test_caso_c_track_partida_em_duas_no_meio():
    """Um único alvo (A) é corretamente seguido pelo tracker 10 nos quadros 0-4,
    perdido (sem nenhuma detecção) nos quadros 5-6, e retomado pelo tracker 11 nos
    quadros 7-9 — uma fragmentação real causada por um buraco de detecção, não por
    uma simples troca de rótulo como no caso (b).

    Esperado (deduzido à mão):
      - 1 ID switch (o último id conhecido de A muda de 10 para 11 ao retomar) e
        1 fragmentação (A estava sendo rastreado, ficou sem correspondência por
        >=1 quadro, e voltou a ser casado depois);
      - idtp = 5 (só os quadros 0-4, cobertos pelo tracker mais longo, 10, que o
        assignment global escolhe); idfn = 10-5 = 5; idfp = 3 (os quadros do
        tracker 11, que fica sem par real já que só pode haver um tracker por
        alvo); IDF1 = 2*5/(10+5+3) = 10/18 ≈ 0.5556.
      Note que o número de switches (1, não 2) e o mecanismo do erro (FN+FP
      genuínos por um buraco, não uma realocação simétrica de rótulos) são
      exatamente o que diferencia este caso do caso (b), mesmo a magnitude do
      IDF1 sendo parecida.
    """
    gt = make_track(1, BOX_A, range(10))
    pred = make_track(10, BOX_A, range(0, 5)) + make_track(11, BOX_A, range(7, 10))

    m = compute_mot_metrics(gt, pred)
    assert m["id_switches"] == 1, m
    assert m["fragmentations"] == 1, m
    assert m["idtp"] == 5 and m["idfn"] == 5 and m["idfp"] == 3, m
    assert math.isclose(m["idf1"], 10 / 18, rel_tol=1e-9), m

    # garante explicitamente que o efeito em IDF1 não é "o mesmo" do caso (b):
    # aqui o erro vem de FN+FP genuínos (idfp=3 > 0 por frames sem par real),
    # lá o erro vinha só de uma realocação simétrica (idfp == idfn == metade de cada alvo).
    m_b = compute_mot_metrics(
        make_track(1, BOX_A, range(10)) + make_track(2, BOX_B, range(10)),
        make_track(10, BOX_A, range(0, 5)) + make_track(10, BOX_B, range(5, 10))
        + make_track(20, BOX_B, range(0, 5)) + make_track(20, BOX_A, range(5, 10)),
    )
    assert m["id_switches"] != m_b["id_switches"] or m["fragmentations"] != m_b["fragmentations"]


def test_contagem_de_identidades():
    gt = make_track(1, BOX_A, range(5)) + make_track(2, BOX_B, range(5))
    pred = make_track(10, BOX_A, range(5)) + make_track(11, BOX_B, range(5)) + make_track(12, BOX_B, range(5, 6))
    m = compute_mot_metrics(gt, pred)
    assert m["n_gt_ids"] == 2
    assert m["n_pred_ids"] == 3
    assert m["diff"] == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
