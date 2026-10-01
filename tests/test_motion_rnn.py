import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa2.models.motion_rnn import MotionGRU, MotionRNN, matched_hidden_size  # noqa: E402

CELL_TYPES = ["rnn", "lstm", "gru"]


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_rollout_teacher_forcing_com_pesos_zero_reproduz_a_entrada(cell_type):
    """Com a cabeça de saída inicializada em zero, delta=0, então em modo
    totalmente observado (teacher forcing) a previsão em cada passo deve
    ser exatamente a caixa de entrada daquele passo (nenhum movimento
    aprendido ainda -- só confere que o encadeamento t->t+1 está certo).
    Vale pras 3 células (RNN simples, LSTM, GRU): a cabeça zerada anula
    qualquer coisa que a célula computar internamente."""
    torch.manual_seed(0)
    model = MotionRNN(cell_type=cell_type, hidden_size=8)

    B, T = 2, 4
    boxes = torch.rand(B, T, 4)
    confs = torch.ones(B, T)
    observed_mask = torch.ones(B, T, dtype=torch.bool)

    preds = model.rollout(boxes, confs, observed_mask)

    assert preds.shape == (B, T, 4)
    assert torch.allclose(preds, boxes, atol=1e-6)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_rollout_free_running_usa_a_propria_previsao_quando_nao_observado(cell_type):
    """T=4, observado = [T,T,F,T]. Com pesos zero (delta=0):
      t=0: entrada=box0 -> previsão0=box0
      t=1: entrada=box1 (observado) -> previsão1=box1
      t=2: NÃO observado -> entrada = previsão1 (=box1), não box2!
           -> previsão2 = box1 (não box2)
      t=3: observado -> entrada=box3 -> previsão3=box3
    """
    torch.manual_seed(0)
    model = MotionRNN(cell_type=cell_type, hidden_size=8)

    B, T = 1, 4
    boxes = torch.rand(B, T, 4)
    confs = torch.ones(B, T)
    observed_mask = torch.tensor([[True, True, False, True]])

    preds = model.rollout(boxes, confs, observed_mask)

    assert torch.allclose(preds[:, 0], boxes[:, 0], atol=1e-6)
    assert torch.allclose(preds[:, 1], boxes[:, 1], atol=1e-6)
    assert torch.allclose(preds[:, 2], boxes[:, 1], atol=1e-6), "quadro oculto deveria copiar a previsão anterior, não ver a caixa verdadeira"
    assert not torch.allclose(preds[:, 2], boxes[:, 2], atol=1e-6)
    assert torch.allclose(preds[:, 3], boxes[:, 3], atol=1e-6)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_rollout_decisao_e_por_amostra_no_batch(cell_type):
    """Duas amostras no mesmo batch com máscaras de observação DIFERENTES
    não podem vazar a decisão uma pra outra (bug do .all() que corrigimos)."""
    torch.manual_seed(0)
    model = MotionRNN(cell_type=cell_type, hidden_size=8)

    B, T = 2, 3
    boxes = torch.rand(B, T, 4)
    confs = torch.ones(B, T)
    # amostra 0: tudo observado. amostra 1: quadro do meio oculto.
    observed_mask = torch.tensor([[True, True, True], [True, False, True]])

    preds = model.rollout(boxes, confs, observed_mask)

    assert torch.allclose(preds[0, 1], boxes[0, 1], atol=1e-6)  # amostra 0, t=1: observado
    assert torch.allclose(preds[1, 1], boxes[1, 0], atol=1e-6)  # amostra 1, t=1: free-running (copia t=0)


def test_motiongru_alias_continua_funcionando():
    """Compatibilidade com a Parte 2: MotionGRU(...) ainda existe e devolve
    um MotionRNN configurado pra GRU."""
    model = MotionGRU(hidden_size=16)
    assert isinstance(model, MotionRNN)
    assert model.cell_type == "gru"
    assert model.hidden_size == 16


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_matched_hidden_size_aproxima_o_orcamento_de_parametros(cell_type):
    """A Parte 3 pede 'mesmo orçamento aproximado de parâmetros' entre as
    3 células -- confere que matched_hidden_size encontra um hidden_size
    cujo nº de parâmetros da célula fica razoavelmente perto do alvo."""
    target = MotionRNN(cell_type="gru", hidden_size=64).num_cell_parameters()
    h = matched_hidden_size(cell_type, target_params=target)
    achieved = MotionRNN(cell_type=cell_type, hidden_size=h).num_cell_parameters()
    # tolerância generosa (10%): o espaço de hidden_size é discreto, não dá
    # pra bater o alvo exatamente, só "aproximado" como o enunciado pede.
    assert abs(achieved - target) / target < 0.10, (cell_type, h, achieved, target)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
