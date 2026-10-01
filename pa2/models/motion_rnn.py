"""Parte 2, Trilha A — RNN como modelo de movimento.
Parte 3 — generalizado pra suportar as 3 células do Eixo 1 da ablação
(RNN simples / LSTM / GRU) atrás da MESMA interface, pra poder trocar a
célula sem mudar nem o laço de treino nem o tracker.

Estado por track: uma célula recorrente (RNNCell, LSTMCell ou GRUCell). A
cada quadro, a entrada é a caixa (x,y,w,h, normalizada pelo tamanho da
imagem) mais um escalar de "confiança" (score do detector na inferência;
visibilidade do gt no treino — ver pa2/trajectories.py). A saída é um
RESÍDUO somado à caixa de entrada (previsão "velocidade-like": prever o
deslocamento é mais bem condicionado numericamente do que prever a posição
absoluta).

Sob oclusão (sem observação real naquele quadro), o modelo roda "livre":
em vez de alimentar a caixa verdadeira, alimenta a PRÓPRIA previsão do
passo anterior — exatamente o "estado roda para frente sem observação" do
enunciado. A função `rollout` implementa esse mecanismo de forma idêntica
no treino e na inferência (pa2/motion_tracker.py chama a mesma função).

O estado interno é sempre uma TUPLA (h,) para RNN/GRU ou (h,c) para LSTM,
pra tratar as três células com o mesmo código em `step`/`rollout`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CELL_CLASSES = {"rnn": nn.RNNCell, "lstm": nn.LSTMCell, "gru": nn.GRUCell}


class MotionRNN(nn.Module):
    def __init__(self, cell_type: str = "gru", hidden_size: int = 64, input_extra: int = 1):
        """cell_type: 'rnn', 'lstm' ou 'gru'. input_extra: dimensões além
        da caixa (aqui, 1: a confiança/visibilidade)."""
        super().__init__()
        assert cell_type in CELL_CLASSES, f"cell_type deve ser um de {list(CELL_CLASSES)}"
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.input_size = 4 + input_extra
        self.cell = CELL_CLASSES[cell_type](self.input_size, hidden_size)
        self.head = nn.Linear(hidden_size, 4)
        # começa previsão ~0 (residuo pequeno no início do treino, mais estável)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def num_cell_parameters(self) -> int:
        """Só os parâmetros da célula recorrente (sem a cabeça de saída) --
        é essa contagem que a Parte 3 equaliza entre RNN/LSTM/GRU."""
        return sum(p.numel() for p in self.cell.parameters())

    def init_hidden(self, batch_size: int, device=None) -> tuple:
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        if self.cell_type == "lstm":
            c = torch.zeros(batch_size, self.hidden_size, device=device)
            return (h, c)
        return (h,)

    def step(self, state: tuple, box: torch.Tensor, conf: torch.Tensor):
        """Um passo: state_{t-1}, caixa_t, conf_t -> state_t, caixa_prevista_{t+1}.

        box: (B,4). conf: (B,1) ou (B,). state: tupla, ver `init_hidden`.
        """
        if conf.dim() == 1:
            conf = conf.unsqueeze(-1)
        x = torch.cat([box, conf], dim=-1)

        if self.cell_type == "lstm":
            h, c = state
            h_new, c_new = self.cell(x, (h, c))
            new_state = (h_new, c_new)
            h_for_head = h_new
        else:
            (h,) = state
            h_new = self.cell(x, h)
            new_state = (h_new,)
            h_for_head = h_new

        delta = self.head(h_for_head)
        pred_next_box = box + delta
        return new_state, pred_next_box

    def rollout(self, boxes: torch.Tensor, confs: torch.Tensor, observed_mask: torch.Tensor):
        """Desenrola a sequência inteira, decidindo quadro a quadro se
        alimenta a caixa verdadeira (observada) ou a própria previsão
        anterior (free-running, simulando oclusão / detecção perdida).

        boxes:  (B, T, 4)  caixas verdadeiras (gt no treino, ou observação
                real do detector na inferência quando existir).
        confs:  (B, T)     confiança/visibilidade de cada quadro.
        observed_mask: (B, T) bool -- True = há observação real neste
                quadro (usa `boxes`/`confs` como entrada); False = não há
                (usa a previsão do passo anterior como entrada, com
                confiança 0). observed_mask[:,0] deve ser sempre True (uma
                track só nasce a partir de uma observação real).

        Retorna predictions: (B, T, 4), a previsão feita em cada passo t
        para o quadro t+1 (ou seja, predictions[:, t] tenta acertar
        boxes[:, t+1] -- o chamador compara predictions[:, :-1] contra
        boxes[:, 1:]).
        """
        B, T, _ = boxes.shape
        device = boxes.device
        state = self.init_hidden(B, device=device)

        cur_box = boxes[:, 0]
        cur_conf = confs[:, 0]
        predictions = []

        for t in range(T):
            state, pred_next = self.step(state, cur_box, cur_conf)
            predictions.append(pred_next)
            if t < T - 1:
                # decide, POR AMOSTRA do batch, se o próximo passo usa a observação
                # real (t+1) ou a própria previsão (free-running) -- nunca um .all()
                # colapsando o batch inteiro numa única decisão.
                obs_next = observed_mask[:, t + 1].unsqueeze(-1).float()
                cur_box = obs_next * boxes[:, t + 1] + (1 - obs_next) * pred_next
                cur_conf = observed_mask[:, t + 1].float() * confs[:, t + 1]

        return torch.stack(predictions, dim=1)


def MotionGRU(hidden_size: int = 64, input_extra: int = 1) -> MotionRNN:
    """Alias de compatibilidade com a Parte 2 (onde só existia GRU)."""
    return MotionRNN(cell_type="gru", hidden_size=hidden_size, input_extra=input_extra)


def matched_hidden_size(cell_type: str, target_params: int, input_extra: int = 1, search_range=(2, 400)) -> int:
    """Acha o hidden_size cujo nº de parâmetros da célula fica mais perto de
    `target_params` -- usado pra equalizar o orçamento de parâmetros entre
    RNN simples / LSTM / GRU (Parte 3, Eixo 1: "mesmo orçamento aproximado
    de parâmetros")."""
    best_h, best_diff = None, float("inf")
    for h in range(*search_range):
        model = MotionRNN(cell_type=cell_type, hidden_size=h, input_extra=input_extra)
        diff = abs(model.num_cell_parameters() - target_params)
        if diff < best_diff:
            best_h, best_diff = h, diff
    return best_h
