"""Parte 2, Trilha A — RNN como modelo de movimento.

Estado por track: um GRUCell. A cada quadro, a entrada é a caixa (x,y,w,h,
normalizada pelo tamanho da imagem) mais um escalar de "confiança" (score
do detector na inferência; visibilidade do gt no treino — ver
pa2/trajectories.py para a justificativa de usar visibilidade como proxy).
A saída é um RESÍDUO somado à caixa de entrada (previsão "velocidade-like":
prever o deslocamento é mais bem condicionado numericamente do que prever a
posição absoluta, e dá uma interpretação direta de movimento).

Sob oclusão (sem observação real naquele quadro), o modelo roda "livre":
em vez de alimentar a caixa verdadeira, alimenta a PRÓPRIA previsão do
passo anterior — exatamente o "estado roda para frente sem observação" do
enunciado. A função `rollout` implementa esse mecanismo de forma idêntica
no treino e na inferência (pa2/motion_tracker.py chama a mesma função).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MotionGRU(nn.Module):
    def __init__(self, hidden_size: int = 64, input_extra: int = 1):
        """input_extra: dimensões além da caixa (aqui, 1: a confiança/visibilidade)."""
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = 4 + input_extra
        self.cell = nn.GRUCell(self.input_size, hidden_size)
        self.head = nn.Linear(hidden_size, 4)
        # começa previsão ~0 (residuo pequeno no início do treino, mais estável)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def step(self, h: torch.Tensor, box: torch.Tensor, conf: torch.Tensor):
        """Um passo: h_{t-1}, caixa_t, conf_t -> h_t, caixa_prevista_{t+1}.

        box: (B,4). conf: (B,1) ou (B,).
        """
        if conf.dim() == 1:
            conf = conf.unsqueeze(-1)
        x = torch.cat([box, conf], dim=-1)
        h_new = self.cell(x, h)
        delta = self.head(h_new)
        pred_next_box = box + delta
        return h_new, pred_next_box

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
        h = self.init_hidden(B, device=device)

        cur_box = boxes[:, 0]
        cur_conf = confs[:, 0]
        predictions = []

        for t in range(T):
            h, pred_next = self.step(h, cur_box, cur_conf)
            predictions.append(pred_next)
            if t < T - 1:
                # decide, POR AMOSTRA do batch, se o próximo passo usa a observação
                # real (t+1) ou a própria previsão (free-running) -- nunca um .all()
                # colapsando o batch inteiro numa única decisão.
                obs_next = observed_mask[:, t + 1].unsqueeze(-1).float()
                cur_box = obs_next * boxes[:, t + 1] + (1 - obs_next) * pred_next
                cur_conf = observed_mask[:, t + 1].float() * confs[:, t + 1]

        return torch.stack(predictions, dim=1)
