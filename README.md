# PA2 — Identidade ao longo do tempo

Detecção + recorrência + rastreamento multi-objeto (MOT), sem rastreadores
prontos. Ver `PA2.pdf` para o enunciado completo.

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

Principais dependências: `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`,
`matplotlib`, `Pillow`, `pytest`. (sem GPU disponível neste ambiente de
desenvolvimento — os experimentos foram desenhados para caber em CPU: vídeos
sintéticos pequenos, e um subconjunto reduzido de sequências do MOT17.)

## Estrutura

```
metrics.py                  # IDF1, ID switches, fragmentações (implementação própria)
pa2/
  synthetic.py               # gerador de vídeo sintético + simulador de detector (Parte 0)
  nms.py                     # NMS implementado do zero
  association.py             # matching guloso / húngaro por IoU
  baseline_tracker.py        # tracker ingênuo por quadro (Partes 0 e 1)
  mot17.py                   # leitura de det.txt / gt.txt / seqinfo.ini do MOT17
  detection_metrics.py       # Average Precision (mAP de 1 classe), implementação própria
  torch_detector.py          # Faster R-CNN pré-treinado do torchvision (2ª fonte de detecção)
  models/                    # modelo temporal recorrente (Parte 2+)
tests/
  test_metrics.py             # os 3 casos de teste construídos à mão da Parte 0
  test_detection_metrics.py   # sanity checks do AP
scripts/
  download_mot17_subset.py           # baixa só as 4 sequências usadas (range-request HTTP)
  part0_synthetic_experiments.py     # gera todos os artefatos da Parte 0
  part1_baseline.py                  # escolha do detector público + baseline + gráfico obrigatório
  part1_torchvision_detector.py      # 2ª fonte de detecção (torchvision) + comparação
outputs/                     # figuras geradas pelos scripts, por parte
data/MOT17/                  # dados baixados (fora do git, ver .gitignore)
notebooks/
  inferencia.ipynb            # entrega final (Parte 4/5)
checkpoints/                  # pesos do modelo temporal
AI_LOG.md                     # uso de IA no projeto
```

## Comandos

Testes (Parte 0 e 1):

```bash
python -m pytest tests/ -v
```

Artefatos completos da Parte 0 (gerador, simulador de detector, piso fácil,
sweep de dificuldade):

```bash
python scripts/part0_synthetic_experiments.py
```

Dados do MOT17 (baixa só o subconjunto necessário, ~10MB de texto + imagens
das 4 sequências via range-request, sem puxar o zip de 5.5GB inteiro):

```bash
python scripts/download_mot17_subset.py --text-only   # rápido: gt.txt/det.txt/seqinfo.ini
python scripts/download_mot17_subset.py                # imagens (mais lento)
```

Parte 1 — baseline por quadro no MOT17 real (escolha do detector público,
associação ingênua, métricas, gráfico do descolamento):

```bash
python scripts/part1_baseline.py
```

Parte 1 — segunda fonte de detecção (torchvision Faster R-CNN) e comparação
com a fonte pública padrão (roda só nas sequências cujas imagens já foram
baixadas; pode ser reexecutado conforme o download avança):

```bash
python scripts/part1_torchvision_detector.py
```

(demais comandos — treino e avaliação do modelo temporal — serão adicionados
conforme a Parte 2 for implementada.)

## Decisões registradas

- **Trilha da Parte 2:** A (RNN como modelo de movimento).
- **Eixo da Parte 3 (ablação):** célula recorrente (RNN simples vs. LSTM vs. GRU,
  variando T do BPTT truncado).
- **Teste de estresse da Parte 5:** degradação do detector (reaproveita o
  simulador da Parte 0).
- **Dados do MOT17:** subconjunto de 4 sequências (não o pacote completo de
  ~5.5GB), baixadas via range-request HTTP para evitar baixar as outras 3
  sequências de treino + o conjunto de teste inteiro:
  - `MOT17-02` (treino): câmera parada, densidade 31.0 ped/quadro, 600 quadros.
  - `MOT17-09` (validação): câmera parada, densidade 10.1 ped/quadro, 525 quadros
    — mesma classe de câmera do treino, densidade/viewpoint diferentes, usada
    só para ajuste de hiperparâmetro.
  - `MOT17-04` (diagnóstico extra, não usada em treino/tuning): câmera parada,
    densidade 45.3 ped/quadro (a mais densa das 7 sequências de treino do
    MOT17), 1050 quadros — só entra no gráfico de dificuldade da Parte 1.
  - `MOT17-11` (**teste, held-out**): câmera **em movimento**, densidade
    10.5 ped/quadro, 900 quadros — nunca tocada em treino/tuning. Critério de
    split: tipo de movimento de câmera (parada vs. móvel), o eixo que o
    próprio enunciado sugere; garante que o modelo temporal é avaliado em uma
    dinâmica de câmera que ele nunca viu.
- **Detector público padrão (Parte 1, item 1):** `SDP`. Escolhido por AP
  (Average Precision, IoU≥0.5) contra o gt nas 4 sequências — SDP tem o maior
  AP em **todas** as 4, não só na média (DPM: AP médio 0.454; FRCNN: 0.497;
  SDP: 0.635). Ver `scripts/part1_baseline.py` e `outputs/part1/`.

## Parte 1 — regra de associação e gestão de tracks

A associação é a mesma usada no "piso fácil" da Parte 0
(`pa2/baseline_tracker.NaiveIoUTracker`), agora rodada sobre as detecções
reais do MOT17:

1. A cada quadro `t`, calcula-se a IoU entre a última caixa observada de
   cada track viva (de `t-1`, ou mais antiga se a track está "congelada" sem
   observação) e as detecções do quadro `t`.
2. O casamento é resolvido pelo algoritmo húngaro (ótimo global do quadro),
   com corte por limiar fixo de IoU (`iou_threshold=0.3`) — pares abaixo do
   limiar não são aceitos mesmo que sejam a melhor opção restante.
3. Detecção que não casou com nenhuma track vira uma **track nova** (ID
   nunca usado antes).
4. Track que não casou com nenhuma detecção soma 1 ao seu contador de
   quadros sem observação; ao ultrapassar `max_age=5` quadros consecutivos
   sem observação, a track **morre** (é removida; o ID nunca mais é
   reutilizado).
5. Não há nenhum modelo de movimento: a "posição esperada" de uma track no
   quadro seguinte é simplesmente a última caixa observada (hipótese de
   velocidade zero). É exatamente essa hipótese que a Parte 2 substitui.

Esses dois hiperparâmetros (`iou_threshold`, `max_age`) foram fixados a
partir do piso fácil da Parte 0 e não foram re-ajustados para o MOT17 — o
objetivo da Parte 1 é expor o fracasso dessa regra simples em dado real
(ver `outputs/part1/01_descolamento.png`), não otimizá-la.
