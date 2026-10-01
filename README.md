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
  part0_metric_cases_figure.py       # figura quadro-a-quadro dos 3 casos de métrica (item 3)
  part1_baseline.py                  # escolha do detector público + baseline + gráfico obrigatório
  part1_torchvision_detector.py      # 2ª fonte de detecção (torchvision) + comparação
  part1_summary_figures.py           # gráficos de barra resumindo a escolha de detector
outputs/                     # figuras geradas pelos scripts, por parte
data/MOT17/                  # dados usados pelo código (fora do git, ver .gitignore)
data/MOT17_SITE/              # extração bruta do MOT17.zip completo, se baixado manualmente
                               # (fora do git; só serve como fonte pra preencher data/MOT17/,
                               # nenhum script lê daqui diretamente)
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

Dados do MOT17 — duas formas, qualquer uma deixa `data/MOT17/` no estado que
o código espera:

```bash
# opção A: baixa só o subconjunto necessário via range-request HTTP, sem
# puxar o MOT17.zip de 5.5GB inteiro (mais lento por causa de outages
# intermitentes do servidor, mas baixa só ~1% do volume do pacote completo)
python scripts/download_mot17_subset.py --text-only   # rápido: gt.txt/det.txt/seqinfo.ini
python scripts/download_mot17_subset.py                # imagens (mais lento)

# opção B: baixe o MOT17.zip completo manualmente (motchallenge.net/data/MOT17/)
# e extraia em data/MOT17_SITE/; depois copie as img1/ das 4 sequências usadas
# (MOT17-02-FRCNN, MOT17-09-FRCNN, MOT17-04-FRCNN, MOT17-11-FRCNN) para dentro
# de data/MOT17/{train,val,diagnostic,test}/MOT17-XX/img1/ respectivamente.
```

Usamos as duas: começamos pela opção A (mais econômica), e quando o servidor
ficou instável no meio do download, a opção B (baixar o pacote completo e
copiar o que faltava) terminou de preencher as imagens que faltavam.

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
  SDP: 0.635). Ver `scripts/part1_baseline.py`, `scripts/part1_summary_figures.py`
  e `outputs/part1/03_detector_publico_comparacao.png`.

## Parte 1 — resultados

**Segunda fonte de detecção (torchvision Faster R-CNN, classe person do COCO)
vs. SDP**, AP@0.5 em 40 quadros amostrados por sequência (mesma amostra pras
duas fontes; ver `outputs/part1/04_torchvision_vs_sdp_resumo.png` e as
figuras `02_torchvision_vs_publico_MOT17-*.png`):

| Sequência | AP torchvision | AP SDP |
|---|---|---|
| MOT17-02 | 0.398 | 0.407 |
| MOT17-09 | **0.704** | 0.656 |
| MOT17-04 | 0.553 | **0.748** |
| MOT17-11 | 0.657 | **0.747** |

O detector genérico (nunca ajustado pro MOT17) empata, às vezes ganha
(MOT17-09) e às vezes perde com folga (MOT17-04, a cena mais densa) do
detector "nativo" do dataset — o resultado depende da cena, não é uma
vitória uniforme de nenhum dos dois lados.

**Avaliação como trajetórias** (detector SDP, associação ingênua da Parte 0,
ordenado por densidade; ver `outputs/part1/01_descolamento.png`):

| Sequência | câmera | densidade | AP | IDF1 | switches | frags | ids pred/gt | erro de contagem (análogo PA1) | razão |
|---|---|---|---|---|---|---|---|---|---|
| MOT17-09 | parada | 10.1 | 0.649 | 0.476 | 56 | 152 | 93/26 | **+67** | 3.58x |
| MOT17-11 | movimento | 10.5 | 0.738 | 0.574 | 140 | 237 | 194/75 | **+119** | 2.59x |
| MOT17-02 | parada | 31.0 | 0.406 | 0.355 | 334 | 836 | 455/62 | **+393** | 7.34x |
| MOT17-04 | parada | 45.3 | 0.747 | 0.647 | 199 | 873 | 210/83 | **+127** | 2.53x |

"Erro de contagem" aqui é `n_ids_previstas - n_ids_verdadeiras` (o análogo
temporal direto do erro de contagem de instâncias do PA1): o tracker ingênuo
sempre **superconta** identidades (nunca subconta), porque toda vez que uma
track morre e um objeto reaparece, vira uma identidade nova — nunca o
contrário.

Dois achados centrais: (1) em **toda** sequência o IDF1 fica sistematicamente
abaixo do mAP — boa detecção por quadro não implica identidade boa ao longo
do tempo; (2) densidade sozinha não prediz bem a dificuldade (MOT17-02 teve
o pior resultado das 4, pior até que MOT17-04, que é mais densa ainda) —
outros fatores da cena pesam tanto ou mais que a contagem bruta de pessoas.

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
