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
metrics.py              # IDF1, ID switches, fragmentações (implementação própria)
pa2/
  synthetic.py           # gerador de vídeo sintético + simulador de detector (Parte 0)
  nms.py                 # NMS implementado do zero
  association.py         # matching guloso / húngaro por IoU
  baseline_tracker.py    # tracker ingênuo por quadro (Partes 0 e 1)
  mot17.py               # leitura de det.txt / gt.txt do MOT17 (Parte 1+)
  models/                # modelo temporal recorrente (Parte 2+)
tests/
  test_metrics.py        # os 3 casos de teste construídos à mão da Parte 0
scripts/
  part0_synthetic_experiments.py  # gera todos os artefatos da Parte 0
outputs/                 # figuras geradas pelos scripts, por parte
notebooks/
  inferencia.ipynb        # entrega final (Parte 4/5)
checkpoints/              # pesos do modelo temporal
AI_LOG.md                 # uso de IA no projeto
```

## Comandos

Testes de métrica (Parte 0, item 3):

```bash
python -m pytest tests/ -v
```

Artefatos completos da Parte 0 (gerador, simulador de detector, piso fácil,
sweep de dificuldade):

```bash
python scripts/part0_synthetic_experiments.py
```

(demais comandos — treino e avaliação do modelo temporal — serão adicionados
conforme as Partes 1 e 2 forem implementadas.)

## Decisões registradas

- **Trilha da Parte 2:** A (RNN como modelo de movimento).
- **Eixo da Parte 3 (ablação):** célula recorrente (RNN simples vs. LSTM vs. GRU,
  variando T do BPTT truncado).
- **Teste de estresse da Parte 5:** degradação do detector (reaproveita o
  simulador da Parte 0).
- **Dados do MOT17:** subconjunto pequeno de sequências (não o pacote completo
  de ~5.5GB), para caber em CPU.
