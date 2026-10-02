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

Parte 2 — pré-treino no sintético, treino no MOT17, e avaliação/comparação
com a Parte 1 (nessa ordem; cada um depende do checkpoint do anterior):

```bash
python scripts/part2_pretrain_synthetic.py   # -> checkpoints/motion_gru_pretrain_synthetic.pt
python scripts/part2_train_mot17.py          # -> checkpoints/motion_gru_mot17.pt
python scripts/part2_evaluate.py             # métricas + figuras de comparação
```

Parte 3 — ablação da célula recorrente (Eixo 1). Progresso salvo
incrementalmente em `outputs/part3/sweep_results.json`: rodar de novo
retoma de onde parou, em vez de refazer tudo.

```bash
python scripts/part3_ablation.py
```

Parte 4 — galeria de falhas, horizonte de memória (analítico + empírico),
e a correção:

```bash
python scripts/part4_failure_gallery.py
python scripts/part4_memory_horizon.py
python scripts/part4_correction.py
```

Parte 5 — teste de estresse de qualidade do detector:

```bash
python scripts/part5_detector_stress_test.py
```

Com isso, todas as partes do enunciado (0 a 5) estão implementadas.

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

## Parte 2 — Trilha A: RNN como modelo de movimento

**Arquitetura** (`pa2/models/motion_rnn.py`): um `GRUCell` por track.
Entrada a cada quadro = caixa `(x,y,w,h)` normalizada pelo tamanho da
imagem + confiança (score do detector na inferência; visibilidade do gt no
treino, ver justificativa em `pa2/trajectories.py`). A saída é um
**resíduo** somado à caixa de entrada (previsão "tipo velocidade", melhor
condicionada numericamente que prever posição absoluta). Perda smooth-L1
sobre a caixa; sem incerteza/log-verossimilhança por enquanto.

**Oclusão.** Quando uma track não tem observação real num quadro, o estado
roda "para frente" alimentando a **própria previsão anterior** (não a caixa
verdadeira) com confiança 0 — isso é implementado por uma única função
(`MotionGRU.rollout`) usada de forma idêntica no treino (decidindo via
limiar de visibilidade do gt se um quadro "conta como observado") e na
inferência via `pa2/motion_tracker.MotionRNNTracker` (decidindo pela
presença de uma detecção casada).

**Teste de regressão:** com a cabeça de saída recém-inicializada (zerada,
delta=0 sempre), o `MotionRNNTracker` reproduz **exatamente** a saída do
`NaiveIoUTracker` da Parte 1 nas mesmas detecções — confirma que a troca de
"última posição observada" por "posição prevista" foi implementada
corretamente, antes de qualquer treino real entrar em cena
(`tests/test_motion_tracker.py`).

**Treino em duas etapas:**
1. Pré-treino no sintético da Parte 0 (`scripts/part2_pretrain_synthetic.py`,
   60 vídeos de treino + 10 de validação nunca vistos, 200 épocas). Achado
   honesto: no cenário sintético sem ruído, o objeto roteirizado se move em
   linha reta perfeita, então uma extrapolação de velocidade constante tem
   erro ~0 — o GRU aprendido fica *atrás* até do baseline trivial de
   "posição congelada" nesse caso específico (ver AI_LOG.md). O objetivo
   desta etapa não era bater esse número, e sim confirmar que o modelo
   aprende a tarefa certa antes de ir pro MOT17 real.
2. Treino no MOT17 (`scripts/part2_train_mot17.py`): trajetórias de gt do
   MOT17-02 (treino, 57 trajetórias), validação em MOT17-09 — a mesma
   divisão de sequências da Parte 1. Começa do checkpoint sintético (warm
   start). Overfitting claro depois da época ~25 (treino quase zero, val
   piora) — mitigado com early stopping (guarda a melhor época por
   validação) + weight decay leve. Checkpoint final:
   `checkpoints/motion_gru_mot17.pt`.

**Avaliação e comparação com a Parte 1** (`scripts/part2_evaluate.py`,
mesma fonte de detecção SDP, mesmas 4 sequências,
`outputs/part2/04_comparacao_parte1_vs_parte2.png`):

| Sequência | IDF1 baseline | IDF1 MotionGRU | switches baseline | switches MotionGRU |
|---|---|---|---|---|
| MOT17-09 | 0.476 | 0.493 | 56 | 63 |
| MOT17-11 | 0.574 | 0.572 | 140 | 144 |
| MOT17-02 | 0.355 | 0.323 | 334 | 365 |
| MOT17-04 | 0.647 | 0.655 | 199 | 210 |

**O resultado é honesto: a melhora é pequena ou nula** (e até piora no
MOT17-02). Investigamos a causa mecanística (`outputs/part2/05_deslocamento_relativo_mot17.png`):
medimos o deslocamento mediano do centro da caixa entre quadros
consecutivos, relativo à própria largura da caixa — a mesma grandeza que o
sweep de dificuldade da Parte 0 varia no eixo "velocidade". No sweep
sintético, a velocidade mais baixa testada (0.5 px/quadro, onde o baseline
ingênuo já ia bem, IDF1~0.82) corresponde a um deslocamento relativo de
0.030; o ponto onde o baseline começa a quebrar de verdade (IDF1~0.55) fica
em 0.153. **As 4 sequências do MOT17 medem entre 0.010 e 0.037** — todas no
regime "fácil" do sweep, nenhuma chega perto do regime onde velocidade
quebra o tracker ingênuo. Pedestres reais a 30fps simplesmente não se
deslocam o suficiente por quadro, relativo ao próprio tamanho, pra a
hipótese de velocidade zero ser um problema sério. O gargalo que a Parte 1
expôs (switches/fragmentação alta em cenas densas) vem de outro lugar —
ambiguidade de identidade entre pessoas próximas e morte de track por
oclusão longa — problemas que um modelo de **movimento puro** (sem
informação de aparência, a Trilha B que não escolhemos) não ataca.

**Teste direcionado: dá pra demonstrar o mecanismo em dado real?**
(`scripts/part2_frame_subsampling_demo.py`,
`outputs/part2/06_subamostragem_frame_rate.png`). Se o motivo do ganho ser
pequeno é "o MOT17 vive no regime fácil do sweep", a previsão natural é:
empurrando o MOT17 pro regime difícil (subamostrando quadros — mantém 1 a
cada *k*, o que multiplica o deslocamento real por quadro por ~*k*, sem
retreinar nada), o MotionGRU deveria abrir vantagem clara sobre o baseline.

**Essa previsão NÃO se confirmou.** Testamos *k* ∈ {1,2,3,4,5,8} nas 4
sequências: o resultado é ruidoso, sem vantagem crescente e consistente
pro MotionGRU — no MOT17-11, ele chega a ficar bem pior no *k* mais
agressivo (IDF1 -0.060 vs. baseline). Duas razões, ambas informativas:

1. O MotionGRU não recebe ∆t como entrada (decisão da Parte 2) e foi
   treinado só no MOT17-02 (a sequência de MENOR deslocamento nativo, 0.010).
   Ele aprendeu a prever resíduos calibrados pra esse deslocamento mínimo, e
   não tem nenhum mecanismo pra reescalar a extrapolação quando o ∆t efetivo
   muda — ele continua prevendo resíduos pequenos mesmo quando o objeto
   realmente andou mais, continuando perto de "velocidade quase zero" errado.
2. Subamostrar dado real mexe em três eixos ao mesmo tempo (desloca mais,
   reduz o nº de quadros, muda a duração efetiva de cada oclusão em
   quadros), diferente do sweep sintético da Parte 0 que isolava só a
   velocidade. Prova disso: nem o baseline sozinho degrada de forma limpa
   com *k* (ex. MOT17-09: 0.476→0.508→0.504→0.443→...).

**Isso não invalida o mecanismo — refina a resposta.** O enunciado da
Parte 5 (opcional, não escolhida) pergunta exatamente isso: *"por que um
modelo de movimento aprendido em ∆t fixo quebra quando ∆t muda? Alimentar
∆t na recorrência resolveria?"*. Este teste é evidência empírica direta
de que sim, quebra — e exatamente pela ausência de ∆t na entrada, uma
limitação de arquitetura já conhecida, não uma falha de implementação.

## Parte 3 — Ablação, Eixo 1: a célula recorrente

Escolhemos o **Eixo 1** (célula recorrente): RNN simples vs. LSTM vs. GRU,
no mesmo orçamento aproximado de parâmetros, variando o comprimento da
janela de BPTT truncado `T ∈ {4, 8, 16, 32}`, com 3 seeds por configuração
(36 combinações no total). Rodado no **sintético da Parte 0** (não no
MOT17): a pergunta é sobre o comportamento estrutural da célula sob
dependências de longo alcance, e só o gerador sintético deixa controlar
essa dependência (duração de oclusão) de forma limpa.

**Generalização do modelo** (`pa2/models/motion_rnn.py`): o `MotionGRU` da
Parte 2 virou `MotionRNN(cell_type=...)`, suportando as 3 células atrás da
mesma interface — o estado é sempre uma tupla opaca (`(h,)` ou `(h,c)` pra
LSTM), que o tracker nunca abre. `MotionGRU(...)` continua existindo como
alias de compatibilidade. Todos os testes da Parte 2 foram parametrizados
pras 3 células e continuaram passando sem nenhuma mudança de expectativa.

**Orçamento de parâmetros equalizado** (`matched_hidden_size`, dentro de
~0.6% do alvo):

| Célula | hidden_size | parâmetros da célula |
|---|---|---|
| RNN simples | 113 | 13.560 |
| LSTM | 55 | 13.640 |
| GRU | 64 | 13.632 |

**Resultado** (`outputs/part3/01_ablacao_celula_recorrente.png`,
`outputs/part3/sweep_results.json` com o progresso completo das 36
combinações), duas métricas por configuração:

| Célula | T | perda de validação | erro na oclusão longa (24 quadros) |
|---|---|---|---|
| RNN | 4 | 0.000049 ± 0.000000 | 0.0450 ± 0.0000 |
| RNN | 8 | 0.000069 ± 0.000000 | 0.0445 ± 0.0000 |
| RNN | 16 | 0.000098 ± 0.000001 | 0.0365 ± 0.0007 |
| RNN | 32 | 0.000122 ± 0.000000 | 0.0293 ± 0.0005 |
| LSTM | 4 | 0.000049 ± 0.000000 | 0.0450 ± 0.0000 |
| LSTM | 8 | 0.000069 ± 0.000000 | 0.0448 ± 0.0002 |
| LSTM | 16 | 0.000099 ± 0.000000 | 0.0382 ± 0.0004 |
| LSTM | 32 | 0.000121 ± 0.000001 | 0.0297 ± 0.0003 |
| GRU | 4 | 0.000049 ± 0.000000 | 0.0450 ± 0.0000 |
| GRU | 8 | 0.000069 ± 0.000000 | 0.0438 ± 0.0003 |
| GRU | 16 | 0.000097 ± 0.000000 | 0.0339 ± 0.0005 |
| GRU | 32 | 0.000119 ± 0.000000 | **0.0275** ± 0.0003 |

A perda de validação (mesma janela do treino) não diferencia as células em
nenhum T. O erro na sonda de oclusão longa (fixa, 24 quadros, maior que a
maioria dos T testados) sim diferencia, e a vantagem do GRU **cresce** com
T — exatamente onde a dependência de longo alcance passa a importar.

**Resposta à pergunta específica** ("onde a RNN simples quebra, bate com a
história de gradiente que some?"): sim. A curva direta de
`||∂L/∂h_{t-k}||` (`outputs/part3/02_gradiente_que_some.png`, T=32,
checkpoints seed=0) mostra a RNN simples caindo de ~10⁻³ a **zero
numérico** em 16 passos pra trás, enquanto o GRU mantém gradiente
mensurável até os 30 passos inteiros — bate com o mecanismo exato dos
slides de aula (RNN simples: `h_t = tanh(W·[h_{t-1};x_t])`, o gradiente
envolve multiplicar por `W^T` repetidamente, decaindo geometricamente se o
maior valor singular de `W` é menor que 1; GRU/LSTM: o caminho dominante é
multiplicação elemento-a-elemento por um portão aprendido, não uma
potência de matriz fixa, evitando o colapso).

**Obstáculo de engenharia real, documentado:** a primeira tentativa de
rodar as 36 combinações parecia travar — o processo tinha acumulado
18.063s de CPU em só 45min de relógio. Causa: o PyTorch paraleliza
automaticamente em várias threads mesmo operações minúsculas (o modelo
inteiro tem ~14k parâmetros, rodado passo a passo num laço Python), e o
custo de sincronizar threads ficava maior que a conta em si. Corrigido com
`torch.set_num_threads(1)` no topo do script (confirmado: uso de CPU foi
de quase 0% pra ~96%). O script também salva o progresso incrementalmente
em `sweep_results.json`, retomando de onde parou se interrompido.

## Parte 4 — Galeria de falhas e horizonte de memória

### Galeria de falhas (3 trechos, minerados automaticamente)

`scripts/part4_failure_gallery.py` roda o `MotionRNNTracker` (modelo final,
`max_age=5`, a config "antes" da correção) de verdade no MOT17-02 com
detecção SDP, e reaproveita a lógica interna de casamento quadro-a-quadro
do `metrics.py` (`_match_frame_with_continuity`) pra achar **exatamente**
em que quadro e entre quais identidades cada ID switch acontece — não
garimpado a olho.

| Falha | Figura | Diagnóstico |
|---|---|---|
| 1 — oclusão longa | `outputs/part4/04a_falha_oclusao_longa.png` | Objeto gt=3 fica sem detecção por **191 quadros** (≫ max_age=5) — a track morre por regra de nascimento/morte antes mesmo do horizonte de memória do modelo (k≈7, ver abaixo) virar o fator limitante; ao reaparecer, vira ID novo. |
| 2 — troca por proximidade | `outputs/part4/04b_falha_troca_proximidade.png` | Dois objetos reais (gt 32, gt 33) passam perto o suficiente que o casamento por IoU troca os rótulos **no mesmo quadro, sem nenhum buraco de detecção envolvido** — mecanismo diferente: gargalo de identidade em cena densa (já diagnosticado na Parte 2), não de memória. |
| 3 — oclusão curta já custa caro | `outputs/part4/04c_falha_oclusao_curta.png` | Um buraco de só **14 quadros** (perto da mediana real do dataset, 15) já troca o ID — mostra que a margem é pequena mesmo fora do caso extremo. |

### Horizonte de memória efetivo — analítica

`scripts/part4_memory_horizon.py`, `outputs/part4/01_horizonte_analitico.png`:
`||∂L/∂h_{t-k}||` no **modelo final** (`motion_gru_mot17.pt`), numa trajetória
real do MOT17-02. Horizonte efetivo (gradiente cai abaixo de 1% do valor em
k=0): **k=7**. Isso é bem menor que o k≈30 do GRU medido na ablação da
Parte 3 (que foi treinado com janela T=32) — o horizonte aprendido parece
acompanhar a **janela de treino** (nossa Parte 2 usou T=16), não só o tipo
de célula.

Como fizemos o Eixo 1 da Parte 3, reaproveitamos a comparação RNN simples
vs. modelo com portas, na mesma janela, que já tínhamos construído lá —
os checkpoints já existiam: `outputs/part3/02_gradiente_que_some.png`
(T=32, checkpoints seed=0) mostra a RNN simples caindo a zero numérico em
16 passos contra o GRU sobrevivendo até os 30 passos inteiros.

### Horizonte de memória efetivo — empírica

`outputs/part4/02_horizonte_empirico.png`: taxa de sobrevivência de
identidade vs. duração da oclusão, com o tracker de verdade. Para isolar a
variável (duração) sem confundir com velocidade relativa — a oclusão
*roteirizada* do gerador da Parte 0 liga as duas por construção —, aqui a
gente **força um buraco de detecção de duração exata**, num objeto que se
move devagar o tempo todo.

Resultado: com `max_age=5` (atual), a sobrevivência despenca a 0 exatamente
em `d=6` (passou do limite) e nunca mais sobe. Com `max_age=15` ou `25`, a
sobrevivência se estende bem mais, mas as duas curvas quase coincidem entre
si — um **segundo limite** aparece (deriva da previsão em free-running),
distinto do `max_age`. Comparado com a distribuição **real** de duração de
oclusão do MOT17 (mediana = 15 quadros; **78%** dos eventos duram mais que
o `max_age=5` atual).

### A correção

`scripts/part4_correction.py`, `outputs/part4/03_correcao_antes_depois.png`:
diagnóstico → `max_age=5` é pequeno demais pro MOT17 real. Correção:
`max_age=15` (não 25 — a sonda empírica mostrou que 25 mal ajuda mais que
15, satura no segundo limite acima; 15 já casa com a mediana real de
oclusão do dataset).

Antes/depois, **mesmo checkpoint**, mesmas 4 sequências, só o `max_age` do
tracker muda:

| Sequência | IDF1 antes | IDF1 depois | razão ids antes | razão ids depois |
|---|---|---|---|---|
| MOT17-09 | 0.493 | 0.499 | 3.58 | 2.69 |
| MOT17-11 | 0.572 | 0.581 | 2.64 | 2.29 |
| MOT17-02 | 0.323 | 0.343 | 7.60 | 5.55 |
| MOT17-04 | 0.655 | 0.690 | 2.57 | 1.81 |

IDF1 melhora nas 4 sequências (+0.006 a +0.036), e a razão de identidades
cai bastante em todas. Efeito colateral honesto: fragmentações **sobem**
um pouco em 3 das 4 sequências — deixar tracks "penduradas" por mais tempo
cria mais oportunidades de transição tracked→untracked→tracked, mesmo
quando a identidade final continua certa.

## Parte 5 — Teste de estresse: qualidade do detector

Escolhemos **qualidade do detector** (não queda de frame rate). Feito **sem
retreinar**, em cima do modelo final (`motion_gru_mot17.pt`, `max_age=15`
já com a correção da Parte 4). Reaproveita `pa2.synthetic.simulate_detections`
(construído na Parte 0 — o próprio enunciado avisa que esse é "exatamente
um experimento da Parte 5"): em vez do det.txt público (ruído fixo, não
controlável), geramos detecções a partir do `gt.txt` + visibilidade real do
MOT17, com ruído que a gente controla, em **3 intensidades** (leve/média/
severa — `drop_prob`, ruído de coordenada/tamanho e taxa de falsos
positivos crescentes), 3 seeds cada, nas 4 sequências.

*(figura: `outputs/part5/01_degradacao_detector.png`)* — mAP (detecção) e
IDF1 (baseline ingênuo vs. MotionGRU) praticamente coincidem em quase todas
as 12 combinações (sequência × intensidade): conforme o detector piora, as
três curvas caem juntas.

*(figura: `outputs/part5/02_absorve_ou_amplifica.png`)* — pergunta do
enunciado: o modelo temporal **absorve ou amplifica** a falha do detector?
Medimos a razão `(queda de IDF1) / (queda de mAP)` do cenário limpo pro
severo: um valor **abaixo de 1** significa que o IDF1 cai menos que o mAP
(absorve); **acima de 1**, cai mais (amplifica).

| Sequência | razão baseline | razão MotionGRU |
|---|---|---|
| MOT17-09 | 0.95 | 0.94 |
| MOT17-11 | 0.92 | 0.94 |
| MOT17-02 | 0.97 | 0.97 |
| MOT17-04 | 1.07 | 1.09 |

Em 3 das 4 sequências, a razão fica perto de 1 pras duas trackers — nem
absorve nem amplifica de forma marcante, a identidade degrada
proporcionalmente à detecção. A exceção é o MOT17-04 (a sequência mais
densa): ali a falha do detector **amplifica** um pouco (razão > 1) pras
duas, e o MotionGRU amplifica marginalmente mais que o baseline (1.09 vs.
1.07) — consistente com o achado da Parte 2 de que cenas densas são onde
qualquer ruído extra (de detecção ou de um resíduo de movimento aprendido)
tem mais chance de confundir o casamento com o vizinho errado.
