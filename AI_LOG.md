# AI_LOG

Registro de como usamos IA (Claude Code, modelo Claude Sonnet 5) ao longo do
PA2. Atualizado conforme avançamos pelas partes do enunciado.

## Como trabalhamos

O projeto inteiro foi feito em par com o Claude Code: eu (aluno) discuto a
abordagem, tomo as decisões que o enunciado pede que sejam nossas (trilha da
Parte 2, eixo da ablação, teste de estresse, critério de split, limiares),
reviso cada arquivo gerado, rodo os testes/scripts localmente e peço ajuste
quando o resultado não é o que eu esperava ou não entendo alguma escolha.
Nada foi aceito sem eu entender o porquê — em vários pontos abaixo registro
onde pedi para reformular algo porque a primeira versão tinha um problema que
eu percebi ao revisar.

## Parte 0 — ambiente sintético e métricas

**Decisões que tomei antes de qualquer código ser escrito:** trilha A para a
Parte 2 (RNN de movimento, mais simples de depurar sem GPU e conecta direto
com a análise de horizonte de memória da Parte 4), eixo 1 para a Parte 3
(célula recorrente), teste de estresse de degradação de detector para a
Parte 5 (reaproveita o simulador que já ia construir na Parte 0), e
subconjunto pequeno do MOT17 em vez do pacote completo de 5.5GB — minha
máquina só tem PyTorch CPU (sem CUDA), confirmei isso rodando
`torch.cuda.is_available()` antes de decidir.

**IDF1.** Pedi para implementar seguindo a formulação original (Ristani et
al. 2016: matching quadro-a-quadro com preferência de continuidade +
assignment global de identidades via Hungaro), não uma versão simplificada
"primeira correspondência que aparece". O ponto crítico que verifiquei à mão:
o assignment global precisa de linhas/colunas-dummy na matriz de custo para
permitir identidades sem par (senão o scipy.linear_sum_assignment com matriz
retangular força correspondências espúrias). Conferi a matemática construindo
os 3 casos de teste manualmente ANTES de rodar o código, calculando IDF1
esperado no papel (ex: caso do swap de identidades: IDF1=0.5 porque o
assignment global só pode escolher um tracker por alvo, cobrindo exatamente
metade dos quadros de cada um) e só aceitei a implementação quando bateu
exatamente com a conta.

**Convenção de ID switch vs. fragmentação.** Essa parte exigiu mais
iteração: a definição "oficial" do CLEAR-MOT tem uma sutileza — o casamento
quadro-a-quadro deve *preferir* manter a correspondência da track viva no
quadro anterior (mesmo que não seja a de maior IoU), senão você conta
switches espúrios toda vez que duas tracks próximas trocam de "melhor
casamento" por motivo numérico. Pedi para o Claude explicar essa regra antes
de implementar, e depois validei construindo o caso (b) (troca de identidades)
onde eu sabia que o número certo de switches era 2, não 1 ou 4.

**Gerador sintético — a parte mais trabalhosa.** O requisito "oclusão
verificável" (uma elipse precisa sumir de verdade, não só sobrepor) me fez
pedir para repensar a abordagem duas vezes: a primeira tentativa usava
movimento totalmente aleatório e torcia para duas elipses se cruzarem — nem
sempre acontecia, e quando acontecia a duração da oclusão não era
controlável. A versão final resolve isso roteirizando analiticamente um par
(occluder parado, tipo poste; occluded atravessando por trás dele com
velocidade relativa derivada da duração de oclusão desejada), o que garante
matematicamente N quadros de oclusão total, independente de sorte. Também
pedi para expor a visibilidade real (fração de área não coberta, calculada
por máscara, não estimada) porque isso é exatamente o campo `visibility` do
gt.txt do MOT17 — reaproveitamos esse código depois.

**Bug que corrigi na primeira rodada:** a figura da Parte 0 saiu com os eixos
do gráfico de visibilidade sobrepostos (texto "1.0" duplicado várias vezes) —
causa: misturar `plt.subplots(2,7)` com um `plt.subplot(2,1,2)` manual depois,
que cria uma segunda grade conflitante. Pedi para trocar por `GridSpec`
explícito, que resolveu.

**O que eu não aceitei sem entender:** a fórmula de custo do assignment
húngaro em `metrics.py` (`nf_gt + nf_pred - 2*ntp`) — pedi para o Claude
explicar por que esse é exatamente IDFN+IDFP para aquele par antes de seguir
em frente, porque sem entender isso eu não ia conseguir defender o código na
apresentação.

## Parte 1 — baseline no MOT17 real

**Dados.** Descobri (pedindo pro Claude checar com `curl -I`) que o servidor
do motchallenge.net responde `Accept-Ranges: bytes`, o que permite usar a
lib `remotezip` pra ler só os arquivos específicos de dentro do
`MOT17.zip` de 5.5GB sem baixar o arquivo inteiro — baixei só 4 sequências
(texto completo + imagens) em vez do pacote inteiro. Antes de escolher
*quais* 4 sequências, pedi pra computar densidade real (pedestres/quadro)
das 7 sequências de treino a partir do `gt.txt` (não confiar só em
descrição de memória da literatura) — os números (MOT17-02: 31.0,
MOT17-04: 45.3, MOT17-09: 10.1, MOT17-11: 10.5, etc.) é que guiaram a
escolha do split, documentada no README.

**Detector público padrão.** Em vez de escolher DPM/FRCNN/SDP só por
reputação ("SDP costuma ser o melhor"), pedi pra calcular AP de verdade
(Average Precision, IoU≥0.5) contra o gt nas 4 sequências baixadas. SDP
venceu nas 4, não só na média — decisão com número, não com achismo.
Implementamos o AP do zero (`pa2/detection_metrics.py`, estilo de
interpolação "all-point" do VOC2012/COCO) com testes de sanidade antes de
confiar nele pra essa decisão.

**Detector do torchvision em CPU é lento.** Faster R-CNN pré-treinado
levou ~5.3s/quadro em CPU numa imagem 1920x1080. Rodar nas ~600-1050
imagens de cada sequência levaria horas por sequência só pra essa
comparação secundária (a fonte pública continua sendo a "fonte padrão do
resto do PA", o enunciado só pede pra também *usar* o torchvision, não pra
ele carregar o resto do projeto). Decidi (e documentei no próprio script)
amostrar 40 quadros igualmente espaçados por sequência em vez de rodar em
tudo — mantém a comparação estatisticamente razoável sem inviabilizar o
tempo. Resultado interessante pra apresentação: nos 40 quadros testados do
MOT17-02, o torchvision (AP=0.397) ficou quase empatado com o SDP
(AP=0.408), mesmo sendo um detector genérico nunca ajustado pro MOT17.

**Reaproveitamento confirmado.** O tracker ingénuo da Parte 0 (item 4,
`pa2/baseline_tracker.py`) foi usado em produção aqui sem nenhuma mudança —
só trocou a fonte dos dados (sintético -> MOT17 real). Os números pioraram
muito (IDF1 caiu pra 0.36-0.65, switches na casa das centenas pra sequências
densas) — exatamente o "fracasso" que a Parte 1 pede pra quantificar, e que
motiva a Parte 2.

## Próximas entradas

Vamos continuar registrando aqui conforme avançamos para a Parte 2 (RNN de
movimento) em diante. Falta ainda rodar a comparação torchvision-vs-público
nas outras 3 sequências assim que o download das imagens terminar.
