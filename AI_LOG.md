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

**Download instável, resolvido com uma segunda via.** O servidor do
motchallenge.net teve outages intermitentes no meio do download via
range-request (timeouts de conexão, não só quedas de conexão já aberta).
Primeiro deixei o script mais resiliente (timeout explícito de 30s em vez
de `None`, até 6 tentativas com backoff exponencial, idempotente por imagem
dentro de um lote, uma sequência falhando não derruba as outras). Mesmo
assim, pra não ficar refém da instabilidade do servidor, baixei também o
`MOT17.zip` completo manualmente (a opção "crua", sem o truque de
range-request) e copiei as pastas `img1/` que faltavam
(`data/MOT17_SITE/MOT17/train/MOT17-{04,11}-FRCNN/img1/` ->
`data/MOT17/{diagnostic,test}/MOT17-{04,11}/img1/`). Isso completou as 4
sequências (900/900 imagens no MOT17-11, que antes tinha zero) sem precisar
esperar o range-request terminar. Lição: tinha esquecido de generalizar o
`.gitignore` pra cobrir `data/MOT17_SITE/` também (só tinha `data/MOT17/`
explicitamente) — percebi e troquei pra ignorar `data/` inteiro antes de
qualquer commit, senão teria tentado versionar ~5.7GB de imagem.

Com as 4 sequências completas, rodei a comparação torchvision-vs-SDP nas 4
(antes só tinha MOT17-02). Resultado mais rico do que o esperado: o
detector genérico não é uniformemente pior nem melhor — empata no MOT17-02
(0.398 vs 0.407), **ganha** no MOT17-09 (0.704 vs 0.656), e **perde** com
folga no MOT17-04 e MOT17-11 (0.553 vs 0.748; 0.657 vs 0.747). Isso vira um
bom ponto de apresentação: a vantagem de um detector "nativo" do dataset
(treinado/calibrado nessas cenas específicas) aparece justamente nas cenas
mais densas/difíceis, não nas mais fáceis.

Com isso a Parte 1 está completa nos 5 itens pedidos.

## Parte 2 — RNN de movimento (Trilha A)

**Decisões de arquitetura, antes de escrever qualquer linha de treino:**
entrada = caixa (x,y,w,h normalizada) + confiança; perda só smooth-L1 na
caixa (sem incerteza por enquanto — fica como extensão futura se sobrar
tempo); célula GRU (menos parâmetro que LSTM, mais rápida em CPU); e
pré-treinar no sintético da Parte 0 antes de arriscar tempo de CPU no
MOT17 — mesmo princípio da Parte 0 inteira.

**Bug pego pelos próprios testes antes de qualquer treino real.** A
primeira versão do `rollout()` (a função que decide, quadro a quadro, se
alimenta a observação real ou a própria previsão anterior — o mecanismo de
"rodar para frente sem observação" sob oclusão) usava `observed_mask.all()`
pra decidir isso, o que colapsa a decisão do BATCH INTEIRO numa amostra só.
Pedi pra escrever o teste `test_rollout_decisao_e_por_amostra_no_batch`
ANTES de confiar na função (duas amostras no mesmo batch com máscaras
diferentes), ele pegou o bug de cara, e só aceitei a correção depois de ver
o teste passar.

**Escolha de usar visibilidade do gt como "confiança" no treino.** Como o
enunciado pede treinar "em trajetórias do ground truth" (não em detecções
reais), não existe um score de detector de verdade disponível pra alimentar
o modelo durante o treino. Em vez de usar uma constante (ex. sempre 1.0),
decidi usar a visibilidade do gt como proxy: é um sinal real, variável, que
cai sob oclusão — e uso o mesmo limiar pra decidir se um quadro "conta como
observado" (vira teacher forcing) ou não (vira free-running), ligando
diretamente esse mecanismo de treino ao conceito de oclusão que já
construímos na Parte 0.

**Resultado do pré-treino sintético — achado honesto.** Com 40 épocas, o
modelo já acompanhava bem a trajetória ANTES da oclusão, mas durante a
janela de oclusão (free-running) ele extrapolava na direção ERRADA antes de
"corrigir" de golpe assim que a observação real voltava (erro médio de 8.53
px). Com 200 épocas, melhorou bastante (2.76 px), mas ainda não supera nem
o baseline trivial de "manter a posição congelada" (2.40 px) nesse cenário
específico — e fica muito atrás de uma extrapolação de velocidade
constante, que nesse caso dá erro ~0 porque o objeto roteirizado se move em
LINHA RETA PERFEITA, sem ruído nenhum (é assim que o gerador da Parte 0
constrói a oclusão). Isso bate exatamente com o que o próprio enunciado
avisa: "[o filtro de Kalman de velocidade constante] é um baseline honesto
e frequentemente difícil de bater". Decidi não ficar perseguindo bater esse
número no sintético perfeitamente limpo — o objetivo desta etapa era só
confirmar que o modelo aprende a tarefa certa (e aprende: ele claramente
tenta extrapolar movimento, não só congela), e pedestres de verdade no
MOT17 não se movem em linha reta perfeita, então é lá que um modelo não
linear aprendido tem chance real de ganhar de um filtro linear simples.

**Avaliação final — achado mais importante da Parte 2.** Depois de treinar
no MOT17-02 e avaliar nas 4 sequências com a mesma fonte de detecção (SDP)
da Parte 1, a melhora de IDF1 foi pequena ou nula (e piorou no MOT17-02).
Antes de aceitar isso como "não funcionou" sem entender por quê, pedi pra
investigar a causa mecanística. A hipótese: será que pedestres de verdade
se movem rápido o suficiente, por quadro, pra um modelo de movimento fazer
diferença? Medimos o deslocamento do centro da caixa entre quadros
consecutivos relativo à própria largura (a mesma grandeza que o sweep de
velocidade da Parte 0 varia) nas 4 sequências do MOT17 real, e comparamos
com os mesmos vídeos sintéticos do sweep. Resultado: os pedestres do MOT17
se deslocam entre 1% e 3.7% da própria largura por quadro — isso é **igual
ou menor** que a velocidade mais BAIXA testada no sweep sintético (0.5
px/quadro = 3.0% de deslocamento relativo), onde o próprio Parte 0 já tinha
mostrado que o tracker ingênuo vai bem (IDF1~0.82). Nenhuma das 4 sequências
chega perto do regime onde o sweep mostrou o baseline quebrar de verdade
(IDF1~0.55, deslocamento relativo 15.3%).

Isso muda completamente a interpretação do resultado: não é que o MotionGRU
"não funcionou" — é que, a 30fps, a hipótese de velocidade zero do tracker
ingênuo já era quase ótima pros pedestres do MOT17, então não havia muito
espaço pra um modelo de movimento melhorar. O gargalo real que a Parte 1
expôs (switches e fragmentação explodindo em cenas densas) vem de outro
lugar: ambiguidade de identidade entre pessoas próximas e morte de track
por oclusão longa — nenhum dos dois é resolvido por um modelo que só prevê
POSIÇÃO, sem nenhuma informação de APARÊNCIA (exatamente o que a Trilha B,
que não escolhemos, atacaria). Isso vira um ponto central e defensável da
apresentação: entender os limites do que a trilha escolhida pode resolver é
tão importante quanto o número de IDF1 em si.

## Próximas entradas

Vamos continuar registrando aqui conforme avançamos para a Parte 3
(ablação da célula recorrente) em diante.
