# emgpred: estimação de ângulos dos dedos a partir de HD-sEMG

Código da tese de doutorado: estimação contínua dos 5 ângulos de flexão dos dedos a partir de EMG
de alta densidade (128 eletrodos), com o **GlobalViewTransformer (GVT)**, em encoders temporal e
espacial. O trabalho também compara a seleção de 32 eletrodos, fixa (por saliência) ou dinâmica
(autopruning), com o uso dos 128. Todo o fluxo roda pela linha de comando `emgpred`; os notebooks
em `notebooks/` chamam os mesmos comandos, um por etapa.

## Instalação

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[cuda,notebooks,dev]"     # sem GPU: pip install -e ".[notebooks,dev]"
export EMGPRED_DATA=/caminho/para/data       # contém "Prosthetic Control - October 2024/" e processed/
export EMGPRED_LOGS=/caminho/para/logs       # onde está importancia_eletrodos/ (saliência)
```

## Fluxo

Rode da raiz do repositório (os configs padrão ficam em `configs/`).

| # | Etapa | Comando | Notebook |
|---|---|---|---|
| 1 | CSV -> NPZ por paciente | `emgpred dataset` | `01_dataset` |
| 2 | Saliência + seleção de 32 (Markowitz) | `emgpred importance` | `02_importancia_eletrodos` |
| 3 | Treino individual, 128 e seleções fixas | `emgpred train individual [--set data.selection=media\|sharpe]` | `03_treino_individual` |
| 4 | Autopruning 128 -> 32 | `emgpred train autopruning` | `04_autopruning` |
| 4b | Baselines (Conv(2+1)D, ResNet-50, PET) | `emgpred train individual -c configs/baselines.toml [--set model.name=resnet50\|pet]` | `08_baselines` |
| 4c | Busca de augmentation | `emgpred aug-search` | `09_busca_augmentation` |
| 5 | Leave-one-subject-out | `emgpred train loso [--set data.selection=best\|media\|todos]` | `05_loso` |
| 6 | Métricas, Wilcoxon, não inferioridade | `emgpred evaluate --runs ...` | `06_resultados` |
| 7 | Latência de inferência | `emgpred latency [--dynamic] [--cpu]` | `07_latencia` |

Opções comuns:
- `-p BBS YSA`: pacientes (padrão: todos);
- `-c outro.toml`: outro arquivo de config;
- `--no-skip`: retreina quem já tem `results_summary.csv`;
- `--set secao.chave=valor`: sobrescreve qualquer chave do config, por exemplo `--set model.d_model=32 training.epochs=50`.

Use `emgpred <comando> -h` para ver todas as opções.

Exemplo de avaliação, com os modos do autopruning lidos por `DIR|modo`:

```bash
emgpred evaluate --out results/ \
  --runs 128=logs/fixo_todos_8020 media=logs/fixo_media_8020 "dyn=logs/autopruning_32_8020|dyn" \
  --noninf media:128 --margin 0.01
```

### Saídas

Cada paciente gera `logs/<experimento>/paciente_<PID>/`:

| Arquivo | Conteúdo |
|---|---|
| `predictions.csv`, `ground_truth.csv` | colunas 0..4, sem suavização, na ordem do `_test.npz` |
| `results_summary.csv` | R²/MAE brutos e suavizados, melhor época, *rollbacks*, eletrodos |
| `model.weights.h5` | pesos |
| `training_history.csv` | histórico do treino |
| `predicao.png` | gráfico predição × ground truth |

Os nomes das pastas seguem os notebooks: `fixo_{todos,media,sharpe}_8020[_d32]`,
`autopruning_32_8020[_d32]` e `loso_{best,media,todos}[_d32]`. O autopruning salva também
`predictions_{dyn,pat,pattask}.csv`, as máscaras fixas e a matriz de confusão do classificador
de tarefa.

## Protocolos

| | Individual | Autopruning | LOSO |
|---|---|---|---|
| Treino / validação | `_train.npz` 80/20 (`train_test_split`, seed 42) | idem | `_train.npz` dos outros pacientes; validação = `_test.npz` dos 4 pacientes seguintes ao retido (fora do treino) |
| Teste | `_test.npz` do paciente | idem | `_test.npz` do retido |
| Modelo | GVT 4+4 camadas, d_model 128 (32 nas ablações), 4 cabeças, mlp 4, dropout 0.4 | mesmo backbone + seletor (k 128 -> 32 em 15 épocas, após 3 de warmup) | mesmo backbone, d_model 32; entrada padronizada por paciente |
| Otimização | Adam 1e-3, clipnorm 1, Huber, lote 128, até 400 épocas, ES 15, ReduceLR 0.75/10 | idem (ES/ReduceLR só após o pruning) | lote 512, 2000 passos/época, até 200, ES 15 e ReduceLR 0.75/5 na média de 3 épocas da val_loss |

Todos os treinos usam o `CollapseGuard`: se a `val_loss` passar de 3× a melhor, os melhores pesos
são restaurados e o LR cai pela metade.

Seleções de 32 eletrodos (`data.selection`):

| Seleção | Eletrodos |
|---|---|
| `media` | os 32 de maior saliência média |
| `sharpe` | a carteira de maior retorno/risco da fronteira eficiente (`melhores_32_eletrodos.csv`) |
| `best` | a seleção proposta (`configs/channels/best32.json`), em ordem crescente como nos notebooks |
| `aleatorio` | 32 sorteados com `data.random_seed` (controle de acaso; pasta `fixo_aleatorio<seed>_...`) |
| `todos` | os 128 |

### LOSO

Mesma configuração do `PhD_LOSO` usada no artigo (`loso_best_d32_vp4_ma3_norm_trainonly`):

| Chave | Valor | Efeito |
|---|---|---|
| `data.val_patients` | 4 | validação no `_test.npz` dos 4 pacientes seguintes ao retido (0 = sorteio na pool) |
| `training.val_smooth` | 3 | checkpoint, ES e ReduceLR na média móvel de 3 épocas da val_loss |
| `data.normalize` | true | cada canal padronizado com média/desvio do `_train` do próprio paciente |
| `data.pool_test` | false | pool só com `_train`, o que libera o `_test` dos pacientes de treino para a avaliação dos "vistos" (`vistos.csv`, colunas `seen_*`) |
| `paths.init_from` | "" | pasta de modelos individuais para inicializar os pesos |

O fold de 32 canais ocupa ~28 GB de RAM (128 canais: ~113 GB). Para a arquitetura do LOSO antigo
(`logs/LOO`): `--set model.d_model=128 model.n_layers_time=6 model.n_layers_space=6 model.mlp_ratio=6.0 model.dropout=0.2`.

### Busca de augmentation

`emgpred aug-search` (`configs/aug_search.toml`) treina o modelo individual (best, d_model 32) no BBS com
combinações das camadas e escolhe a de menor perda de validação. A validação é temporal (20% finais de cada
gravação do `_train.npz`, 4 janelas de folga), então o `_test.npz` não entra na escolha. A busca é gulosa
(acrescenta a camada que mais reduz a perda enquanto houver ganho ≥ 1%) ou exaustiva. As tentativas ficam
em `trials.csv` e a escolha, em `escolhida.json`. O treino final é o `emgpred train individual` com
`augmentation.names` = a combinação escolhida.

### Estatística

O `emgpred evaluate` faz Friedman entre os modelos (com 3 ou mais) e Wilcoxon pareado em cada par, com o
p ajustado por Holm e o tamanho de efeito rank-biserial. Com `--noninf A:B`, faz o teste de não
inferioridade (margem 0,01 de R², Wilcoxon unilateral e IC 95% bootstrap). O `emgpred importance` gera
também o heatmap das duas grades (top-32 por saliência média e subconjunto Markowitz).

### Baselines

Mesmo protocolo do GVT individual (128 eletrodos, sem augmentation), definições iguais às do
`PhD_Baselines`. A contagem de parâmetros é verificada nos testes.

| `model.name` | Arquitetura | Parâmetros |
|---|---|---|
| `conv2plus1d` | Conv(2+1)D residual com BatchNormalization (run `resnet18`) | 245.237 |
| `resnet50` | `keras.applications.ResNet50` sem pré-treino, canal repetido 3× | 23.597.957 |
| `pet` | External Attention em paralelo com o MLP, 2 camadas, d 128 | 215.045 |
| `raulnet` | CNN 3D + MLP com SMU, 2 grades de 64 (fora do artigo: ~2 h/época) | 22.710.906 |

### Data augmentation

Só no treino, na entrada do modelo (`[augmentation] names`), com as camadas e os parâmetros dos
notebooks. A ordem de aplicação é fixa: `amplitude` (U(0,8; 1,2) por canal), `passabanda`, `fase`,
`deslocamento`, `mascara`, `ruido`, `mascara_freq`, `warp`, `queda_canal` e `eletrodo`. O
deslocamento de eletrodo atua na grade de 128 canais antes da seleção, então com ele o modelo
recebe os 128 e seleciona internamente. A pasta de saída ganha o sufixo `_aug-com-<nomes>`,
por exemplo:

```bash
emgpred train individual --set data.selection=best model.d_model=32 'augmentation.names=["amplitude"]'
# -> logs/fixo_best_8020_d32_aug-com-amplitude
```

## Estrutura

```
src/emgpred/
  cli.py                 comando `emgpred`
  config.py              leitura dos TOML (+ --set, variáveis de ambiente)
  constants.py           tarefas, dedos, frequências
  data/preprocessing.py  filtros, normalização, janelamento (CSV -> janelas)
  data/datasets.py       NPZ por paciente, cache .npy, splits, fold LOSO, tf.data
  models/                GVT e camadas, seletor do autopruning, baselines, augmentation
  training/              individual, LOSO, autopruning, callbacks
  postprocessing.py      Kalman (janela ±h, atraso fixo, causal, RTS) e afim
  evaluation/            métricas, consolidação, Wilcoxon, não inferioridade, saliência/Markowitz,
                         latência, figuras
configs/                 um TOML por etapa + lista de eletrodos da seleção proposta
notebooks/               01..09, um por etapa do fluxo
tests/                   testes (inclui equivalência com o código original dos notebooks)
```

## Observações de reprodutibilidade

- **Alinhamento EMG/ângulo** (`file_layout`): o alvo deveria ser o ângulo dentro da janela
  atual (amostra 96 de 100). Nos datasets da tese, porém, os ângulos foram interpolados para o
  comprimento do EMG antes do downsample 2048 -> 1024 Hz. Com isso, o índice 96 + 10i está em escala
  de 2048 Hz e a janela, em 1024 Hz, e o alvo da janela *i* fica em ~metade do tempo da janela de EMG.
  Num arquivo de 45 s, o EMG cobre 0–45 s e os ângulos usados, só 0–22,5 s. Com o alinhamento
  corrigido, o envelope do EMG se correlaciona com a flexão (|r| = 0,42–0,74 em dois arquivos do
  BBS); no legado, |r| < 0,05. `legacy_alignment = true` (em `configs/dataset.toml`) reproduz os
  datasets usados; use `false` em datasets novos.
- **Normalização**: o min/max do EMG (-> [-1, 1]) e dos ângulos (-> [0, 1]) é calculado por
  paciente, só com os trechos de treino de todos os arquivos dele, e aplicado igual ao treino e ao
  teste; os parâmetros ficam salvos no NPZ (`normalization`). Os datasets usados na tese
  normalizavam cada arquivo inteiro, inclusive o trecho de teste, antes do split, e precisam ser
  regerados (`emgpred dataset`). Resta um vazamento mínimo: o `filtfilt` (fase zero) e o `resample`
  por FFT são aplicados à gravação inteira antes do split. Filtros causais eliminariam isso e
  corresponderiam ao que o dispositivo faz em tempo real.
- **Corte do fim do EMG**: o `process_file_pair` do notebook descarta as últimas 100 amostras do EMG
  antes de janelar (mantido aqui), mas os NPZ usados na tese foram gerados sem esse corte (10 janelas
  a mais por arquivo).
- **Suavização**: o padrão é `windowed_kalman`, um RTS numa janela de ±3 janelas (atraso de 3
  janelas ≈ 29 ms), ajustável em `kalman.half_window`. Também há `filter` (causal),
  `fixed_lag_kalman` e `smooth` (sequência inteira, offline). A recalibração afim, quando usada, é
  ajustada num trecho inicial do teste e avaliada no restante (`--affine-calib`).
- **Seleção por Markowitz**: os notebooks originais sorteavam os subconjuntos sem semente, então
  cada execução gerou uma lista diferente (`melhores_32_eletrodos.csv` × `best32.json`). A
  função atual fixa a semente.
- **LOSO**: os chunks antigos (`CHUNKS`/`CHUNKS_FULL`) estavam ordenados por movimento. A
  validação deles era só Power, e no LOSO antigo o Power ficou fora do treino. O fold agora é
  montado do RAW, com validação sorteada.
- **Runs antigos** (`novo`, `novo_32`, `no_data_aug_*`, `resnet18`, `raul`, `LOO`) usaram outros
  splits: `novo`, `resnet18` e `raul` treinaram com 20% do `_train.npz`, `novo_32` com 30% e
  `no_data_aug_50` com 50%. Compare modelos só com o mesmo split. O modelo da pasta `resnet18` é o
  `conv2plus1d` (Conv2D espacial + convolução temporal); não é uma ResNet-18.
