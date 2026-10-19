# Abordagem em duas fases para detecção e classificação de ataques em redes IoT com computação em névoa

Este repositório reúne os materiais complementares de um estudo sobre detecção e classificação de ataques em redes IoT no contexto de computação em névoa. O fluxo combina uma etapa não supervisionada, usada para analisar agrupamentos de tráfego, com uma etapa supervisionada, usada para avaliar a classificação final de ataques.

O objetivo do projeto é documentar, reproduzir e organizar os artefatos gerados durante os experimentos: notebooks, base de dados processada, métricas, matrizes de confusão e gráficos utilizados na análise.

## Visão geral do fluxo

1. A fase exploratória compara KMeans, DBSCAN e agrupamento hierárquico usando métricas internas e externas de clustering.
2. A configuração com KMeans `k=30` é selecionada como alternativa principal por equilibrar aderência às classes conhecidas, ausência de ruído e simplicidade operacional.
3. O pipeline principal usa os clusters como parte da análise em duas fases e avalia os resultados para `cluster_kmeans30`, `type` e `label`.
4. Os artefatos finais são salvos em tabelas `.csv`, gráficos `.png` e um arquivo intermediário `.parquet` com os clusters atribuídos.

## Estrutura do repositório

```text
.
|-- codigo/
|   |-- COLAB_01_FASE_EXPLORATORIA.ipynb
|   `-- COLAB_02_PIPELINE_ARTIGO.ipynb
|-- dados/
|   |-- train_test_network.parquet
|   `-- train_test_network_with_clusters_k30.parquet
|-- gráficos/
|   |-- exploratório/
|   `-- pipeline/
|-- resultados/
|   |-- exploratorio/
|   |-- pipeline/
|   `-- comparativo_metricas_exploratorias.md
`-- README.md
```

## Conteúdo principal

- `codigo/COLAB_01_FASE_EXPLORATORIA.ipynb`: executa a fase exploratória, cria embeddings, compara algoritmos de clustering e gera gráficos comparativos.
- `codigo/COLAB_02_PIPELINE_ARTIGO.ipynb`: executa o pipeline principal, gera o artefato intermediário com clusters KMeans `k=30` e calcula métricas supervisionadas.
- `dados/train_test_network.parquet`: base de entrada utilizada nos experimentos.
- `dados/train_test_network_with_clusters_k30.parquet`: base enriquecida com os clusters gerados pelo KMeans.
- `resultados/exploratorio/`: tabelas com métricas comparativas de KMeans, DBSCAN e agrupamento hierárquico.
- `resultados/pipeline/`: métricas, matrizes de confusão e distribuições geradas pelo pipeline principal.
- `gráficos/`: figuras exportadas pelos notebooks, separadas entre fase exploratória e pipeline principal.

## Como reproduzir

Os notebooks foram preparados para execução no Google Colab, usando arquivos no Google Drive. Antes de executar, ajuste os caminhos definidos no início de cada notebook, se necessário:

```python
DATASET_PATH = Path('/content/drive/MyDrive/SBCUP/Dataset/train_test_network.parquet')
RESULTADOS_DIR = Path('/content/drive/MyDrive/SBCUP/Resultados')
GRAFICOS_DIR = Path('/content/drive/MyDrive/SBCUP/Gráficos')
```

Dependências usadas nos notebooks:

```bash
pip install -r requirements.txt
```

Ordem recomendada de execução:

1. Execute `codigo/COLAB_01_FASE_EXPLORATORIA.ipynb` para reproduzir a comparação dos algoritmos de clustering.
2. Execute `codigo/COLAB_02_PIPELINE_ARTIGO.ipynb` para gerar o pipeline final, o artefato intermediário e as métricas supervisionadas.
3. Consulte `resultados/comparativo_metricas_exploratorias.md` para uma síntese das configurações avaliadas.

## Resultados em destaque

- Na etapa exploratória, KMeans com `k=30` apresentou o melhor alinhamento externo com a classe `type` entre as configurações avaliadas no notebook.
- DBSCAN obteve bons valores em métricas internas de separação, mas produziu ruído e maior fragmentação em clusters.
- O agrupamento hierárquico ficou próximo do KMeans em métricas externas, mas foi avaliado em amostra e tem maior custo computacional.
- No pipeline principal, as métricas supervisionadas ficam registradas em `resultados/pipeline/metricas_globais_supervisionadas.csv`.

## Observações

- Os arquivos em `dados/`, `resultados/` e `gráficos/` são artefatos do experimento e podem ser grandes.
- Os caminhos dos notebooks apontam para o ambiente do Colab; ao executar localmente, adapte os diretórios de entrada e saída.
- A pasta `gráficos/` mantém o nome em português para preservar a organização original dos artefatos já gerados.


## Publicações e apresentações da IC

- [Organização da publicação SBCUP 2026](publicacoes/sbcup-2026/README.md) — artigo, fontes e documentação de publicação (PDF a adicionar nesta pasta).
- [Painel do SIA 2026](apresentacoes/sia-2026/README.md) — **Arquitetura híbrida para detecção de tráfego malicioso em ambientes IoT** (PDF e PowerPoint revisados a adicionar).
- [Índice dos materiais da IC](docs/ORGANIZACAO_DA_IC.md).

### Algumas figuras já disponíveis

- [Matriz de confusão de `cluster_kmeans30`](gráficos/pipeline/matriz_confusao_cluster_kmeans30.png)
- [Matriz de confusão de `type`](gráficos/pipeline/matriz_confusao_type.png)
- [Heatmap de tipos de ataque por cluster](gráficos/pipeline/kmeans30_heatmap_type_por_cluster.png)
- [Pureza por cluster](gráficos/pipeline/kmeans30_pureza_por_cluster.png)
- [F1 por classe de ataque](gráficos/pipeline/f1_por_classe_type.png)

### Projetos relacionados

- [ERAD-SE 2026 — desempenho CPU/GPU](https://github.com/Math-BUG/experimento_erad)
- [Pesquisa de generalização e avaliação em maior escala](https://github.com/Math-BUG/ids_generalization_pipeline)

O relatório final assinado da IC não é disponibilizado nos repositórios públicos.
