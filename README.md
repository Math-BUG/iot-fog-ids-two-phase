# Materiais complementares do artigo "Abordagem em duas fases para detecção e classificação de ataques em redes IoT com computação em névoa"

Este repositório reúne o código principal, os dados utilizados e os resultados associados ao fluxo descrito no artigo.

## Estrutura

- `codigo/`: código principal utilizado no pipeline do artigo.
- `dados/`: dados de entrada e artefatos intermediários usados pelo código.
- `resultados/`: resultados gerados pelas execuções associadas ao artigo.

O script `codigo/gerar_graficos_exploratorios.py` gera os comparativos e gráficos da fase exploratória envolvendo KMeans, DBSCAN e clustering hierárquico.

Também há dois notebooks para execução no Google Colab:

- `codigo/COLAB_01_FASE_EXPLORATORIA.ipynb`: gera o comparativo exploratório e os gráficos.
- `codigo/COLAB_02_PIPELINE_ARTIGO.ipynb`: executa o pipeline principal e salva o artefato intermediário e as métricas supervisionadas.

