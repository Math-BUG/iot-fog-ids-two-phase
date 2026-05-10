# Comparativo de métricas exploratórias

Valores extraídos do notebook `TPF_INF_493 (1).ipynb`. As métricas com `type` comparam os clusters com a classe de ataque; as métricas com `label` comparam os clusters com a classe binária.

## Melhores configurações observadas

| Algoritmo | Configuração | Silhouette | Davies-Bouldin | Calinski-Harabasz | AMI type | V type | AMI label | V label | Observação |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| KMeans | k=30 | 0.5865 | 1.0370 | - | 0.6668 | 0.6676 | 0.2640 | 0.2642 | Melhor AMI type entre os Ks avaliados no notebook. |
| DBSCAN | eps=4.0754, min_samples=10 | 0.7199 | 0.5211 | 2069.5788 | 0.6316 | 0.6353 | 0.2396 | 0.2407 | 126 clusters sem ruído; ruído aproximado de 4.18%. |
| Hierárquico | ward, k=30 | 0.5668 | 0.7980 | 1984.2383 | 0.6563 | 0.6584 | 0.2610 | - | Avaliado em amostra de 8000 pontos com grafo kNN conectado. |

## Leituras principais

- O KMeans com `k=30` teve o maior alinhamento externo com `type` no notebook, com `AMI_type=0.6668` e `V_type=0.6676`.
- O DBSCAN teve a melhor métrica interna de separação para a configuração destacada, com `silhouette=0.7199` e `Davies-Bouldin=0.5211`, mas gerou muitos clusters pequenos e ruído.
- O hierárquico com `ward, k=30` ficou próximo do KMeans nas métricas externas, com `AMI_type=0.6563` e `V_type=0.6584`.
- No teste de permutação para KMeans em TEST, o alinhamento entre clusters e `type` foi estatisticamente acima do acaso: `AMI=0.6023`, `V=0.6041`, `p≈0.0050`.
- A escolha do KMeans se justifica por combinar boa aderência externa aos rótulos conhecidos, ausência de pontos classificados como ruído, número controlado de grupos e maior simplicidade operacional em relação às alternativas avaliadas.

## Código reexecutável

O script `codigo/gerar_graficos_exploratorios.py` reproduz a geração das tabelas e dos gráficos comparativos, salvando os artefatos em `resultados/exploratorio/`.

Além das curvas por `k` do KMeans, o script gera:

- painel comparativo de desempenho entre KMeans, DBSCAN e hierárquico;
- projeção 2D dos dados usando os dois primeiros componentes do SVD, colorida por `type`;
- projeção 2D dos dados usando os dois primeiros componentes do SVD, colorida pelos clusters KMeans com `k=30`;
- heatmap de distribuição de `type` por cluster KMeans.
