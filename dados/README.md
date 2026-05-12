# Dados

Esta pasta armazena os dados usados e produzidos pelo pipeline.

- `train_test_network.parquet`: base de entrada usada nos experimentos.
- `train_test_network_with_clusters_k30.parquet`: base com a coluna de clusters KMeans `k=30`, gerada pelo pipeline principal.

Os arquivos `.parquet` podem ser grandes. Caso publique este projeto em um repositório remoto, verifique antes se os dados podem ser redistribuídos e se o tamanho é adequado para Git.
