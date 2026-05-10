# -*- coding: utf-8 -*-
"""Gera comparativos e graficos da fase exploratoria de clustering.

Este script consolida a parte de metricas e visualizacao exploratoria do
notebook TPF_INF_493 (1).ipynb para comparar KMeans, DBSCAN e clustering
hierarquico sobre o mesmo embedding usado no fluxo do artigo.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from pipeline_meshi_stages import PortsConnectivityFeaturizer, build_groups, load_prepare_input


RSEED = 42


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_split_indices_simple(df: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    y_type = df["type"].astype(str)
    groups = build_groups(df)
    feature_cols = [
        c
        for c in df.columns
        if c not in ["label", "type", "split", "cluster_kmeans30", "cluster_name", "purity_valid", "purity_test"]
    ]
    dummy_x = np.zeros((len(df), 1), dtype=np.uint8)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RSEED)
    tr_idx, te_idx = next(sgkf.split(dummy_x, y_type, groups=groups))
    return feature_cols, np.asarray(tr_idx), np.asarray(te_idx)


def make_embedding(df: pd.DataFrame, max_sample: int, seed: int) -> tuple[np.ndarray, pd.Series, pd.Series]:
    feature_cols, tr_idx, te_idx = build_split_indices_simple(df)

    feat = PortsConnectivityFeaturizer(topN=20, log_counts=True, drop_raw_ips=True)
    x_train = feat.fit_transform(df.iloc[tr_idx][feature_cols])
    x_test = feat.transform(df.iloc[te_idx][feature_cols])

    num_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = x_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    for col in cat_cols:
        x_train[col] = x_train[col].astype("string")
        x_test[col] = x_test[col].astype("string")

    already_logged = {"pair_count", "src_out_count", "dst_in_count", "src_unique_dst", "dst_unique_src"}
    skew_cols = [c for c in num_cols if c not in ["src_port_wk", "dst_port_wk"] and c not in already_logged]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                skew_cols,
            ),
            (
                "num_flags",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                [c for c in num_cols if c not in skew_cols],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    embedder = Pipeline(
        [
            ("preprocess", preprocess),
            ("svd", TruncatedSVD(n_components=50, random_state=seed)),
            ("scaler", StandardScaler()),
        ]
    )

    embedder.fit(x_train)
    z_test = embedder.transform(x_test)
    y_type = df.iloc[te_idx]["type"].astype(str).reset_index(drop=True)
    y_label = df.iloc[te_idx]["label"].astype(str).reset_index(drop=True)

    if max_sample and z_test.shape[0] > max_sample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(z_test.shape[0], size=max_sample, replace=False)
        z_test = z_test[idx]
        y_type = y_type.iloc[idx].reset_index(drop=True)
        y_label = y_label.iloc[idx].reset_index(drop=True)

    return z_test, y_type, y_label


def internal_metrics(embedding: np.ndarray, labels: np.ndarray, ignore_noise: bool = True, seed: int = RSEED) -> dict:
    labels = np.asarray(labels)
    if ignore_noise and np.any(labels == -1):
        mask = labels != -1
        embedding_eval = embedding[mask]
        labels_eval = labels[mask]
    else:
        embedding_eval = embedding
        labels_eval = labels

    unique_labels = np.unique(labels_eval)
    if embedding_eval.shape[0] < 2 or len(unique_labels) < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "n_points_eval": int(embedding_eval.shape[0]),
            "n_clusters_eval": int(len(unique_labels)),
        }

    max_sil_samples = min(5000, embedding_eval.shape[0])
    sil = silhouette_score(embedding_eval, labels_eval, sample_size=max_sil_samples, random_state=seed)
    return {
        "silhouette": float(sil),
        "davies_bouldin": float(davies_bouldin_score(embedding_eval, labels_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(embedding_eval, labels_eval)),
        "n_points_eval": int(embedding_eval.shape[0]),
        "n_clusters_eval": int(len(unique_labels)),
    }


def external_metrics(labels: np.ndarray, y_type: pd.Series, y_label: pd.Series, suffix: str = "") -> dict:
    labels = np.asarray(labels)
    return {
        f"ARI_type{suffix}": float(adjusted_rand_score(y_type, labels)),
        f"AMI_type{suffix}": float(adjusted_mutual_info_score(y_type, labels, average_method="arithmetic")),
        f"V_type{suffix}": float(v_measure_score(y_type, labels)),
        f"ARI_label{suffix}": float(adjusted_rand_score(y_label, labels)),
        f"AMI_label{suffix}": float(adjusted_mutual_info_score(y_label, labels, average_method="arithmetic")),
        f"V_label{suffix}": float(v_measure_score(y_label, labels)),
    }


def cluster_size_stats(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    labels = labels[labels != -1]
    if labels.size == 0:
        return {"min_size": 0, "p1_size": 0.0, "n_small_lt_10": 0, "n_small_lt_50": 0, "max_size": 0}
    vc = pd.Series(labels).value_counts()
    return {
        "min_size": int(vc.min()),
        "p1_size": float(np.percentile(vc.values, 1)),
        "n_small_lt_10": int((vc < 10).sum()),
        "n_small_lt_50": int((vc < 50).sum()),
        "max_size": int(vc.max()),
    }


def sweep_kmeans(embedding: np.ndarray, y_type: pd.Series, y_label: pd.Series, k_min: int, k_max: int) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        model = MiniBatchKMeans(n_clusters=k, random_state=RSEED, batch_size=4096, n_init=10)
        labels = model.fit_predict(embedding)
        rows.append(
            {
                "algo": "KMeans",
                "param": f"k={k}",
                "k": k,
                "inertia": float(model.inertia_),
                "n_clusters": int(k),
                "noise_ratio": 0.0,
                **internal_metrics(embedding, labels, ignore_noise=False),
                **external_metrics(labels, y_type, y_label),
                **cluster_size_stats(labels),
            }
        )
    return pd.DataFrame(rows)


def propose_eps_from_kdist(embedding: np.ndarray, min_samples: int, quantiles: tuple[float, ...]) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min_samples, metric="euclidean", n_jobs=1)
    nn.fit(embedding)
    distances, _ = nn.kneighbors(embedding, return_distance=True)
    kth_dist = np.sort(distances[:, -1])
    return np.unique(np.quantile(kth_dist, quantiles))


def sweep_dbscan(embedding: np.ndarray, y_type: pd.Series, y_label: pd.Series) -> pd.DataFrame:
    z = StandardScaler().fit_transform(embedding)
    rows = []
    for min_samples in [10, 20, 30]:
        eps_values = propose_eps_from_kdist(z, min_samples, (0.90, 0.93, 0.95, 0.97, 0.98, 0.99))
        for eps in eps_values:
            model = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="euclidean", n_jobs=1)
            labels = model.fit_predict(z)
            noise_ratio = float((labels == -1).mean())
            n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))

            row = {
                "algo": "DBSCAN",
                "param": f"eps={eps:.4f}, min_samples={min_samples}",
                "eps": float(eps),
                "min_samples": int(min_samples),
                "n_clusters": n_clusters,
                "noise_ratio": noise_ratio,
                **internal_metrics(z, labels, ignore_noise=True),
                **external_metrics(labels, y_type, y_label, suffix="_all"),
                **cluster_size_stats(labels),
            }

            mask = labels != -1
            if mask.sum() >= 2 and len(np.unique(labels[mask])) >= 2:
                row.update(external_metrics(labels[mask], y_type.iloc[mask], y_label.iloc[mask], suffix="_no_noise"))
            rows.append(row)
    return pd.DataFrame(rows)


def make_connected_knn_graph(embedding: np.ndarray, k_start: int = 30, k_step: int = 20, k_max: int = 400):
    last = None
    for k in range(k_start, k_max + 1, k_step):
        conn = kneighbors_graph(embedding, n_neighbors=k, include_self=False, n_jobs=-1)
        n_components, _ = connected_components(conn)
        last = (conn, int(n_components), int(k))
        if n_components == 1:
            return last
    return last


def sweep_hierarchical(embedding: np.ndarray, y_type: pd.Series, y_label: pd.Series) -> pd.DataFrame:
    rows = []
    conn, n_components, k_used = make_connected_knn_graph(embedding)
    for linkage in ["ward", "average", "complete"]:
        for k in [9, 10, 20, 30, 40]:
            model = AgglomerativeClustering(
                n_clusters=int(k),
                linkage=linkage,
                connectivity=conn,
                compute_full_tree=False,
            )
            labels = model.fit_predict(embedding)
            rows.append(
                {
                    "algo": "Hierarquico",
                    "param": f"linkage={linkage}, k={k}",
                    "linkage": linkage,
                    "k": int(k),
                    "n_clusters": int(k),
                    "noise_ratio": 0.0,
                    "knn_components": n_components,
                    "k_used_knn": k_used,
                    **internal_metrics(embedding, labels, ignore_noise=False),
                    **external_metrics(labels, y_type, y_label),
                    **cluster_size_stats(labels),
                }
            )
    return pd.DataFrame(rows)


def select_best_rows(kmeans_df: pd.DataFrame, dbscan_df: pd.DataFrame, hierarchical_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(kmeans_df.loc[kmeans_df["AMI_type"].idxmax()])
    rows.append(dbscan_df.loc[dbscan_df["AMI_type_all"].idxmax()])
    rows.append(hierarchical_df.loc[hierarchical_df["AMI_type"].idxmax()])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["AMI_type_comparavel"] = out["AMI_type"].fillna(out.get("AMI_type_all"))
    out["V_type_comparavel"] = out["V_type"].fillna(out.get("V_type_all"))
    out["AMI_label_comparavel"] = out["AMI_label"].fillna(out.get("AMI_label_all"))
    out["V_label_comparavel"] = out["V_label"].fillna(out.get("V_label_all"))
    return out


def load_metric_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "KMeans": output_dir / "metricas_kmeans.csv",
        "DBSCAN": output_dir / "metricas_dbscan.csv",
        "Hierarquico": output_dir / "metricas_hierarquico.csv",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Não foi possível gerar o comparativo porque faltam tabelas de métricas: "
            + ", ".join(missing)
            + ". Execute antes as etapas kmeans, dbscan e hierarquico."
        )
    return (
        pd.read_csv(paths["KMeans"]),
        pd.read_csv(paths["DBSCAN"]),
        pd.read_csv(paths["Hierarquico"]),
    )


def make_comparison_from_existing(output_dir: Path, figures_dir: Path) -> pd.DataFrame:
    kmeans_df, dbscan_df, hierarchical_df = load_metric_tables(output_dir)
    best_df = select_best_rows(kmeans_df, dbscan_df, hierarchical_df)
    best_df.to_csv(output_dir / "comparativo_melhores_configuracoes.csv", index=False)
    plot_algorithm_comparison(best_df, figures_dir)
    plot_algorithm_performance_panel(best_df, figures_dir)
    return best_df


def plot_kmeans_curves(kmeans_df: pd.DataFrame, output_dir: Path) -> None:
    mk = kmeans_df.sort_values("k")
    specs = [
        ("silhouette", "Silhouette"),
        ("davies_bouldin", "Davies-Bouldin"),
        ("calinski_harabasz", "Calinski-Harabasz"),
        ("inertia", "Inercia"),
    ]
    for col, ylabel in specs:
        plt.figure(figsize=(6, 4))
        plt.plot(mk["k"], mk[col], marker="o")
        plt.axvline(30, color="red", linestyle="--", label="k=30")
        plt.xlabel("k")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"kmeans_{col}.png", dpi=200)
        plt.close()


def plot_algorithm_comparison(best_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("silhouette", "Silhouette"),
        ("davies_bouldin", "Davies-Bouldin"),
        ("calinski_harabasz", "Calinski-Harabasz"),
        ("AMI_type_comparavel", "AMI type"),
        ("V_type_comparavel", "V-measure type"),
        ("AMI_label_comparavel", "AMI label"),
        ("V_label_comparavel", "V-measure label"),
    ]
    labels = best_df["algo"].tolist()
    for col, ylabel in metrics:
        plt.figure(figsize=(7, 4))
        plt.bar(labels, best_df[col])
        plt.ylabel(ylabel)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"comparativo_{col}.png", dpi=200)
        plt.close()


def plot_algorithm_performance_panel(best_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = best_df.copy()
    plot_df["label"] = plot_df["algo"] + "\n" + plot_df["param"].astype(str)

    metrics = [
        ("silhouette", "Silhouette\n(maior melhor)"),
        ("davies_bouldin", "Davies-Bouldin\n(menor melhor)"),
        ("AMI_type_comparavel", "AMI type\n(maior melhor)"),
        ("V_type_comparavel", "V-measure type\n(maior melhor)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar(plot_df["label"], plot_df[col])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "comparativo_desempenho_algoritmos.png", dpi=220)
    plt.close(fig)


def plot_svd_2d_projection(embedding: np.ndarray, y_type: pd.Series, output_dir: Path) -> None:
    if embedding.shape[1] < 2:
        return

    categories = pd.Categorical(y_type.astype(str))
    codes = categories.codes

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=codes,
        cmap="tab10",
        s=8,
        alpha=0.65,
        linewidths=0,
    )
    handles, _ = scatter.legend_elements(num=len(categories.categories))
    plt.legend(handles, categories.categories, title="type", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.xlabel("SVD componente 1")
    plt.ylabel("SVD componente 2")
    plt.tight_layout()
    plt.savefig(output_dir / "svd_2d_por_type.png", dpi=220)
    plt.close()

    km = MiniBatchKMeans(n_clusters=30, random_state=RSEED, batch_size=4096, n_init=10)
    clusters = km.fit_predict(embedding)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=clusters,
        cmap="tab20",
        s=8,
        alpha=0.65,
        linewidths=0,
    )
    plt.xlabel("SVD componente 1")
    plt.ylabel("SVD componente 2")
    plt.colorbar(label="cluster KMeans k=30")
    plt.tight_layout()
    plt.savefig(output_dir / "svd_2d_por_cluster_kmeans30.png", dpi=220)
    plt.close()


def plot_cluster_type_heatmap_from_kmeans(df: pd.DataFrame, output_dir: Path) -> None:
    if "cluster_kmeans30" not in df.columns:
        return
    ct = pd.crosstab(df["cluster_kmeans30"], df["type"])
    pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    plt.figure(figsize=(10, 6))
    plt.imshow(pct.values, aspect="auto")
    plt.xticks(range(pct.shape[1]), pct.columns, rotation=45, ha="right")
    plt.yticks(range(pct.shape[0]), pct.index.astype(str))
    plt.xlabel("type")
    plt.ylabel("cluster_kmeans30")
    plt.colorbar(label="proporcao no cluster")
    plt.tight_layout()
    plt.savefig(output_dir / "kmeans_heatmap_type_por_cluster.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="dados/train_test_network.parquet")
    parser.add_argument("--output-dir", default="resultados/exploratorio")
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--max-sample", type=int, default=8000)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=60)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["all"],
        choices=["all", "kmeans", "dbscan", "hierarquico", "comparativo", "svd"],
        help=(
            "Etapas a executar. Use chamadas separadas no Colab para reduzir memória, "
            "por exemplo: --algorithms kmeans; depois dbscan; depois hierarquico; "
            "por fim comparativo."
        ),
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(args.figures_dir) if args.figures_dir else ensure_dir(output_dir / "graficos")

    requested = set(args.algorithms)
    if "all" in requested:
        requested = {"kmeans", "dbscan", "hierarquico", "comparativo", "svd"}

    needs_embedding = bool(requested & {"kmeans", "dbscan", "hierarquico", "svd"})
    if needs_embedding:
        df = load_prepare_input(args.data_path)
        embedding, y_type, y_label = make_embedding(df, max_sample=args.max_sample, seed=RSEED)
    else:
        df = None
        embedding = y_type = y_label = None

    if "kmeans" in requested:
        kmeans_df = sweep_kmeans(embedding, y_type, y_label, args.k_min, args.k_max)
        kmeans_df.to_csv(output_dir / "metricas_kmeans.csv", index=False)
        plot_kmeans_curves(kmeans_df, figures_dir)
        del kmeans_df
        gc.collect()

    if "dbscan" in requested:
        dbscan_df = sweep_dbscan(embedding, y_type, y_label)
        dbscan_df.to_csv(output_dir / "metricas_dbscan.csv", index=False)
        del dbscan_df
        gc.collect()

    if "hierarquico" in requested:
        hierarchical_df = sweep_hierarchical(embedding, y_type, y_label)
        hierarchical_df.to_csv(output_dir / "metricas_hierarquico.csv", index=False)
        del hierarchical_df
        gc.collect()

    if "svd" in requested:
        plot_svd_2d_projection(embedding, y_type, figures_dir)
        if df is not None:
            plot_cluster_type_heatmap_from_kmeans(df, figures_dir)

    if "comparativo" in requested:
        make_comparison_from_existing(output_dir, figures_dir)

    print("Arquivos salvos em:", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
