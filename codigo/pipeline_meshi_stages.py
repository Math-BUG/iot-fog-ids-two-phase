# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
import sys
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.metrics.cluster import adjusted_mutual_info_score, v_measure_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RSEED = 42
np.random.seed(RSEED)

SCRIPT_VERSION = "2026-04-28-cv-insufficient-groups-skip"

PARTITION_BASENAME_PATTERN = "Network_dataset_*.parquet"

UINT_COLUMNS = {
    "ts",
    "src_port",
    "dst_port",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_request_body_len",
    "http_response_body_len",
    "http_status_code",
    "label",
    "cluster_kmeans30",
}

FLOAT_COLUMNS = {
    "duration",
    "purity_valid",
    "purity_test",
}

NUMERIC_OBJECT_COLUMNS = {
    "src_bytes",
}

CATEGORY_COLUMNS = {
    "src_ip",
    "dst_ip",
    "proto",
    "service",
    "conn_state",
    "dns_query",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_version",
    "ssl_cipher",
    "ssl_resumed",
    "ssl_established",
    "ssl_subject",
    "ssl_issuer",
    "http_trans_depth",
    "http_method",
    "http_uri",
    "http_referrer",
    "http_version",
    "http_status_code",
    "http_user_agent",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_name",
    "weird_addl",
    "weird_notice",
    "type",
    "split",
    "cluster_name", 
}


def natural_key(path: str):
    name = os.path.basename(path)
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else name


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in UINT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="unsigned")

    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    for col in NUMERIC_OBJECT_COLUMNS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                has_fractional = not np.allclose(
                    numeric.dropna().to_numpy(),
                    np.floor(numeric.dropna().to_numpy()),
                )
                downcast = "float" if has_fractional else "unsigned"
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast=downcast)

    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df.to_csv(path, index=False)
        return

    if ext in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return

    if ext == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            raise RuntimeError(
                "Falha ao salvar em Parquet. Instale pyarrow/fastparquet ou use extensão .pkl/.csv. "
                f"Erro original: {e}"
            ) from e
        return

    raise ValueError(f"Extensão não suportada para salvar: {ext}")


def load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    ext = os.path.splitext(path)[1].lower()
    print(f"Lendo: {path}")

    if ext == ".csv":
        return optimize_dataframe(pd.read_csv(path))

    if ext in {".pkl", ".pickle"}:
        return optimize_dataframe(pd.read_pickle(path))

    if ext == ".parquet":
        try:
            return optimize_dataframe(pd.read_parquet(path))
        except Exception as e:
            raise RuntimeError(
                "Falha ao ler Parquet. Instale pyarrow/fastparquet ou use .pkl/.csv. "
                f"Erro original: {e}"
            ) from e

    raise ValueError(f"Extensão não suportada para leitura: {ext}")

def resolve_partition_files(data_path: str) -> list[str]:
    has_glob = any(ch in data_path for ch in "*?[]")

    if has_glob:
        files = glob.glob(data_path)
    elif os.path.isdir(data_path):
        files = glob.glob(os.path.join(data_path, PARTITION_BASENAME_PATTERN))
    else:
        files = []

    files = [f for f in files if os.path.isfile(f) and f.lower().endswith(".parquet")]
    files = sorted(files, key=natural_key)
    return files



def load_prepare_input(data_path: str) -> pd.DataFrame:
    if os.path.isfile(data_path):
        ext = os.path.splitext(data_path)[1].lower()
        if ext != ".parquet":
            raise ValueError(
                f"Na etapa prepare, o arquivo único precisa ser .parquet. Recebido: {data_path}"
            )
        return load_dataframe(data_path)

    partition_files = resolve_partition_files(data_path)
    if not partition_files:
        raise FileNotFoundError(
            "Nenhum arquivo Parquet de entrada encontrado. "
            f"Recebido: {data_path}"
        )

    print(f"Entrada particionada detectada: {len(partition_files)} arquivos")
    print("Primeiro arquivo:", partition_files[0])
    print("Último arquivo:", partition_files[-1])

    dfs = []
    for i, path in enumerate(partition_files, start=1):
        print(f"[{i}/{len(partition_files)}] Lendo: {path}")
        dfs.append(load_dataframe(path))

    print("Concatenando partições...")
    df = pd.concat(dfs, ignore_index=True, copy=False)
    del dfs
    df = optimize_dataframe(df)
    mem_gib = df.memory_usage(deep=True).sum() / (1024 ** 3)
    print("Shape após concatenação:", df.shape)
    print(f"Memória aproximada após otimização: {mem_gib:.2f} GiB")
    return df


class PortsConnectivityFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, topN=20, log_counts=True, drop_raw_ips=True):
        self.topN = topN
        self.log_counts = log_counts
        self.drop_raw_ips = drop_raw_ips

    def fit(self, X, y=None):
        X = X.copy()

        X["src_port"] = pd.to_numeric(X["src_port"], errors="coerce")
        X["dst_port"] = pd.to_numeric(X["dst_port"], errors="coerce")

        self.top_src_ = set(X["src_port"].value_counts().head(self.topN).index)
        self.top_dst_ = set(X["dst_port"].value_counts().head(self.topN).index)

        X["src_ip"] = X["src_ip"].astype(str).fillna("NA")
        X["dst_ip"] = X["dst_ip"].astype(str).fillna("NA")
        pair = X["src_ip"] + "->" + X["dst_ip"]

        self.vc_pair_ = pair.value_counts()
        self.vc_src_ = X["src_ip"].value_counts()
        self.vc_dst_ = X["dst_ip"].value_counts()

        self.src_unique_dst_ = X.groupby("src_ip")["dst_ip"].nunique()
        self.dst_unique_src_ = X.groupby("dst_ip")["src_ip"].nunique()

        return self

    def transform(self, X):
        X = X.copy()

        X["src_port"] = pd.to_numeric(X["src_port"], errors="coerce")
        X["dst_port"] = pd.to_numeric(X["dst_port"], errors="coerce")

        X["src_port_wk"] = (X["src_port"] < 1024).astype("int8")
        X["dst_port_wk"] = (X["dst_port"] < 1024).astype("int8")

        X["src_port_top"] = X["src_port"].where(X["src_port"].isin(self.top_src_), other="other").astype(str)
        X["dst_port_top"] = X["dst_port"].where(X["dst_port"].isin(self.top_dst_), other="other").astype(str)

        X = X.drop(columns=["src_port", "dst_port"])

        X["src_ip"] = X["src_ip"].astype(str).fillna("NA")
        X["dst_ip"] = X["dst_ip"].astype(str).fillna("NA")
        pair = X["src_ip"] + "->" + X["dst_ip"]

        X["pair_count"] = pair.map(self.vc_pair_).fillna(0).astype(float)
        X["src_out_count"] = X["src_ip"].map(self.vc_src_).fillna(0).astype(float)
        X["dst_in_count"] = X["dst_ip"].map(self.vc_dst_).fillna(0).astype(float)
        X["src_unique_dst"] = X["src_ip"].map(self.src_unique_dst_).fillna(0).astype(float)
        X["dst_unique_src"] = X["dst_ip"].map(self.dst_unique_src_).fillna(0).astype(float)

        if self.log_counts:
            for c in ["pair_count", "src_out_count", "dst_in_count", "src_unique_dst", "dst_unique_src"]:
                X[c] = np.log1p(X[c])

        if self.drop_raw_ips:
            X = X.drop(columns=["src_ip", "dst_ip"], errors="ignore")

        return X


def shuffle_test(metric_fn, y_true, clusters, n=200, seed=42):
    rng = np.random.RandomState(seed)
    base = metric_fn(y_true, clusters)
    null = []
    y = np.array(y_true)
    for _ in range(n):
        ys = y.copy()
        rng.shuffle(ys)
        null.append(metric_fn(ys, clusters))
    null = np.array(null)
    q95 = np.quantile(null, 0.95)
    p = (np.sum(null >= base) + 1) / (n + 1)
    return base, q95, p


def purity_table(df_split, cluster_col="cluster_kmeans30", y_col="type"):
    ct = pd.crosstab(df_split[cluster_col], df_split[y_col])
    dom = ct.idxmax(axis=1)
    size = ct.sum(axis=1)
    purity = ct.max(axis=1) / size
    out = pd.DataFrame({"size": size, "dominant": dom, "purity": purity}).sort_values(
        ["purity", "size"], ascending=[False, False]
    )
    return out, ct


def coverage_at_threshold(purity_df, thr=0.95):
    good = purity_df[purity_df["purity"] >= thr]["size"].sum()
    total = purity_df["size"].sum()
    return good / total if total > 0 else 0.0


def topk(series, k=5):
    vc = series.value_counts().head(k)
    return list(zip(vc.index.tolist(), vc.values.tolist()))


def build_groups(df: pd.DataFrame) -> pd.Series:
    group_cols = [c for c in ["src_ip", "dst_ip", "service", "proto"] if c in df.columns]
    if not group_cols:
        raise ValueError("Nenhuma coluna de grupo encontrada entre src_ip, dst_ip, service, proto.")

    group_df = df[group_cols].astype("string").fillna("NA")
    return pd.util.hash_pandas_object(group_df, index=False).astype("uint64")


def resolve_group_cv_splits(
    y: pd.Series,
    groups: pd.Series,
    requested_splits: int,
    context: str,
) -> int:
    groups_per_class = count_groups_per_class(y, groups)
    min_groups_per_class = int(groups_per_class.min())
    actual_splits = min(requested_splits, min_groups_per_class)

    if actual_splits < 2:
        raise ValueError(
            f"Não há grupos suficientes por classe para CV em {context}. "
            f"Mínimo observado: {min_groups_per_class}."
        )

    if actual_splits != requested_splits:
        print(
            f"Reduzindo n_splits de {requested_splits} para {actual_splits} em {context} "
            f"por limitação de grupos por classe."
        )

    return actual_splits


def count_groups_per_class(y: pd.Series, groups: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({
        "y": pd.Series(y).astype(str).to_numpy(),
        "group": pd.Series(groups).to_numpy(),
    })
    return tmp.groupby("y")["group"].nunique().sort_values()


def print_group_cv_diagnostics(y: pd.Series, groups: pd.Series, min_groups: int = 2) -> None:
    tmp = pd.DataFrame({
        "y": pd.Series(y).astype(str).to_numpy(),
        "group": pd.Series(groups).to_numpy(),
    })
    diag = (
        tmp.groupby("y")
        .agg(n_rows=("y", "size"), n_groups=("group", "nunique"))
        .sort_values(["n_groups", "n_rows"], ascending=[True, False])
    )
    rare = diag[diag["n_groups"] < min_groups]
    if rare.empty:
        return

    print(f"Classes com menos de {min_groups} grupos no treino:")
    print(rare.to_string())


def choose_best_holdout_split(
    candidate_splits: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_series = pd.Series(y).reset_index(drop=True).astype(str)
    class_counts = y_series.value_counts().sort_index()
    total_classes = len(class_counts)

    best_score = None
    best_split = None
    best_meta = None

    for fold_id, (train_idx, holdout_idx) in enumerate(candidate_splits, start=1):
        holdout_counts = y_series.iloc[holdout_idx].value_counts().reindex(class_counts.index, fill_value=0)
        present_classes = int((holdout_counts > 0).sum())
        min_count = int(holdout_counts.min())
        holdout_fraction = len(holdout_idx) / len(y_series)
        expected_counts = class_counts * holdout_fraction
        deviation = float((holdout_counts - expected_counts).abs().sum())
        score = (present_classes, min_count, -deviation)

        if best_score is None or score > best_score:
            best_score = score
            best_split = (train_idx, holdout_idx)
            best_meta = {
                "fold_id": fold_id,
                "present_classes": present_classes,
                "min_count": min_count,
                "deviation": deviation,
                "holdout_size": len(holdout_idx),
                "num_candidates": len(candidate_splits),
                "total_classes": total_classes,
            }

    print(
        f"{context}: escolhendo fold {best_meta['fold_id']}/{best_meta['num_candidates']} "
        f"com {best_meta['present_classes']}/{best_meta['total_classes']} classes no holdout, "
        f"mínimo por classe={best_meta['min_count']}, "
        f"desvio total da distribuição esperada={best_meta['deviation']:.1f}, "
        f"holdout_size={best_meta['holdout_size']}"
    )

    return best_split


def build_split_indices(df: pd.DataFrame, y_type: pd.Series, groups: pd.Series):
    drop_cols = ["label", "type", "split", "cluster_kmeans30", "cluster_name", "purity_valid", "purity_test"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    dummy_X = np.zeros((len(df), 1), dtype=np.uint8)

    outer_splits = resolve_group_cv_splits(y_type, groups, 5, "prepare/outer_split")
    sgkf = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    outer_candidates = list(sgkf.split(dummy_X, y_type, groups=groups))
    trva_idx, te_idx = choose_best_holdout_split(outer_candidates, y_type, "prepare/test")

    inner_splits = resolve_group_cv_splits(
        y_type.iloc[trva_idx],
        groups.iloc[trva_idx],
        5,
        "prepare/inner_split",
    )
    sgkf2 = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=123)
    inner_y = y_type.iloc[trva_idx].reset_index(drop=True)
    inner_candidates = list(
        sgkf2.split(
            np.zeros((len(trva_idx), 1), dtype=np.uint8),
            inner_y,
            groups=groups.iloc[trva_idx],
        )
    )
    tr_idx_rel, va_idx_rel = choose_best_holdout_split(inner_candidates, inner_y, "prepare/valid")
    tr_idx = np.array(trva_idx)[tr_idx_rel]
    va_idx = np.array(trva_idx)[va_idx_rel]

    return feature_cols, tr_idx, va_idx, te_idx


def display(obj):
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


def save_supervised_evaluation(y_true, y_pred, target: str, output_dir: str | None) -> None:
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)
    labels = sorted(pd.Series(y_true).astype(str).unique().tolist())
    y_true_str = pd.Series(y_true).astype(str)
    y_pred_str = pd.Series(y_pred).astype(str)

    summary = pd.DataFrame(
        [
            {
                "target": target,
                "accuracy": accuracy_score(y_true_str, y_pred_str),
                "balanced_accuracy": balanced_accuracy_score(y_true_str, y_pred_str),
                "macro_f1": f1_score(y_true_str, y_pred_str, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true_str, y_pred_str, average="weighted", zero_division=0),
                "mcc": matthews_corrcoef(y_true_str, y_pred_str),
            }
        ]
    )
    summary.to_csv(os.path.join(output_dir, f"{target}_metricas_resumo.csv"), index=False)

    report = classification_report(
        y_true_str,
        y_pred_str,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, f"{target}_metricas_por_classe.csv"))

    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(output_dir, f"{target}_matriz_confusao.csv"))

    cm_norm = confusion_matrix(y_true_str, y_pred_str, labels=labels, normalize="true")
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_df.to_csv(os.path.join(output_dir, f"{target}_matriz_confusao_normalizada.csv"))


def run_prepare_stage(data_path: str, intermediate_path: str) -> None:
    df = load_prepare_input(data_path)

    y_type = df["type"]
    groups = build_groups(df)

    feature_cols, tr_idx, va_idx, te_idx = build_split_indices(df, y_type, groups)
    print("sizes:", len(tr_idx), len(va_idx), len(te_idx))

    feat = PortsConnectivityFeaturizer(topN=20, log_counts=True, drop_raw_ips=True)

    X_tr_raw = df.iloc[tr_idx].loc[:, feature_cols]
    X_tr = feat.fit_transform(X_tr_raw)
    del X_tr_raw

    X_va_raw = df.iloc[va_idx].loc[:, feature_cols]
    X_va = feat.transform(X_va_raw)
    del X_va_raw

    X_te_raw = df.iloc[te_idx].loc[:, feature_cols]
    X_te = feat.transform(X_te_raw)
    del X_te_raw

    num_cols = X_tr.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X_tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("string")
        X_va[c] = X_va[c].astype("string")
        X_te[c] = X_te[c].astype("string")

    already_logged = {"pair_count", "src_out_count", "dst_in_count", "src_unique_dst", "dst_unique_src"}
    skew_cols = [c for c in num_cols if c not in ["src_port_wk", "dst_port_wk"] and c not in already_logged]

    for c in cat_cols:
        tipos = X_tr[c].dropna().map(type).value_counts()
        if len(tipos) > 1:
            print(f"Coluna com tipos mistos: {c}")
            print(tipos)

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]),
                skew_cols,
            ),
            (
                "num_flags",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler(with_mean=False)),
                ]),
                [c for c in num_cols if c not in skew_cols],
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    embedder = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=50, random_state=42)),
        ("scaler", StandardScaler()),
    ])

    kmeans30 = MiniBatchKMeans(n_clusters=30, random_state=42, batch_size=4096, n_init=10)

    Z_tr = embedder.fit_transform(X_tr)
    kmeans30.fit(Z_tr)

    cl_tr = kmeans30.labels_
    cl_va = kmeans30.predict(embedder.transform(X_va))
    cl_te = kmeans30.predict(embedder.transform(X_te))

    df["split"] = "none"
    split_col = df.columns.get_loc("split")
    df.iloc[tr_idx, split_col] = "train"
    df.iloc[va_idx, split_col] = "valid"
    df.iloc[te_idx, split_col] = "test"

    df["cluster_kmeans30"] = -1
    cluster_col = df.columns.get_loc("cluster_kmeans30")
    df.iloc[tr_idx, cluster_col] = cl_tr
    df.iloc[va_idx, cluster_col] = cl_va
    df.iloc[te_idx, cluster_col] = cl_te
    df = optimize_dataframe(df)

    print(df["cluster_kmeans30"].value_counts())

    for split in ["valid", "test"]:
        m = df["split"].eq(split)
        cl = df.loc[m, "cluster_kmeans30"].values
        yt = df.loc[m, "type"].astype(str).values
        yb = df.loc[m, "label"].astype(str).values

        ami_t, q95_t, p_t = shuffle_test(
            lambda a, b: adjusted_mutual_info_score(a, b, average_method="arithmetic"), yt, cl
        )
        v_t, q95v, pv = shuffle_test(v_measure_score, yt, cl)

        ami_b, q95b, pb2 = shuffle_test(
            lambda a, b: adjusted_mutual_info_score(a, b, average_method="arithmetic"), yb, cl
        )
        v_b, q95vb, pvb = shuffle_test(v_measure_score, yb, cl)

        print(f"\n== {split.upper()} ==")
        print(f"TYPE: AMI={ami_t:.4f} (q95_shuffle={q95_t:.4f}, p={p_t:.4f}) | V={v_t:.4f} (q95={q95v:.4f}, p={pv:.4f})")
        print(f"BIN : AMI={ami_b:.4f} (q95_shuffle={q95b:.4f}, p={pb2:.4f}) | V={v_b:.4f} (q95={q95vb:.4f}, p={pvb:.4f})")

    df_test = df[df["split"] == "test"].copy()

    pur, ct = purity_table(df_test, "cluster_kmeans30", "type")
    print("Coverage purity>=0.90:", coverage_at_threshold(pur, 0.90))
    print("Coverage purity>=0.95:", coverage_at_threshold(pur, 0.95))
    display(pur.head(15))

    for t in df_test["type"].astype(str).unique():
        cols = ct[t].sort_values(ascending=False).head(3)
        print(f"\nType={t} top clusters:", list(zip(cols.index.tolist(), cols.values.tolist())))

    df_test = df[df["split"] == "test"].copy()
    ct = pd.crosstab(df_test["cluster_kmeans30"], df_test["type"])

    TOPN = 3
    top_by_type = {t: ct[t].nlargest(TOPN) for t in ct.columns}
    for t in ct.columns:
        pairs = list(zip(top_by_type[t].index.tolist(), top_by_type[t].values.tolist()))
        print(f"Type={t} top clusters: {pairs}")

    clusters_show = sorted(set().union(*[s.index.tolist() for s in top_by_type.values()]))
    print("\nclusters_show =", clusters_show)

    ct_sel = ct.loc[clusters_show].astype(int)
    display(ct_sel)

    pct_sel = (ct_sel.div(ct_sel.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(2)
    display(pct_sel)

    size = ct_sel.sum(axis=1)
    dominant = ct_sel.idxmax(axis=1)
    purity = (ct_sel.max(axis=1) / size.replace(0, np.nan)).round(4)

    summary = pd.DataFrame({"size": size, "dominant": dominant, "purity": purity}).sort_values(
        ["purity", "size"], ascending=[False, False]
    )
    display(summary)

    print("Resumo por split e type:")
    display(pd.crosstab(df["split"], df["type"]))

    print("Estimando cluster_name e purity_valid usando TRAIN+VALID como referência.")
    df_trva = df.loc[df["split"].isin(["train", "valid"]), ["cluster_kmeans30", "type"]].copy()
    ct = pd.crosstab(df_trva["cluster_kmeans30"], df_trva["type"])

    dom = ct.idxmax(axis=1)
    size = ct.sum(axis=1)
    purity = (ct.max(axis=1) / size.replace(0, np.nan)).fillna(0)

    cluster_info = pd.DataFrame({
        "dominant_type": dom,
        "purity_valid": purity,
        "size_valid": size,
    })
    cluster_info["cluster_name"] = cluster_info["dominant_type"].astype(str) + "_like"

    display(cluster_info.sort_values(["purity_valid", "size_valid"], ascending=[False, False]))

    df["cluster_name"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["purity_valid"] = np.nan

    name_map = cluster_info["cluster_name"]
    purity_map = cluster_info["purity_valid"]

    m_clustered = df["cluster_kmeans30"].ge(0)
    df.loc[m_clustered, "cluster_name"] = df.loc[m_clustered, "cluster_kmeans30"].map(name_map)
    df.loc[m_clustered, "purity_valid"] = df.loc[m_clustered, "cluster_kmeans30"].map(purity_map)

    df_test = df[df["split"] == "test"].copy()
    ct = pd.crosstab(df_test["cluster_kmeans30"], df_test["type"])
    sizes = ct.sum(axis=1)

    base = (df_test["type"] == "mitm").mean()
    mitm_cnt = ct.get("mitm", pd.Series(0, index=ct.index))
    mitm_rate = (mitm_cnt / sizes).fillna(0)
    lift = (mitm_rate / base).replace([np.inf, -np.inf], np.nan).fillna(0)

    mitm_view = pd.DataFrame({
        "size": sizes,
        "mitm_count": mitm_cnt,
        "mitm_rate": mitm_rate,
        "lift_vs_base": lift,
    }).sort_values(["lift_vs_base", "mitm_count"], ascending=False)
    display(mitm_view.head(15))

    clusters_story = [0, 2, 4, 5, 8, 9, 10, 11, 12, 14, 26, 28, 29]
    df_focus = df_test[df_test["cluster_kmeans30"].isin(clusters_story)].copy()

    num_feats = [
        "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes",
        "dst_bytes", "missed_bytes", "duration"
    ]

    for c in ["src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "dst_bytes", "missed_bytes", "duration"]:
        if c in df_focus.columns:
            df_focus[c] = np.log1p(df_focus[c])

    numeric_stats_feats = [c for c in num_feats if c in df_focus.columns]
    if numeric_stats_feats:
        stats = df_focus.groupby("cluster_kmeans30")[numeric_stats_feats].agg(["count", "median", "mean"]).round(3)
        display(stats)

    if "http_status_code" in df_focus.columns:
        http_status_summary = df_focus.groupby("cluster_kmeans30")["http_status_code"].apply(lambda s: topk(s.astype(str), 5))
        display(http_status_summary.to_frame("top_http_status_code"))

    ports_summary = []
    for cid, g in df_focus.groupby("cluster_kmeans30"):
        ports_summary.append({
            "cluster": cid,
            "size": len(g),
            "dominant_type": g["type"].value_counts().idxmax(),
            "purity": (g["type"].value_counts().max() / len(g)),
            "top_src_ports": topk(g["src_port"], 5) if "src_port" in g else None,
            "top_dst_ports": topk(g["dst_port"], 5) if "dst_port" in g else None,
        })

    ports_summary = pd.DataFrame(ports_summary).sort_values(["purity", "size"], ascending=[False, False])
    display(ports_summary)

    print(f"Salvando artefato intermediário em: {intermediate_path}")
    save_dataframe(df, intermediate_path)
    print("Preparação concluída com sucesso.")


TARGET_CONFIG = {
    "cluster": {
        "target_col": "cluster_kmeans30",
        "title": "CLUSTER prediction",
    },
    "type": {
        "target_col": "type",
        "title": "TYPE prediction",
    },
    "binary": {
        "target_col": "label",
        "title": "BINARY prediction",
    },
}


def run_classification_stage(
    intermediate_path: str,
    target: str,
    cv_splits: int,
    rf_n_jobs: int,
    cv_n_jobs: int,
    metrics_output_dir: str | None,
) -> None:
    if target not in TARGET_CONFIG:
        raise ValueError(f"Target inválido: {target}")

    df = load_dataframe(intermediate_path)

    assert "cluster_kmeans30" in df.columns
    assert "type" in df.columns
    assert "label" in df.columns
    assert "split" in df.columns

    groups = build_groups(df)

    leak_cols = [
        "label", "type", "cluster_kmeans30",
        "cluster_name", "purity_valid", "purity_test",
        "split",
        "src_ip", "dst_ip",
        "src_port", "dst_port",
    ]

    feature_cols = [c for c in df.columns if c not in leak_cols]

    m_train = df["split"].eq("train")
    m_valid = df["split"].eq("valid")
    m_test = df["split"].eq("test")

    X_tr = df.loc[m_train, feature_cols]
    X_va = df.loc[m_valid, feature_cols]
    X_te = df.loc[m_test, feature_cols]

    target_col = TARGET_CONFIG[target]["target_col"]
    title = TARGET_CONFIG[target]["title"]

    if target == "cluster":
        y_tr = df.loc[m_train, target_col].astype(int)
        y_va = df.loc[m_valid, target_col].astype(int)
        y_te = df.loc[m_test, target_col].astype(int)
    else:
        y_tr = df.loc[m_train, target_col].astype(str)
        y_va = df.loc[m_valid, target_col].astype(str)
        y_te = df.loc[m_test, target_col].astype(str)

    print(X_tr.shape, X_va.shape, X_te.shape)
    print("Target:", target, "Train classes:", y_tr.nunique())

    num_cols = X_tr.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X_tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("string")
        X_va[c] = X_va[c].astype("string")
        X_te[c] = X_te[c].astype("string")

    preprocess_sup = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    rf_candidates = [
        {"name": "rf_baseline", "n_estimators": 150, "max_depth": None, "min_samples_leaf": 1},
        {"name": "rf_regularized", "n_estimators": 300, "max_depth": 30, "min_samples_leaf": 2},
    ]

    best_cfg = None
    best_valid_macro_f1 = -np.inf

    for cfg in rf_candidates:
        clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=10000,
            n_jobs=rf_n_jobs,
            class_weight="balanced_subsample",
        )
        pipe = Pipeline([
            ("prep", preprocess_sup),
            ("clf", clf),
        ])

        pipe.fit(X_tr, y_tr)
        pred_va = pipe.predict(X_va)
        valid_macro_f1 = f1_score(y_va, pred_va, average="macro")
        valid_acc = accuracy_score(y_va, pred_va)

        print(
            f"VALID {cfg['name']}: macroF1={valid_macro_f1:.4f} | acc={valid_acc:.4f} | "
            f"n_estimators={cfg['n_estimators']} | max_depth={cfg['max_depth']} | "
            f"min_samples_leaf={cfg['min_samples_leaf']}"
        )

        if valid_macro_f1 > best_valid_macro_f1:
            best_valid_macro_f1 = valid_macro_f1
            best_cfg = cfg

    print(f"Melhor configuração no VALID: {best_cfg['name']} (macroF1={best_valid_macro_f1:.4f})")

    clf = RandomForestClassifier(
        n_estimators=best_cfg["n_estimators"],
        max_depth=best_cfg["max_depth"],
        min_samples_leaf=best_cfg["min_samples_leaf"],
        random_state=10000,
        n_jobs=rf_n_jobs,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline([
        ("prep", preprocess_sup),
        ("clf", clf),
    ])

    groups_tr = groups.loc[m_train]
    try:
        actual_cv_splits = resolve_group_cv_splits(y_tr, groups_tr, cv_splits, f"classify/{target}")
    except ValueError as e:
        print(f"CV macroF1 (train): pulado; VALID/TEST continuam. Motivo: {e}")
        print_group_cv_diagnostics(y_tr, groups_tr, min_groups=2)
    else:
        cv = StratifiedGroupKFold(n_splits=actual_cv_splits, shuffle=True, random_state=10000)
        cv_scores = cross_val_score(
            pipe,
            X_tr,
            y_tr,
            cv=cv,
            groups=groups_tr,
            scoring="f1_macro",
            n_jobs=cv_n_jobs,
        )
        print(f"CV macroF1 (train, {actual_cv_splits}-fold):", cv_scores.mean(), "+/-", cv_scores.std())

    X_trva = pd.concat([X_tr, X_va], axis=0, copy=False)
    y_trva = pd.concat([y_tr, y_va], axis=0, copy=False)

    pipe.fit(X_trva, y_trva)

    pred_te = pipe.predict(X_te)
    print(f"\n=== {title} (TEST) ===")
    print("acc:", accuracy_score(y_te, pred_te))
    print("balanced_acc:", balanced_accuracy_score(y_te, pred_te))
    print("macroF1:", f1_score(y_te, pred_te, average="macro", zero_division=0))
    print("weightedF1:", f1_score(y_te, pred_te, average="weighted", zero_division=0))
    print("MCC:", matthews_corrcoef(y_te.astype(str), pd.Series(pred_te).astype(str)))
    print(classification_report(y_te, pred_te, digits=4, zero_division=0))
    save_supervised_evaluation(y_te, pred_te, target, metrics_output_dir)

    print("Classificação concluída com sucesso.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "classify"], required=True)
    parser.add_argument(
        "--data-path",
        default="dados/train_test_network.parquet",
        help="Arquivo .parquet, diretório com partições Network_dataset_*.parquet, ou padrão com curinga.",
    )
    parser.add_argument(
        "--intermediate-path",
        default="dados/intermediarios/train_test_network_with_clusters_k30.parquet",
        help="Artefato intermediário enriquecido com split e cluster_kmeans30.",
    )
    parser.add_argument(
        "--target",
        choices=["cluster", "type", "binary"],
        help="Alvo supervisionado (usado em --stage classify).",
    )
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=12,
        help="Paralelismo interno da RandomForest.",
    )
    parser.add_argument(
        "--cv-n-jobs",
        type=int,
        default=1,
        help="Paralelismo do cross_val_score. Mantido em 1 por padrão para evitar paralelismo aninhado.",
    )
    parser.add_argument(
        "--log-file",
        default="resultados/execucoes/pipeline_resultados.txt",
        help="Arquivo de log da etapa executada.",
    )
    parser.add_argument(
        "--metrics-output-dir",
        default="resultados/metricas_supervisionadas",
        help="Diretório para salvar métricas por classe e matrizes de confusão da etapa classify.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage == "classify" and not args.target:
        raise ValueError("Em --stage classify, você precisa informar --target {cluster,type,binary}.")

    with open(args.log_file, "w", encoding="utf-8") as log_f:
        with redirect_stdout(log_f), redirect_stderr(log_f):
            print("Python:", sys.executable)
            print("Script version:", SCRIPT_VERSION)
            print("Stage:", args.stage)
            print("Log file:", args.log_file)
            print("Data path:", args.data_path)
            print("Intermediate path:", args.intermediate_path)

            if args.stage == "prepare":
                run_prepare_stage(args.data_path, args.intermediate_path)
            else:
                run_classification_stage(
                    intermediate_path=args.intermediate_path,
                    target=args.target,
                    cv_splits=args.cv_splits,
                    rf_n_jobs=args.rf_n_jobs,
                    cv_n_jobs=args.cv_n_jobs,
                    metrics_output_dir=args.metrics_output_dir,
                )


if __name__ == "__main__":
    main()
