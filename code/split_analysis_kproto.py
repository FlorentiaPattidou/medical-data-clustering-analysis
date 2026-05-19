# ============================================================
# split_analysis_kproto.py
# KPrototypes (k=2) — SPLIT ANALYSIS
# Symptomatic | Asymptomatic — separate cluster analysis
#
# PCA plot per group:
#   Title  = "Symptomatic Patients" / "Asymptomatic Patients"
#   Color  = cluster (blue C0 / orange C1)
#   Edge   = follow-up time:
#              RED   = lastfu <= 6 sem. (<=3 years)
#              GREEN = lastfu >  6 sem. (>3 years)
#
# Outputs: outputs_split_kproto/
#   symptomatic/
#   asymptomatic/
# ============================================================

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Sequence, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from kmodes.kprototypes import KPrototypes

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    input_file: str = "acsrsJba_EventsLastFuAge.xlsx"
    base_dir:   str = "outputs_split_kproto"

    target_col:    str = "patstat"
    lastfu_col:    str = "lastfu"
    lastfu_cutoff: int = 6          # <=6 εξάμηνα = <=3 χρόνια

    cat_features: Tuple[str, ...] = (
        "stenosis", "gsm", "plarea", "dwa", "ctiastr", "jba"
    )
    num_features: Tuple[str, ...] = (
        "STN", "GSMN", "PLAREAN", "DWAN", "CTIASTRN", "JBAN", "age"
    )

    k:          int   = 2
    km_init:    str   = "Huang"
    km_n_init:  int   = 20
    gamma:      float = 0.5

    k_min:      int = 2
    k_max:      int = 8
    diag_n_init: int = 5

    silhouette_sample_size: int = 2000
    stability_runs:         int = 20
    random_state:           int = 42

    pca_figsize:  Tuple[int, int] = (10, 7)
    dist_figsize: Tuple[int, int] = (10, 5)
    diag_figsize: Tuple[int, int] = (10, 5)
    dpi: int = 250


# =========================
# LOGGING / IO
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_df(df: pd.DataFrame, path: str, index: bool = False) -> None:
    df.to_csv(path, index=index)


# =========================
# DATA PREP
# =========================
def prepare_data(
    df: pd.DataFrame,
    cat_features: Sequence[str],
    num_features: Sequence[str],
) -> Tuple[np.ndarray, List[int], RobustScaler]:
    all_feats = list(num_features) + list(cat_features)
    df_sub    = df[all_feats].dropna()

    scaler = RobustScaler()
    X_num  = scaler.fit_transform(df_sub[list(num_features)].values.astype(float))
    X_cat  = df_sub[list(cat_features)].astype(str).values

    X           = np.hstack([X_num, X_cat])
    cat_indices = list(range(len(num_features), len(all_feats)))
    return X, cat_indices, scaler


# =========================
# CATEGORY ORDERS
# =========================
def _category_order_for_feature(feat: str) -> List[str]:
    if feat.lower() == "jba":
        return ["low", "medium", "high", "vhigh"]
    return ["low", "medium", "high"]


# =========================
# PLOTS: CATEGORICAL DISTRIBUTIONS
# =========================
def save_cat_distributions(
    data_out: pd.DataFrame,
    out_dir: str,
    features: Sequence[str],
    figsize: Tuple[int, int],
    dpi: int,
) -> None:
    for feat in features:
        tab = pd.crosstab(data_out["Cluster"], data_out[feat]).sort_index()
        desired  = _category_order_for_feature(feat)
        existing = [c for c in desired if c in tab.columns]
        extras   = [c for c in tab.columns if c not in existing]
        tab = tab[existing + sorted(extras)]

        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(tab.index), dtype=float)
        for cat in tab.columns:
            vals = tab[cat].to_numpy()
            ax.bar(tab.index.astype(int), vals, bottom=bottom, label=str(cat))
            bottom += vals

        ax.set_title(f"{feat} distribution per cluster (KProto)")
        ax.set_xlabel("Cluster"); ax.set_ylabel("Count")
        ax.set_xticks(tab.index.astype(int))
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.legend(title=feat, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"07_{feat}_by_cluster.png"), dpi=dpi)
        plt.close(fig)


# =========================
# PLOTS: NUMERICAL BOXPLOTS
# =========================
def save_num_distributions(
    data_out: pd.DataFrame,
    out_dir: str,
    features: Sequence[str],
    figsize: Tuple[int, int],
    dpi: int,
) -> None:
    for feat in features:
        fig, ax = plt.subplots(figsize=figsize)
        cluster_ids = sorted(data_out["Cluster"].unique())
        groups = [data_out.loc[data_out["Cluster"] == cid, feat].dropna().values
                  for cid in cluster_ids]
        ax.boxplot(groups, tick_labels=[f"Cluster {cid}" for cid in cluster_ids])
        ax.set_title(f"{feat} by cluster (KProto)")
        ax.set_ylabel(feat)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"08_{feat}_by_cluster.png"), dpi=dpi)
        plt.close(fig)


# =========================
# PCA EXPLAINED VARIANCE
# =========================
def save_pca_explained_variance(
    explained_ratio: np.ndarray, out_dir: str, dpi: int, figsize: Tuple[int, int]
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    labels = [f"PC{i+1}" for i in range(len(explained_ratio))]
    ax.bar(labels, explained_ratio * 100.0)
    ax.set_title(f"PCA explained variance ({len(explained_ratio)} components)")
    ax.set_ylabel("Explained variance (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_pca_explained_variance.png"), dpi=dpi)
    plt.close(fig)


# =========================
# PCA PLOT
# color  = cluster
# edge   = follow-up time (red <=3yr, green >3yr)
# =========================
def save_pca_plot(
    X_num: np.ndarray,
    df_out: pd.DataFrame,
    clusters: np.ndarray,
    out_dir: str,
    group_label: str,
    k: int,
    lastfu_col: str,
    lastfu_cutoff: int,
    figsize: Tuple[int, int],
    figsize_var: Tuple[int, int],
    dpi: int,
) -> None:
    pca = PCA(n_components=2, svd_solver="full", random_state=0)
    Z   = pca.fit_transform(X_num)

    save_pca_explained_variance(pca.explained_variance_ratio_, out_dir, dpi, figsize_var)

    CLUSTER_FACE = {0: "#2196F3", 1: "#FF5722", 2: "#4CAF50"}
    TIME_MARKER  = {"short": "o", "long": "^"}    # κύκλος=<=3χρ, τρίγωνο=>3χρ
    TIME_SIZE    = {"short": 100, "long": 110}
    TIME_LABEL   = {"short": "≤3 years (≤6 sem.)", "long": ">3 years (>6 sem.)"}

    lastfu_arr = df_out[lastfu_col].to_numpy()
    time_group = np.where(lastfu_arr <= lastfu_cutoff, "short", "long")

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=figsize)

    for cid in range(k):
        for tg in ("short", "long"):
            idx = (clusters == cid) & (time_group == tg)
            if not np.any(idx):
                continue
            pts = Z[idx].copy()
            pts += rng.normal(0, 0.02, pts.shape)
            ax.scatter(
                pts[:, 0], pts[:, 1],
                facecolors=CLUSTER_FACE[cid],
                edgecolors="black",
                linewidths=0.8,
                s=TIME_SIZE[tg],
                alpha=0.85,
                marker=TIME_MARKER[tg],
                zorder=5,
            )

    # cluster centers
    for cid in range(k):
        idx = clusters == cid
        if not np.any(idx):
            continue
        cx, cy = Z[idx].mean(axis=0)
        ax.scatter(cx, cy, marker="*", s=500, zorder=10,
                   facecolors=CLUSTER_FACE[cid],
                   edgecolors="black", linewidths=1.8)
        ax.text(cx + 0.05, cy + 0.05, f"C{cid}",
                fontsize=12, fontweight="bold",
                color=CLUSTER_FACE[cid], zorder=11,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # legends
    cluster_handles = [
        Patch(facecolor=CLUSTER_FACE[cid], edgecolor="black",
              linewidth=0.8, label=f"Cluster {cid}")
        for cid in range(k)
    ]
    time_handles = [
        Line2D([0], [0], marker=TIME_MARKER[tg], color="w",
               markerfacecolor="#777777",
               markeredgecolor="black",
               markeredgewidth=0.8, markersize=10,
               label=TIME_LABEL[tg])
        for tg in ("short", "long")
    ]

    leg1 = ax.legend(handles=cluster_handles, title="Cluster",
                     loc="upper left", fontsize=9, title_fontsize=9, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=time_handles, title="Follow-up Time",
              loc="lower right", fontsize=9, title_fontsize=9, framealpha=0.9)

    ax.set_title(f"{group_label}\nPCA 2D projection (KPrototypes)  |  "
                 "color = cluster  |  marker = follow-up time  (● ≤3yr  ▲ >3yr)",
                 fontsize=11, pad=10)
    # clip axes: κόβουμε outliers που τεντώνουν τον άξονα
    x_lo, x_hi = np.percentile(Z[:, 0], [2.5, 97.5])
    y_lo, y_hi = np.percentile(Z[:, 1], [2.5, 97.5])
    margin = 0.4
    ax.set_xlim(x_lo - margin, x_hi + margin)
    ax.set_ylim(y_lo - margin, y_hi + margin)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_pca_clusters_with_centers.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================
# DIAGNOSTICS
# =========================
def save_diagnostics(
    X: np.ndarray, cat_indices: List[int], out_dir: str,
    k_min: int, k_max: int, km_init: str, km_n_init: int,
    gamma: float, dpi: int, figsize: Tuple[int, int],
) -> None:
    ks = list(range(k_min, k_max + 1))
    costs, silhouettes, dbis, chis = [], [], [], []
    X_num = X[:, :X.shape[1] - len(cat_indices)].astype(float)

    for kk in ks:
        kp = KPrototypes(n_clusters=kk, init=km_init, n_init=km_n_init,
                         verbose=0, gamma=gamma)
        labels = kp.fit_predict(X, categorical=cat_indices)
        costs.append(float(kp.cost_))

        if len(np.unique(labels)) <= 1:
            silhouettes.append(np.nan); dbis.append(np.nan); chis.append(np.nan)
            continue

        silhouettes.append(float(silhouette_score(X_num, labels)))
        dbis.append(float(davies_bouldin_score(X_num, labels)))
        chis.append(float(calinski_harabasz_score(X_num, labels)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, costs, marker="o")
    ax.set_title("Elbow curve (KPrototypes cost)")
    ax.set_xlabel("k"); ax.set_ylabel("Cost")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_elbow_kproto_cost.png"), dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, silhouettes, marker="o", label="Silhouette")
    ax.plot(ks, dbis,        marker="o", label="Davies–Bouldin")
    ax.plot(ks, chis,        marker="o", label="Calinski–Harabasz")
    ax.set_title("Clustering quality metrics vs k (KPrototypes)")
    ax.set_xlabel("k")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_quality_metrics_vs_k.png"), dpi=dpi)
    plt.close(fig)


def save_cluster_sizes_plot(
    clusters: np.ndarray, out_dir: str, k: int, dpi: int, figsize: Tuple[int, int]
) -> None:
    counts = pd.Series(clusters).value_counts().sort_index().reindex(range(k), fill_value=0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index.astype(int), counts.values)
    ax.set_title(f"Cluster sizes (KProto k={k})")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Count")
    ax.set_xticks(counts.index.astype(int))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_cluster_sizes.png"), dpi=dpi)
    plt.close(fig)


# =========================
# FIT + STABILIZE
# =========================
def fit_kprototypes(
    X: np.ndarray, cat_indices: List[int], cfg: Config
) -> Tuple[KPrototypes, np.ndarray]:
    kp = KPrototypes(n_clusters=cfg.k, init=cfg.km_init,
                     n_init=cfg.km_n_init, verbose=0, gamma=cfg.gamma)
    return kp, kp.fit_predict(X, categorical=cat_indices)


def stabilize_labels_by_size(clusters: np.ndarray, k: int) -> np.ndarray:
    counts  = pd.Series(clusters).value_counts().reindex(range(k), fill_value=0)
    order   = list(counts.sort_values(ascending=False).index)
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[int(c)] for c in clusters], dtype=int)


# =========================
# EVALUATION
# =========================
def evaluate_internal(
    X_num: np.ndarray, clusters: np.ndarray,
    silhouette_sample_size: int, random_state: int,
) -> Dict[str, Any]:
    n, k = len(clusters), len(np.unique(clusters))
    if k <= 1:
        return {"n": n, "k": k, "silhouette": np.nan,
                "davies_bouldin": np.nan, "calinski_harabasz": np.nan}

    rng = np.random.default_rng(random_state)
    if n > silhouette_sample_size:
        idx = rng.choice(n, size=silhouette_sample_size, replace=False)
        sil = float(silhouette_score(X_num[idx], clusters[idx]))
    else:
        sil = float(silhouette_score(X_num, clusters))

    return {
        "n": int(n), "k": int(k),
        "silhouette":       sil,
        "davies_bouldin":   float(davies_bouldin_score(X_num, clusters)),
        "calinski_harabasz": float(calinski_harabasz_score(X_num, clusters)),
    }


def evaluate_stability(
    X: np.ndarray, cat_indices: List[int],
    k: int, km_init: str, km_n_init: int,
    gamma: float, n_runs: int, random_state: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    all_labels: List[np.ndarray] = []

    for run_i in range(n_runs):
        perm = rng.permutation(len(X))
        Xp   = X[perm]
        kp   = KPrototypes(n_clusters=k, init=km_init, n_init=km_n_init,
                           verbose=0, gamma=gamma, random_state=run_i)
        yp   = kp.fit_predict(Xp, categorical=cat_indices)
        y    = np.empty_like(yp)
        y[perm] = yp
        all_labels.append(y)

    aris = [
        adjusted_rand_score(all_labels[i], all_labels[j])
        for i in range(n_runs) for j in range(i + 1, n_runs)
    ]
    aris = np.asarray(aris, dtype=float)
    return {
        "n_runs": n_runs,
        "pairwise_ari_mean": float(np.mean(aris)),
        "pairwise_ari_min":  float(np.min(aris)),
        "pairwise_ari_max":  float(np.max(aris)),
    }


# =========================
# CORE GROUP ANALYSIS
# =========================
def run_group_analysis(
    df_group: pd.DataFrame,
    group_name: str,
    group_label: str,
    cfg: Config,
) -> None:
    out_dir = os.path.join(cfg.base_dir, group_name)
    ensure_dir(out_dir)
    logging.info("--- %s (n=%d) ---", group_label, len(df_group))

    all_feats = list(cfg.num_features) + list(cfg.cat_features)
    df_clean  = df_group.dropna(subset=all_feats).reset_index(drop=True)
    logging.info("Rows after dropna: %d", len(df_clean))

    X, cat_indices, scaler = prepare_data(df_clean, cfg.cat_features, cfg.num_features)
    X_num = X[:, :len(cfg.num_features)].astype(float)

    # diagnostics
    save_diagnostics(
        X=X, cat_indices=cat_indices, out_dir=out_dir,
        k_min=cfg.k_min, k_max=cfg.k_max,
        km_init=cfg.km_init, km_n_init=cfg.diag_n_init,
        gamma=cfg.gamma, dpi=cfg.dpi, figsize=cfg.diag_figsize,
    )

    # fit
    kp_model, clusters = fit_kprototypes(X, cat_indices, cfg)
    clusters = stabilize_labels_by_size(clusters, cfg.k)

    df_out = df_clean.copy()
    df_out["Cluster"] = clusters

    # summary
    counts  = df_out["Cluster"].value_counts().sort_index().reindex(range(cfg.k), fill_value=0)
    summary = pd.DataFrame({"Count": counts})
    summary["Percent_%"] = (summary["Count"] / summary["Count"].sum() * 100).round(4)

    save_df(df_out,  os.path.join(out_dir, f"{group_name}_cluster_assignments.csv"))
    save_df(summary, os.path.join(out_dir, f"{group_name}_cluster_summary.csv"), index=True)

    # cluster means per numeric feature
    means = df_out.groupby("Cluster")[list(cfg.num_features)].mean().round(2)
    means.to_csv(os.path.join(out_dir, f"{group_name}_cluster_num_means.csv"))
    logging.info("Cluster means:\n%s", means.to_string())

    # plots
    save_cluster_sizes_plot(clusters, out_dir, cfg.k, cfg.dpi, cfg.diag_figsize)

    save_pca_plot(
        X_num=X_num, df_out=df_out, clusters=clusters,
        out_dir=out_dir, group_label=group_label, k=cfg.k,
        lastfu_col=cfg.lastfu_col, lastfu_cutoff=cfg.lastfu_cutoff,
        figsize=cfg.pca_figsize, figsize_var=cfg.diag_figsize, dpi=cfg.dpi,
    )

    save_cat_distributions(df_out, out_dir, cfg.cat_features, cfg.dist_figsize, cfg.dpi)
    save_num_distributions(df_out, out_dir, cfg.num_features, cfg.dist_figsize, cfg.dpi)

    # evaluation
    internal  = evaluate_internal(X_num, clusters,
                                  cfg.silhouette_sample_size, cfg.random_state)
    stability = evaluate_stability(X, cat_indices, cfg.k, cfg.km_init,
                                   cfg.km_n_init, cfg.gamma,
                                   cfg.stability_runs, cfg.random_state)

    pd.DataFrame([{"group": group_name, **internal}]).to_csv(
        os.path.join(out_dir, "metrics_internal.csv"), index=False)
    pd.DataFrame([{"group": group_name, **stability}]).to_csv(
        os.path.join(out_dir, "metrics_stability.csv"), index=False)

    logging.info("Internal:  %s", internal)
    logging.info("Stability: %s", stability)


# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    cfg = Config()
    ensure_dir(cfg.base_dir)

    logging.info("Reading: %s", cfg.input_file)
    df = pd.read_excel(cfg.input_file)
    df = df[df[cfg.target_col].isin(["sympt", "asympt"])].copy()

    df_sympt  = df[df[cfg.target_col] == "sympt"].copy()
    df_asympt = df[df[cfg.target_col] == "asympt"].copy()

    logging.info("Total: %d | Symptomatic: %d | Asymptomatic: %d",
                 len(df), len(df_sympt), len(df_asympt))

    run_group_analysis(df_sympt,  "symptomatic",  "Symptomatic Patients",  cfg)
    run_group_analysis(df_asympt, "asymptomatic", "Asymptomatic Patients", cfg)

    logging.info("Done. Results in: %s", cfg.base_dir)


if __name__ == "__main__":
    main()