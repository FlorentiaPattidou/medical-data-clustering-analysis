# ============================================================
# patients_analysis_full_kmodes.py
# KMODES (k=2) + PLOTS + EVALUATION for ALL PATIENTS (full dataset)
#
# Runs full analysis into: outputs_k2_full/
#
# Produces:
#   01_pca_explained_variance.png
#   02_elbow_kmodes_cost.png
#   03_quality_metrics_vs_k.png
#   04_cluster_sizes.png
#   05_pca_clusters_with_centers_status.png
#   07_<feature>_by_cluster.png
#   metrics_internal.csv
#   metrics_stability.csv
#   full_cluster_assignments.csv
#   full_cluster_summary.csv
#   full_cluster_modes.csv
#
# Notes:
# - Clustering uses KModes on categorical features (strings).
# - PCA is ONLY for visualization (OrdinalEncoder).
# - Centers in PCA are mean of points per cluster (visual centers).
# - Elbow uses KModes cost_ (NOT KMeans inertia).
# - Cluster labels are stabilized by sorting clusters by size (desc).
# ============================================================

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Sequence, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kmodes.kmodes import KModes

from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
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
    base_dir: str = "outputs_k2_full"

    # features used for clustering
    features: Tuple[str, ...] = ("stenosis", "gsm", "plarea", "dwa", "ctiastr", "jba")

    # optional: keep only rows where patstat is valid
    filter_patstat: bool = True
    patstat_col: str = "patstat"
    patstat_values: Tuple[str, ...] = ("sympt", "asympt")

    # kmodes
    k: int = 2
    diag_n_init: int = 5       #diagnostic plots (elbow/metrics)
    km_init: str = "Huang"
    km_n_init: int = 20

    # diagnostics (k search)
    k_min: int = 2
    k_max: int = 10

    # evaluation
    silhouette_sample_size: int = 2000
    stability_runs: int = 20
    random_state: int = 42

    # plots
    pca_figsize: Tuple[int, int] = (10, 7)
    dist_figsize: Tuple[int, int] = (10, 5)
    diag_figsize: Tuple[int, int] = (10, 5)
    dpi: int = 250


# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# =========================
# IO
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_df(df: pd.DataFrame, path: str, index: bool = False) -> None:
    df.to_csv(path, index=index)


# =========================
# CATEGORY ORDERS (plots)
# =========================
def _category_order_for_feature(feat: str) -> List[str]:
    base = ["low", "medium", "high"]
    if feat.lower() == "jba":
        return ["low", "medium", "high", "vhigh"]
    return base


# =========================
# PLOTS: FEATURE DISTRIBUTIONS
# =========================
def save_feature_distributions_clean(
    data_out: pd.DataFrame,
    out_dir: str,
    features: Sequence[str],
    figsize: Tuple[int, int],
    dpi: int,
) -> None:
    for feat in features:
        tab = pd.crosstab(data_out["Cluster"], data_out[feat]).sort_index()
        tab = tab.reindex(sorted(tab.index), fill_value=0)

        desired = _category_order_for_feature(feat)
        existing = [c for c in desired if c in tab.columns]
        extras = [c for c in tab.columns if c not in existing]
        tab = tab[existing + sorted(extras, key=lambda x: str(x))]

        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(tab.index), dtype=float)

        for cat in tab.columns:
            vals = tab[cat].to_numpy()
            ax.bar(tab.index.astype(int), vals, bottom=bottom, label=str(cat))
            bottom += vals

        ax.set_title(f"{feat} distribution per cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
        ax.set_xticks(tab.index.astype(int))
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.legend(title=feat, bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"07_{feat}_by_cluster.png"), dpi=dpi)
        plt.close(fig)


# =========================
# PCA
# =========================
def save_pca_explained_variance(
    explained_ratio: np.ndarray,
    out_dir: str,
    dpi: int,
    figsize: Tuple[int, int],
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


def save_pca_clusters_with_centers_ordinal(
    X_cat: pd.DataFrame,
    clusters: np.ndarray,
    patstat: pd.Series,
    out_dir: str,
    k: int,
    figsize_scatter: Tuple[int, int],
    figsize_var: Tuple[int, int],
    dpi: int,
) -> None:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_enc = enc.fit_transform(X_cat)

    pca = PCA(n_components=2, svd_solver="full", random_state=0)
    Z = pca.fit_transform(X_enc)

    save_pca_explained_variance(
        explained_ratio=pca.explained_variance_ratio_,
        out_dir=out_dir,
        dpi=dpi,
        figsize=figsize_var,
    )

    # ── Παλέτα ──────────────────────────────────────────────────────────
    CLUSTER_COLORS = {0: "#2196F3", 1: "#FF5722", 2: "#4CAF50"}

    STATUS_OFFSET = {
        "sympt":  np.array([ 0.04,  0.0]),
        "asympt": np.array([-0.04,  0.0]),
    }

    STATUS_CFG = {
        "sympt":  {"marker": "o", "size": 120, "edge": "black", "lw": 1.2, "alpha": 0.90, "label": "Symptomatic"},
        "asympt": {"marker": "X", "size": 160, "edge": "black", "lw": 0.8, "alpha": 0.80, "label": "Asymptomatic"},
    }

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=figsize_scatter)

    # ── Scatter ──────────────────────────────────────────────────────────
    for cid in range(k):
        for status, scfg in STATUS_CFG.items():
            idx = (clusters == cid) & (patstat.to_numpy() == status)
            if not np.any(idx):
                continue

            pts = Z[idx].copy()
            pts += STATUS_OFFSET[status]
            pts += rng.normal(0, 0.018, pts.shape)

            ax.scatter(
                pts[:, 0], pts[:, 1],
                color=CLUSTER_COLORS[cid],
                marker=scfg["marker"],
                s=scfg["size"],
                alpha=scfg["alpha"],
                edgecolors=scfg["edge"],
                linewidths=scfg["lw"],
                zorder=5,
            )

    # ── Cluster centers ──────────────────────────────────────────────────
    for cid in range(k):
        idx = clusters == cid
        if not np.any(idx):
            continue
        cx, cy = Z[idx].mean(axis=0)
        ax.scatter(cx, cy,
                   marker="*", s=500, zorder=10,
                   color=CLUSTER_COLORS[cid],
                   edgecolors="black", linewidths=1.5)
        ax.text(cx + 0.06, cy + 0.06, f"C{cid}",
                fontsize=12, fontweight="bold",
                color=CLUSTER_COLORS[cid], zorder=11,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.7))

    # ── Legends ──────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cluster_handles = [
        Patch(facecolor=CLUSTER_COLORS[cid], edgecolor="black", linewidth=0.8,
              label=f"Cluster {cid}")
        for cid in range(k)
    ]
    status_handles = [
        Line2D([0], [0], marker=scfg["marker"], color="w",
               markerfacecolor="#555555",
               markeredgecolor="black" if scfg["edge"] == "black" else "#555555",
               markeredgewidth=1.2, markersize=9,
               label=scfg["label"])
        for status, scfg in STATUS_CFG.items()
    ]

    leg1 = ax.legend(handles=cluster_handles, title="Cluster",
                     loc="upper left", fontsize=9, title_fontsize=9,
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=status_handles, title="Patient Status",
              loc="lower right", fontsize=9, title_fontsize=9,
              framealpha=0.9)

    ax.set_title("PCA 2D projection  |  color = cluster    marker = patient status",
                 fontsize=11, pad=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.2)
    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "05_pca_clusters_with_centers_status.png"), dpi=dpi
    )
    plt.close(fig)


# =========================
# DIAGNOSTICS (KModes)
# =========================
def _encode_to_int_matrix(X_cat: pd.DataFrame) -> np.ndarray:
    X_int = np.zeros((len(X_cat), X_cat.shape[1]), dtype=int)
    for j, col in enumerate(X_cat.columns):
        codes, _ = pd.factorize(X_cat[col].astype(str), sort=True)
        X_int[:, j] = codes
    return X_int


def save_kmodes_diagnostics(
    X_cat: pd.DataFrame,
    out_dir: str,
    k_min: int,
    k_max: int,
    km_init: str,
    km_n_init: int,
    silhouette_sample_size: int,
    random_state: int,
    dpi: int,
    figsize: Tuple[int, int],
) -> None:
    rng = np.random.default_rng(random_state)

    ks = list(range(k_min, k_max + 1))
    costs: List[float] = []
    silhouettes: List[float] = []
    dbis: List[float] = []
    chis: List[float] = []

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_ord = enc.fit_transform(X_cat).astype(float)

    X_int_full = _encode_to_int_matrix(X_cat)

    for kk in ks:
        km = KModes(n_clusters=kk, init=km_init, n_init=km_n_init, verbose=0)
        labels = km.fit_predict(X_cat)

        costs.append(float(km.cost_))

        if len(np.unique(labels)) <= 1:
            silhouettes.append(np.nan)
            dbis.append(np.nan)
            chis.append(np.nan)
            continue

        dbis.append(float(davies_bouldin_score(X_ord, labels)))
        chis.append(float(calinski_harabasz_score(X_ord, labels)))

        n = len(labels)
        if n > silhouette_sample_size:
            idx = rng.choice(n, size=silhouette_sample_size, replace=False)
            sil = silhouette_score(X_int_full[idx], labels[idx], metric="hamming")
        else:
            sil = silhouette_score(X_int_full, labels, metric="hamming")
        silhouettes.append(float(sil))

    # 02 elbow
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, costs, marker="o")
    ax.set_title("Elbow curve (KModes cost)")
    ax.set_xlabel("k")
    ax.set_ylabel("KModes cost (sum of mismatches)")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_elbow_kmodes_cost.png"), dpi=dpi)
    plt.close(fig)

    # 03 metrics
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, silhouettes, marker="o", label="Silhouette (Hamming)")
    ax.plot(ks, dbis, marker="o", label="Davies–Bouldin (ordinal)")
    ax.plot(ks, chis, marker="o", label="Calinski–Harabasz (ordinal)")
    ax.set_title("Clustering quality metrics vs k (KModes labels)")
    ax.set_xlabel("k")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_quality_metrics_vs_k.png"), dpi=dpi)
    plt.close(fig)


def save_cluster_sizes_plot(
    clusters: np.ndarray,
    out_dir: str,
    k: int,
    dpi: int,
    figsize: Tuple[int, int],
) -> None:
    counts = pd.Series(clusters).value_counts().sort_index().reindex(range(k), fill_value=0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index.astype(int), counts.values)
    ax.set_title(f"Cluster sizes (k={k})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_xticks(counts.index.astype(int))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_cluster_sizes.png"), dpi=dpi)
    plt.close(fig)


# =========================
# FIT + STABILIZE LABELS
# =========================
def fit_kmodes(X_cat: pd.DataFrame, cfg: Config) -> Tuple[KModes, np.ndarray]:
    km = KModes(n_clusters=cfg.k, init=cfg.km_init, n_init=cfg.km_n_init, verbose=0)
    clusters = km.fit_predict(X_cat)
    return km, clusters


def build_summary_tables(
    df_with_clusters: pd.DataFrame,
    features: Sequence[str],
    k: int,
    km_model: KModes,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts = df_with_clusters["Cluster"].value_counts().sort_index().reindex(range(k), fill_value=0)
    summary = pd.DataFrame({"Count": counts})
    summary["Percent_%"] = (summary["Count"] / summary["Count"].sum() * 100).round(4)

    modes = pd.DataFrame(km_model.cluster_centroids_, columns=list(features))
    modes.insert(0, "Cluster", list(range(k)))
    return summary, modes


def stabilize_labels_by_size(
    clusters: np.ndarray,
    modes: pd.DataFrame,
    k: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    counts = pd.Series(clusters).value_counts().reindex(range(k), fill_value=0)
    order = list(counts.sort_values(ascending=False).index)
    mapping = {old: new for new, old in enumerate(order)}

    new_clusters = np.array([mapping[int(c)] for c in clusters], dtype=int)

    modes2 = modes.copy()
    modes2["Cluster"] = modes2["Cluster"].map(mapping)
    modes2 = modes2.sort_values("Cluster").reset_index(drop=True)
    return new_clusters, modes2


# =========================
# EVALUATION
# =========================
def evaluate_internal_metrics(
    df_all: pd.DataFrame,
    features: Sequence[str],
    cluster_col: str,
    silhouette_sample_size: int,
    random_state: int,
) -> Dict[str, Any]:
    labels = df_all[cluster_col].to_numpy()
    n = len(labels)
    k = int(len(np.unique(labels)))

    X_cat = df_all.loc[:, features].astype(str)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_ord = enc.fit_transform(X_cat).astype(float)

    dbi = float(davies_bouldin_score(X_ord, labels)) if k > 1 else np.nan
    chi = float(calinski_harabasz_score(X_ord, labels)) if k > 1 else np.nan

    X_int = _encode_to_int_matrix(X_cat)
    if k > 1:
        if n > silhouette_sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=silhouette_sample_size, replace=False)
            sil = float(silhouette_score(X_int[idx], labels[idx], metric="hamming"))
        else:
            sil = float(silhouette_score(X_int, labels, metric="hamming"))
    else:
        sil = np.nan

    rng = np.random.default_rng(random_state)
    avg_within = []
    for c in np.unique(labels):
        idxc = np.where(labels == c)[0]
        if len(idxc) <= 1:
            continue
        take = idxc
        if len(take) > 300:
            take = rng.choice(take, size=300, replace=False)
        Xc = X_int[take]
        D = (Xc[:, None, :] != Xc[None, :, :]).mean(axis=2)
        tri = D[np.triu_indices_from(D, k=1)]
        avg_within.append(float(tri.mean()) if tri.size else 0.0)

    avg_within_hamming = float(np.mean(avg_within)) if avg_within else np.nan

    return {
        "n": int(n),
        "k": int(k),
        "silhouette_hamming": sil,
        "davies_bouldin_ordinal": dbi,
        "calinski_harabasz_ordinal": chi,
        "avg_within_hamming": avg_within_hamming,
    }


def evaluate_stability_kmodes(
    df_all: pd.DataFrame,
    features: Sequence[str],
    k: int,
    km_init: str,
    km_n_init: int,
    n_runs: int,
    random_state: int,
) -> Dict[str, Any]:
    X_cat = df_all.loc[:, features].astype(str)
    rng = np.random.default_rng(random_state)

    all_labels: List[np.ndarray] = []
    for run_i in range(n_runs):          # ← fix: run_i αντί για _
        perm = rng.permutation(len(X_cat))
        Xp = X_cat.iloc[perm].reset_index(drop=True)

        km = KModes(
            n_clusters=k,
            init=km_init,
            n_init=km_n_init,
            verbose=0,
            random_state=run_i,          # ← fix: deterministic ανά run
        )
        yp = km.fit_predict(Xp)

        y = np.empty_like(yp)
        y[perm] = yp
        all_labels.append(y)

    aris = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    aris = np.asarray(aris, dtype=float)
    return {
        "n_runs": int(n_runs),
        "pairwise_ari_mean": float(np.mean(aris)) if aris.size else np.nan,
        "pairwise_ari_min": float(np.min(aris)) if aris.size else np.nan,
        "pairwise_ari_max": float(np.max(aris)) if aris.size else np.nan,
    }


# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    cfg = Config()
    ensure_dir(cfg.base_dir)

    logging.info("Reading input: %s", cfg.input_file)
    df = pd.read_excel(cfg.input_file)

    if cfg.filter_patstat and cfg.patstat_col in df.columns:
        df = df[df[cfg.patstat_col].isin(list(cfg.patstat_values))].copy()

    logging.info("Total rows used: %d", len(df))

    X_cat = df.loc[:, cfg.features].astype(str)

    # diagnostics across k
    save_kmodes_diagnostics(
        X_cat=X_cat,
        out_dir=cfg.base_dir,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        km_init=cfg.km_init,
        km_n_init=cfg.diag_n_init,
        silhouette_sample_size=cfg.silhouette_sample_size,
        random_state=cfg.random_state,
        dpi=cfg.dpi,
        figsize=cfg.diag_figsize,
    )

    # fit KModes for chosen k
    km_model, clusters = fit_kmodes(X_cat, cfg)

    df_out = df.copy()
    df_out["Cluster"] = clusters

    summary, modes = build_summary_tables(df_out, cfg.features, cfg.k, km_model)

    # stabilize labels to reduce run-to-run flips
    clusters_stable, modes_stable = stabilize_labels_by_size(
        clusters=df_out["Cluster"].to_numpy(),
        modes=modes,
        k=cfg.k,
    )
    df_out["Cluster"] = clusters_stable

    # rebuild summary with stable labels
    counts = df_out["Cluster"].value_counts().sort_index().reindex(range(cfg.k), fill_value=0)
    summary = pd.DataFrame({"Count": counts})
    summary["Percent_%"] = (summary["Count"] / summary["Count"].sum() * 100).round(4)

    # save tables
    save_df(df_out, os.path.join(cfg.base_dir, "full_cluster_assignments.csv"), index=False)
    save_df(summary, os.path.join(cfg.base_dir, "full_cluster_summary.csv"), index=True)
    save_df(modes_stable, os.path.join(cfg.base_dir, "full_cluster_modes.csv"), index=False)

    # plots
    save_cluster_sizes_plot(
        clusters=df_out["Cluster"].to_numpy(),
        out_dir=cfg.base_dir,
        k=cfg.k,
        dpi=cfg.dpi,
        figsize=cfg.diag_figsize,
    )

    save_pca_clusters_with_centers_ordinal(
        X_cat=X_cat,
        clusters=df_out["Cluster"].to_numpy(),
        patstat=df_out[cfg.patstat_col],
        out_dir=cfg.base_dir,
        k=cfg.k,
        dpi=cfg.dpi,
        figsize_scatter=cfg.pca_figsize,
        figsize_var=cfg.diag_figsize,
    )

    save_feature_distributions_clean(
        data_out=df_out,
        out_dir=cfg.base_dir,
        features=cfg.features,
        figsize=cfg.dist_figsize,
        dpi=cfg.dpi,
    )

    # evaluation
    internal = evaluate_internal_metrics(
        df_all=df_out,
        features=cfg.features,
        cluster_col="Cluster",
        silhouette_sample_size=cfg.silhouette_sample_size,
        random_state=cfg.random_state,
    )
    stability = evaluate_stability_kmodes(
        df_all=df_out,
        features=cfg.features,
        k=cfg.k,
        km_init=cfg.km_init,
        km_n_init=cfg.km_n_init,
        n_runs=cfg.stability_runs,
        random_state=cfg.random_state,
    )

    pd.DataFrame([{"group": "full", **internal}]).to_csv(
        os.path.join(cfg.base_dir, "metrics_internal.csv"), index=False
    )
    pd.DataFrame([{"group": "full", **stability}]).to_csv(
        os.path.join(cfg.base_dir, "metrics_stability.csv"), index=False
    )

    logging.info("Done. Results saved in: %s", cfg.base_dir)


if __name__ == "__main__":
    main()