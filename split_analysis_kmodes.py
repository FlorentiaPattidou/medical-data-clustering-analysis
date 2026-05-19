# ============================================================
# split_analysis_kmodes.py
# KModes (k=2) — SPLIT ANALYSIS
# Symptomatic | Asymptomatic — separate cluster analysis
#
# PCA plot per group:
#   Title  = "Symptomatic Patients" / "Asymptomatic Patients"
#   Color  = cluster (blue C0 / orange C1)
#   Edge   = follow-up time:
#              RED   = lastfu <= 6 sem. (<=3 years)
#              GREEN = lastfu >  6 sem. (>3 years)
#
# Outputs: outputs_split_kmodes/
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
    base_dir:   str = "outputs_split_kmodes"

    target_col:     str = "patstat"
    lastfu_col:     str = "lastfu"
    lastfu_cutoff:  int = 6          # <=6 εξάμηνα = <=3 χρόνια

    features: Tuple[str, ...] = (
        "stenosis", "gsm", "plarea", "dwa", "ctiastr", "jba"
    )

    k:          int = 2
    km_init:    str = "Huang"
    km_n_init:  int = 20

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
# CATEGORY ORDERS
# =========================
def _category_order_for_feature(feat: str) -> List[str]:
    if feat.lower() == "jba":
        return ["low", "medium", "high", "vhigh"]
    return ["low", "medium", "high"]


# =========================
# PLOTS: FEATURE DISTRIBUTIONS
# =========================
def save_feature_distributions(
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
        extras   = [c for c in tab.columns if c not in existing]
        tab = tab[existing + sorted(extras)]

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
    X_cat: pd.DataFrame,
    df_out: pd.DataFrame,
    clusters: np.ndarray,
    out_dir: str,
    group_label: str,       # "Symptomatic Patients" / "Asymptomatic Patients"
    k: int,
    lastfu_col: str,
    lastfu_cutoff: int,
    figsize: Tuple[int, int],
    figsize_var: Tuple[int, int],
    dpi: int,
) -> None:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_enc = enc.fit_transform(X_cat)
    pca = PCA(n_components=2, svd_solver="full", random_state=0)
    Z = pca.fit_transform(X_enc)

    save_pca_explained_variance(pca.explained_variance_ratio_, out_dir, dpi, figsize_var)

    # ── Παλέτα ──────────────────────────────────────────────────────────
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

    # ── Legends ─────────────────────────────────────────────────────────
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

    ax.set_title(f"{group_label}\nPCA 2D projection (KModes)  |  "
                 "color = cluster  |  marker = follow-up time  (● ≤3yr  ▲ >3yr)",
                 fontsize=11, pad=10)
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
def _encode_to_int_matrix(X_cat: pd.DataFrame) -> np.ndarray:
    X_int = np.zeros((len(X_cat), X_cat.shape[1]), dtype=int)
    for j, col in enumerate(X_cat.columns):
        codes, _ = pd.factorize(X_cat[col].astype(str), sort=True)
        X_int[:, j] = codes
    return X_int


def save_diagnostics(
    X_cat: pd.DataFrame, out_dir: str,
    k_min: int, k_max: int, km_init: str, km_n_init: int,
    silhouette_sample_size: int, random_state: int,
    dpi: int, figsize: Tuple[int, int],
) -> None:
    rng = np.random.default_rng(random_state)
    ks = list(range(k_min, k_max + 1))
    costs, silhouettes, dbis, chis = [], [], [], []

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_ord     = enc.fit_transform(X_cat).astype(float)
    X_int_full = _encode_to_int_matrix(X_cat)

    for kk in ks:
        km = KModes(n_clusters=kk, init=km_init, n_init=km_n_init, verbose=0)
        labels = km.fit_predict(X_cat)
        costs.append(float(km.cost_))

        if len(np.unique(labels)) <= 1:
            silhouettes.append(np.nan); dbis.append(np.nan); chis.append(np.nan)
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

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, costs, marker="o")
    ax.set_title("Elbow curve (KModes cost)")
    ax.set_xlabel("k"); ax.set_ylabel("KModes cost")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_elbow_kmodes_cost.png"), dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, silhouettes, marker="o", label="Silhouette (Hamming)")
    ax.plot(ks, dbis,        marker="o", label="Davies–Bouldin")
    ax.plot(ks, chis,        marker="o", label="Calinski–Harabasz")
    ax.set_title("Clustering quality metrics vs k (KModes)")
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
    ax.set_title(f"Cluster sizes (KModes k={k})")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Count")
    ax.set_xticks(counts.index.astype(int))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_cluster_sizes.png"), dpi=dpi)
    plt.close(fig)


# =========================
# FIT + STABILIZE
# =========================
def fit_kmodes(X_cat: pd.DataFrame, cfg: Config) -> Tuple[KModes, np.ndarray]:
    km = KModes(n_clusters=cfg.k, init=cfg.km_init, n_init=cfg.km_n_init, verbose=0)
    return km, km.fit_predict(X_cat)


def stabilize_labels_by_size(
    clusters: np.ndarray, modes: pd.DataFrame, k: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    counts = pd.Series(clusters).value_counts().reindex(range(k), fill_value=0)
    order  = list(counts.sort_values(ascending=False).index)
    mapping = {old: new for new, old in enumerate(order)}
    new_clusters = np.array([mapping[int(c)] for c in clusters], dtype=int)
    modes2 = modes.copy()
    modes2["Cluster"] = modes2["Cluster"].map(mapping)
    modes2 = modes2.sort_values("Cluster").reset_index(drop=True)
    return new_clusters, modes2


# =========================
# EVALUATION
# =========================
def evaluate_internal(
    df_out: pd.DataFrame, features: Sequence[str],
    cluster_col: str, silhouette_sample_size: int, random_state: int,
) -> Dict[str, Any]:
    labels = df_out[cluster_col].to_numpy()
    n, k   = len(labels), int(len(np.unique(labels)))
    X_cat  = df_out.loc[:, features].astype(str)

    enc   = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_ord = enc.fit_transform(X_cat).astype(float)
    X_int = _encode_to_int_matrix(X_cat)

    dbi = float(davies_bouldin_score(X_ord, labels))  if k > 1 else np.nan
    chi = float(calinski_harabasz_score(X_ord, labels)) if k > 1 else np.nan

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
        take = idxc if len(idxc) <= 300 else rng.choice(idxc, size=300, replace=False)
        Xc = X_int[take]
        D  = (Xc[:, None, :] != Xc[None, :, :]).mean(axis=2)
        tri = D[np.triu_indices_from(D, k=1)]
        avg_within.append(float(tri.mean()) if tri.size else 0.0)

    return {
        "n": int(n), "k": int(k),
        "silhouette_hamming":      sil,
        "davies_bouldin_ordinal":  dbi,
        "calinski_harabasz_ordinal": chi,
        "avg_within_hamming": float(np.mean(avg_within)) if avg_within else np.nan,
    }


def evaluate_stability(
    df_group: pd.DataFrame, features: Sequence[str],
    k: int, km_init: str, km_n_init: int, n_runs: int, random_state: int,
) -> Dict[str, Any]:
    X_cat = df_group.loc[:, features].astype(str)
    rng   = np.random.default_rng(random_state)
    all_labels: List[np.ndarray] = []

    for run_i in range(n_runs):
        perm = rng.permutation(len(X_cat))
        Xp   = X_cat.iloc[perm].reset_index(drop=True)
        km   = KModes(n_clusters=k, init=km_init, n_init=km_n_init,
                      verbose=0, random_state=run_i)
        yp   = km.fit_predict(Xp)
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
    group_name: str,       # "symptomatic" / "asymptomatic"
    group_label: str,      # "Symptomatic Patients" / "Asymptomatic Patients"
    cfg: Config,
) -> None:
    out_dir = os.path.join(cfg.base_dir, group_name)
    ensure_dir(out_dir)
    logging.info("--- %s (n=%d) ---", group_label, len(df_group))

    X_cat = df_group.loc[:, cfg.features].astype(str)

    # diagnostics
    save_diagnostics(
        X_cat=X_cat, out_dir=out_dir,
        k_min=cfg.k_min, k_max=cfg.k_max,
        km_init=cfg.km_init, km_n_init=cfg.diag_n_init,
        silhouette_sample_size=cfg.silhouette_sample_size,
        random_state=cfg.random_state, dpi=cfg.dpi, figsize=cfg.diag_figsize,
    )

    # fit
    km_model, clusters = fit_kmodes(X_cat, cfg)
    df_out = df_group.copy()
    df_out["Cluster"] = clusters

    modes_df = pd.DataFrame(km_model.cluster_centroids_, columns=list(cfg.features))
    modes_df.insert(0, "Cluster", list(range(cfg.k)))
    clusters_stable, modes_stable = stabilize_labels_by_size(clusters, modes_df, cfg.k)
    df_out["Cluster"] = clusters_stable

    # summary
    counts  = df_out["Cluster"].value_counts().sort_index().reindex(range(cfg.k), fill_value=0)
    summary = pd.DataFrame({"Count": counts})
    summary["Percent_%"] = (summary["Count"] / summary["Count"].sum() * 100).round(4)

    save_df(df_out,       os.path.join(out_dir, f"{group_name}_cluster_assignments.csv"))
    save_df(summary,      os.path.join(out_dir, f"{group_name}_cluster_summary.csv"), index=True)
    save_df(modes_stable, os.path.join(out_dir, f"{group_name}_cluster_modes.csv"))

    # plots
    save_cluster_sizes_plot(clusters_stable, out_dir, cfg.k, cfg.dpi, cfg.diag_figsize)

    save_pca_plot(
        X_cat=X_cat,
        df_out=df_out,
        clusters=clusters_stable,
        out_dir=out_dir,
        group_label=group_label,
        k=cfg.k,
        lastfu_col=cfg.lastfu_col,
        lastfu_cutoff=cfg.lastfu_cutoff,
        figsize=cfg.pca_figsize,
        figsize_var=cfg.diag_figsize,
        dpi=cfg.dpi,
    )

    save_feature_distributions(df_out, out_dir, cfg.features, cfg.dist_figsize, cfg.dpi)

    # evaluation
    internal  = evaluate_internal(df_out, cfg.features, "Cluster",
                                  cfg.silhouette_sample_size, cfg.random_state)
    stability = evaluate_stability(df_group, cfg.features, cfg.k,
                                   cfg.km_init, cfg.km_n_init,
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