# ============================================================
# te2rules_all.py
# TE2Rules explainability for ALL subsets
# Χρησιμοποιεί ΜΟΝΟ categorical features (γρήγορο)
# RF: n_estimators=50, max_depth=5
# ============================================================

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from te2rules.explainer import ModelExplainer

MAPPINGS = {
    "stenosis": {"low": 0, "medium": 1, "high": 2},
    "gsm":      {"low": 0, "medium": 1, "high": 2},
    "plarea":   {"low": 0, "medium": 1, "high": 2},
    "dwa":      {"absent": 0, "present": 1},
    "ctiastr":  {"absent": 0, "present": 1},
    "jba":      {"low": 0, "medium": 1, "high": 2, "vhigh": 3},
}

FEATURES        = ["stenosis", "gsm", "plarea", "dwa", "ctiastr", "jba"]
K               = 2
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH    = 5

SUBSETS = [
    (
        "KModes — Full Dataset",
        "K_MODES/outputs_k2_full/full_cluster_assignments.csv",
        "te2rules_outputs/kmodes_full",
    ),
    (
        "KModes — Symptomatic",
        "K_MODES/outputs_split_kmodes/symptomatic/symptomatic_cluster_assignments.csv",
        "te2rules_outputs/kmodes_symptomatic",
    ),
    (
        "KModes — Asymptomatic",
        "K_MODES/outputs_split_kmodes/asymptomatic/asymptomatic_cluster_assignments.csv",
        "te2rules_outputs/kmodes_asymptomatic",
    ),
    (
        "KPrototypes — Full Dataset",
        "K_PROTOTYPES/outputs_kproto/full_cluster_assignments.csv",
        "te2rules_outputs/kproto_full",
    ),
    (
        "KPrototypes — Symptomatic",
        "K_PROTOTYPES/outputs_split_kproto/symptomatic/symptomatic_cluster_assignments.csv",
        "te2rules_outputs/kproto_symptomatic",
    ),
    (
        "KPrototypes — Asymptomatic",
        "K_PROTOTYPES/outputs_split_kproto/asymptomatic/asymptomatic_cluster_assignments.csv",
        "te2rules_outputs/kproto_asymptomatic",
    ),
]


def encode_features(df):
    X = df[FEATURES].copy()
    for col in FEATURES:
        X[col] = X[col].map(MAPPINGS[col])
    if X.isnull().any().any():
        print("  WARNING: unmapped values found")
        print(X[X.isnull().any(axis=1)])
    return X


def run_te2rules_for_cluster(X_values, y_binary, cluster_id, out_dir, label):
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42,
    )
    model.fit(X_values, y_binary)
    explainer = ModelExplainer(model, feature_names=FEATURES)
    rules = explainer.explain(X_values, y_binary)

    txt_path = os.path.join(out_dir, f"te2rules_cluster_{cluster_id}_vs_rest.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n{label}\nRULES FOR CLUSTER {cluster_id} vs REST\n{'='*60}\n\n")
        f.write(f"Number of rules found: {len(rules)}\n\n")
        for i, r in enumerate(rules, 1):
            f.write(f"Rule {i}:\n{r}\n\n")

    print(f"    -> {len(rules)} rules saved to {txt_path}")
    return rules


def save_summary(all_rules, out_dir, label):
    rows = []
    for cluster_id, rules in all_rules:
        for i, r in enumerate(rules, 1):
            rows.append({"subset": label, "cluster": cluster_id, "rule_index": i, "rule": str(r)})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "te2rules_summary.csv"), index=False)
        print(f"    -> Summary saved to {out_dir}/te2rules_summary.csv")
    else:
        print("    WARNING: No rules found")


def main():
    print("=" * 60)
    print("TE2Rules — All subsets (categorical only, fast RF)")
    print("=" * 60)

    for label, csv_path, out_dir in SUBSETS:
        print(f"\n{'─'*60}\nProcessing: {label}\nCSV: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"  WARNING: File not found — skipping")
            continue

        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(csv_path)

        if "Cluster" not in df.columns:
            print("  WARNING: 'Cluster' column not found — skipping")
            continue

        print(f"  Rows: {len(df)} | Clusters: {df['Cluster'].nunique()}")
        X_values = encode_features(df).values

        all_rules = []
        for cid in range(K):
            if cid not in df["Cluster"].values:
                continue
            print(f"  Running TE2Rules: Cluster {cid} vs rest...")
            y_binary = (df["Cluster"] == cid).astype(int)
            rules = run_te2rules_for_cluster(X_values, y_binary, cid, out_dir, label)
            all_rules.append((cid, rules))

        save_summary(all_rules, out_dir, label)

    print(f"\n{'='*60}\nDone! Results in: te2rules_outputs/\n{'='*60}")


if __name__ == "__main__":
    main()