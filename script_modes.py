import pandas as pd

# Διάβασε το CSV
df = pd.read_csv("outputs_split_kproto/asymptomatic/asymptomatic_cluster_assignments.csv")

# Categorical features
cat_features = ["stenosis", "gsm", "plarea", "dwa", "ctiastr", "jba"]

# Υπολόγισε το mode για κάθε cluster
modes = df.groupby("Cluster")[cat_features].agg(lambda x: x.mode()[0])

print(modes)
print()
modes.to_csv("outputs_split_kproto/asymptomatic/kproto_split_asymptomatic_cluster_modes.csv")
print("Saved: kproto_split_asymptomatic_cluster_modes.csv")