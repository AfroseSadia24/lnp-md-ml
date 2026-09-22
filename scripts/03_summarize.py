from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef


def cluster_interval(pred: pd.DataFrame, metric: str, repeats: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    keys = pred["group_lipid_key"].unique()
    by_key = {k: pred[pred.group_lipid_key == k] for k in keys}
    values = []
    for _ in range(repeats):
        sample = pd.concat([by_key[k] for k in rng.choice(keys, len(keys), replace=True)], ignore_index=True)
        if metric == "macro_f1":
            values.append(f1_score(sample.y_true, sample.y_pred, average="macro"))
        elif metric == "mcc":
            values.append(matthews_corrcoef(sample.y_true, sample.y_pred))
        else:
            values.append(average_precision_score(1 - sample.y_true, 1 - sample.prob_active))
    return np.quantile(values, [0.025, 0.975]).tolist()


def main():
    rows = []
    for path in sorted(Path("results/metrics").glob("*.json")):
        d = json.loads(path.read_text())
        row = {k: d[k] for k in ["model", "scheme", "n", "macro_f1", "mcc", "pr_auc_minority_inactive", "roc_auc"]}
        pred = pd.read_csv(f"results/predictions/{d['model']}_{d['scheme']}.csv")
        for metric in ["macro_f1", "mcc", "pr_auc_minority_inactive"]:
            low, high = cluster_interval(pred, metric)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    if not rows:
        raise SystemExit("No metric JSON files found")
    df = pd.DataFrame(rows).sort_values(["model", "scheme"])
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    df.to_csv("results/main_results.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 7))
    sns.barplot(data=df, x="macro_f1", y="model", hue="scheme")
    plt.xlim(0, 1)
    plt.xlabel("Out-of-fold macro-F1")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("figures/split_scheme_gap.png", dpi=300)
    plt.close()

    long = df.melt(id_vars=["model", "scheme"], value_vars=["macro_f1", "mcc", "pr_auc_minority_inactive"], var_name="metric", value_name="value")
    g = sns.catplot(data=long, x="value", y="model", hue="scheme", col="metric", kind="bar", height=5, aspect=0.8, sharex=False)
    g.set_axis_labels("Out-of-fold score", "")
    g.savefig("figures/model_benchmark.png", dpi=300)
    plt.close("all")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
