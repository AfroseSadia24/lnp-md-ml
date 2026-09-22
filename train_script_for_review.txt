from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CorrelationFilter, set_seed, write_json


def estimator_and_grid(name: str, seed: int, quick: bool):
    if name == "majority":
        return DummyClassifier(strategy="prior"), {}
    if name == "logistic":
        return LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=seed), {"model__C": [1.0] if quick else [0.01, 0.1, 1, 10]}
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=seed), {"model__n_estimators": [200] if quick else [300, 600], "model__max_depth": [None] if quick else [None, 6, 12], "model__min_samples_leaf": [1] if quick else [1, 3, 5]}
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=1, random_state=seed), {"model__n_estimators": [200] if quick else [200, 500], "model__max_depth": [3] if quick else [2, 3, 5], "model__learning_rate": [0.05] if quick else [0.02, 0.05, 0.1], "model__subsample": [0.8], "model__colsample_bytree": [0.8]}
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(class_weight="balanced", verbosity=-1, n_jobs=1, random_state=seed), {"model__n_estimators": [200] if quick else [200, 500], "model__num_leaves": [15] if quick else [7, 15, 31], "model__learning_rate": [0.05] if quick else [0.02, 0.05, 0.1]}
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(max_iter=1000, early_stopping=True, random_state=seed), {"model__hidden_layer_sizes": [(128, 64)] if quick else [(64,), (128, 64), (256, 128)], "model__alpha": [1e-4] if quick else [1e-5, 1e-4, 1e-3], "model__learning_rate_init": [1e-3]}
    raise ValueError(name)


def make_pipeline(df: pd.DataFrame, estimator):
    features = [c for c in df if c.startswith("feat_")]
    categorical = [c for c in features if df[c].dtype == "object"]
    numeric = [c for c in features if c not in categorical and not c.startswith("feat_fp_")]
    fingerprint = [c for c in features if c.startswith("feat_fp_")]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    fp_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0))])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unreported")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("numeric", num_pipe, numeric), ("fingerprint", fp_pipe, fingerprint), ("categorical", cat_pipe, categorical)], sparse_threshold=0)
    return Pipeline([("prep", prep), ("variance", VarianceThreshold()), ("correlation", CorrelationFilter(0.95)), ("model", estimator)])


def metrics(y, pred, prob):
    result = {"macro_f1": f1_score(y, pred, average="macro"), "mcc": matthews_corrcoef(y, pred), "pr_auc_minority_inactive": average_precision_score(1 - y, 1 - prob)}
    result["roc_auc"] = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/transfection_modelready_v1.csv")
    p.add_argument("--models", nargs="+", default=["majority", "logistic", "random_forest", "xgboost", "lightgbm"])
    p.add_argument("--schemes", nargs="+", default=["grouped", "random"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    df = pd.read_csv(args.input, low_memory=False)

    if "group_joint_component" not in df.columns:
        raise RuntimeError(
            "group_joint_component is missing from the model-ready data."
        )

    y = df.target_transfection_active.astype(int).to_numpy()

    for scheme in args.schemes:
        fold_path = Path("data/processed") / f"folds_{scheme}_v1.csv"
        fold_table = pd.read_csv(fold_path)

        if len(fold_table) != len(df):
            raise RuntimeError(
                f"Fold-table row count does not match data: {fold_path}"
            )

        if not np.array_equal(
            fold_table["id_lnp_id"].astype(str).to_numpy(),
            df["id_lnp_id"].astype(str).to_numpy(),
        ):
            raise RuntimeError(
                f"Fold-table row order does not match data: {fold_path}"
            )

        folds = fold_table.fold.to_numpy()
        for model_name in args.models:
            rows = []
            fold_metrics = []
            for fold in sorted(np.unique(folds)):
                tr, te = np.where(folds != fold)[0], np.where(folds == fold)[0]
                if scheme == "grouped":
                    assert set(
                        df.group_paper_doi.iloc[tr]
                    ).isdisjoint(
                        set(df.group_paper_doi.iloc[te])
                    )

                    assert set(
                        df.group_lipid_key.iloc[tr]
                    ).isdisjoint(
                        set(df.group_lipid_key.iloc[te])
                    )

                    assert set(
                        df.group_joint_component.iloc[tr]
                    ).isdisjoint(
                        set(df.group_joint_component.iloc[te])
                    )
                estimator, grid = estimator_and_grid(model_name, args.seed + int(fold), args.quick)
                pipe = make_pipeline(df.iloc[tr], estimator)
                if grid:
                    if scheme == "grouped":
                        inner_groups = (
                            df.group_joint_component.iloc[tr].to_numpy()
                        )

                        number_of_inner_groups = len(
                            np.unique(inner_groups)
                        )

                        if number_of_inner_groups < 2:
                            raise RuntimeError(
                                "Fewer than two joint components are "
                                "available for inner cross-validation."
                            )

                        inner = StratifiedGroupKFold(
                            n_splits=min(
                                3,
                                number_of_inner_groups,
                            ),
                            shuffle=True,
                            random_state=args.seed + int(fold),
                        )

                        search = GridSearchCV(
                            pipe,
                            grid,
                            scoring="f1_macro",
                            cv=inner,
                            n_jobs=4,
                            refit=True,
                        )

                        search.fit(
                            df.iloc[tr],
                            y[tr],
                            groups=inner_groups,
                        )
                    else:
                        inner = StratifiedKFold(3, shuffle=True, random_state=args.seed + int(fold))
                        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=4, refit=True)
                        search.fit(df.iloc[tr], y[tr])
                    fitted = search.best_estimator_
                    best_params = search.best_params_
                else:
                    fitted = pipe.fit(df.iloc[tr], y[tr])
                    best_params = {}
                pred = fitted.predict(df.iloc[te]).astype(int)
                prob = fitted.predict_proba(df.iloc[te])[:, 1]
                fm = metrics(y[te], pred, prob)
                fm.update({"fold": int(fold), "best_params": json.dumps(best_params, sort_keys=True)})
                fold_metrics.append(fm)
                for j, ix in enumerate(te):
                    rows.append({"id_lnp_id": df.id_lnp_id.iloc[ix], "group_paper_doi": df.group_paper_doi.iloc[ix], "group_lipid_key": df.group_lipid_key.iloc[ix], "y_true": y[ix], "y_pred": pred[j], "prob_active": prob[j], "fold": int(fold), "model": model_name, "scheme": scheme})
                Path("models").mkdir(exist_ok=True)
                joblib.dump(fitted, f"models/{model_name}_{scheme}_fold{fold}.joblib")
            pred_df = pd.DataFrame(rows).sort_values("id_lnp_id")
            overall = metrics(pred_df.y_true.to_numpy(), pred_df.y_pred.to_numpy(), pred_df.prob_active.to_numpy())
            overall.update({"model": model_name, "scheme": scheme, "n": len(pred_df), "fold_metrics": fold_metrics})
            Path("results/predictions").mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(f"results/predictions/{model_name}_{scheme}.csv", index=False)
            write_json(overall, f"results/metrics/{model_name}_{scheme}.json")
            print(model_name, scheme, overall)


if __name__ == "__main__":
    main()

