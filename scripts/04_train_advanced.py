from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, allow_nan=True)


def calculate_metrics(y_true, y_pred, probability):
    result = {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "mcc": float(
            matthews_corrcoef(y_true, y_pred)
        ),
        "pr_auc_minority_inactive": float(
            average_precision_score(
                1 - y_true,
                1 - probability,
            )
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(y_true, probability)
        )
    else:
        result["roc_auc"] = float("nan")

    return result


def make_preprocessor(df):
    features = [
        column for column in df.columns
        if column.startswith("feat_")
    ]

    categorical = [
        column for column in features
        if df[column].dtype == "object"
    ]

    fingerprints = [
        column for column in features
        if column.startswith("feat_fp_")
    ]

    numeric = [
        column for column in features
        if column not in categorical
        and column not in fingerprints
    ]

    numeric_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    fingerprint_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0,
                    keep_empty_features=True,
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value="unreported",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            (
                "fingerprint",
                fingerprint_pipeline,
                fingerprints,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical,
            ),
        ],
        sparse_threshold=0,
    )


def fit_feature_pipeline(train_df, train_y, test_df, maximum=100):
    preprocessor = make_preprocessor(train_df)

    x_train = preprocessor.fit_transform(train_df)
    x_test = preprocessor.transform(test_df)

    variance = VarianceThreshold()
    x_train = variance.fit_transform(x_train)
    x_test = variance.transform(x_test)

    number_selected = min(maximum, x_train.shape[1])

    selector = SelectKBest(
        score_func=f_classif,
        k=number_selected,
    )

    x_train = selector.fit_transform(x_train, train_y)
    x_test = selector.transform(x_test)

    x_train = np.nan_to_num(
        x_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    x_test = np.nan_to_num(
        x_test,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    fitted = {
        "preprocessor": preprocessor,
        "variance": variance,
        "selector": selector,
    }

    return x_train, x_test, fitted


class FTTransformer(nn.Module):
    def __init__(
        self,
        number_of_features,
        token_dimension=32,
        number_of_heads=4,
        number_of_blocks=2,
        dropout=0.15,
    ):
        super().__init__()

        self.feature_weight = nn.Parameter(
            torch.empty(
                number_of_features,
                token_dimension,
            )
        )

        self.feature_bias = nn.Parameter(
            torch.empty(
                number_of_features,
                token_dimension,
            )
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, token_dimension)
        )

        nn.init.xavier_uniform_(self.feature_weight)
        nn.init.zeros_(self.feature_bias)

        layer = nn.TransformerEncoderLayer(
            d_model=token_dimension,
            nhead=number_of_heads,
            dim_feedforward=token_dimension * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=number_of_blocks,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(token_dimension),
            nn.Linear(token_dimension, 2),
        )

    def forward(self, x):
        tokens = (
            x.unsqueeze(-1) * self.feature_weight
            + self.feature_bias
        )

        cls = self.cls_token.expand(
            x.shape[0],
            -1,
            -1,
        )

        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.transformer(tokens)

        return self.head(encoded[:, 0])


def validation_indices(y, groups, scheme, seed):
    indices = np.arange(len(y))

    if scheme == "grouped":
        splitter = GroupShuffleSplit(
            n_splits=20,
            test_size=0.20,
            random_state=seed,
        )

        for train_index, validation_index in splitter.split(
            indices,
            y,
            groups,
        ):
            if (
                len(np.unique(y[train_index])) == 2
                and len(np.unique(y[validation_index])) == 2
            ):
                return train_index, validation_index

        raise RuntimeError(
            "Unable to construct a two-class grouped validation split."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    return next(splitter.split(indices, y))


def fit_ft_transformer(
    train_df,
    train_y,
    test_df,
    groups,
    scheme,
    seed,
    device,
):
    fitting_index, validation_index = validation_indices(
        train_y,
        groups,
        scheme,
        seed,
    )

    x_fit, x_validation, _ = fit_feature_pipeline(
        train_df.iloc[fitting_index],
        train_y[fitting_index],
        train_df.iloc[validation_index],
    )

    set_seed(seed)

    model = FTTransformer(
        number_of_features=x_fit.shape[1],
    ).to(device)

    class_counts = np.bincount(
        train_y[fitting_index],
        minlength=2,
    )

    class_weights = (
        len(fitting_index)
        / (2.0 * np.maximum(class_counts, 1))
    )

    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    training_dataset = TensorDataset(
        torch.tensor(x_fit),
        torch.tensor(
            train_y[fitting_index],
            dtype=torch.long,
        ),
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=64,
        shuffle=True,
    )

    validation_x = torch.tensor(
        x_validation,
        device=device,
    )

    best_score = -np.inf
    best_epoch = 1
    epochs_without_improvement = 0

    for epoch in range(1, 301):
        model.train()

        for batch_x, batch_y in training_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_probability = torch.softmax(
                model(validation_x),
                dim=1,
            )[:, 1].cpu().numpy()

        validation_prediction = (
            validation_probability >= 0.5
        ).astype(int)

        score = f1_score(
            train_y[validation_index],
            validation_prediction,
            average="macro",
        )

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 30:
            break

    x_train, x_test, feature_pipeline = fit_feature_pipeline(
        train_df,
        train_y,
        test_df,
    )

    set_seed(seed)

    final_model = FTTransformer(
        number_of_features=x_train.shape[1],
    ).to(device)

    class_counts = np.bincount(
        train_y,
        minlength=2,
    )

    class_weights = (
        len(train_y)
        / (2.0 * np.maximum(class_counts, 1))
    )

    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    final_dataset = TensorDataset(
        torch.tensor(x_train),
        torch.tensor(train_y, dtype=torch.long),
    )

    final_loader = DataLoader(
        final_dataset,
        batch_size=64,
        shuffle=True,
    )

    for _ in range(best_epoch):
        final_model.train()

        for batch_x, batch_y in final_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = final_model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()

    final_model.eval()

    with torch.no_grad():
        probability = torch.softmax(
            final_model(
                torch.tensor(
                    x_test,
                    device=device,
                )
            ),
            dim=1,
        )[:, 1].cpu().numpy()

    details = {
        "best_epoch": int(best_epoch),
        "validation_macro_f1": float(best_score),
        "number_of_selected_features": int(
            x_train.shape[1]
        ),
    }

    return (
        probability,
        final_model,
        feature_pipeline,
        details,
    )


def fit_tabpfn(
    train_df,
    train_y,
    test_df,
    device,
):
    from tabpfn import TabPFNClassifier

    x_train, x_test, feature_pipeline = fit_feature_pipeline(
        train_df,
        train_y,
        test_df,
    )

    model = TabPFNClassifier(device=device)
    model.fit(x_train, train_y)

    probability = model.predict_proba(x_test)[:, 1]

    details = {
        "number_of_selected_features": int(
            x_train.shape[1]
        )
    }

    return probability, model, feature_pipeline, details


def run_model(
    df,
    y,
    folds,
    model_name,
    scheme,
    seed,
    device,
):
    prediction_rows = []
    fold_metrics = []

    for fold in sorted(np.unique(folds)):
        print(
            f"Starting {model_name}, {scheme}, fold {fold}",
            flush=True,
        )

        training_index = np.where(folds != fold)[0]
        testing_index = np.where(folds == fold)[0]

        if scheme == "grouped":
            assert set(
                df.group_paper_doi.iloc[training_index]
            ).isdisjoint(
                set(
                    df.group_paper_doi.iloc[testing_index]
                )
            )

            assert set(
                df.group_lipid_key.iloc[training_index]
            ).isdisjoint(
                set(
                    df.group_lipid_key.iloc[testing_index]
                )
            )

            assert set(
                df.group_joint_component.iloc[training_index]
            ).isdisjoint(
                set(
                    df.group_joint_component.iloc[testing_index]
                )
            )

        train_df = df.iloc[training_index].reset_index(
            drop=True
        )

        test_df = df.iloc[testing_index].reset_index(
            drop=True
        )

        train_y = y[training_index]

        fold_seed = seed + int(fold)

        if model_name == "ft_transformer":
            groups = (
                train_df.group_joint_component.to_numpy()
                if scheme == "grouped"
                else np.arange(len(train_df))
            )

            (
                probability,
                model,
                feature_pipeline,
                details,
            ) = fit_ft_transformer(
                train_df,
                train_y,
                test_df,
                groups,
                scheme,
                fold_seed,
                device,
            )

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "details": details,
                },
                f"models/{model_name}_{scheme}_fold{fold}.pt",
            )

        elif model_name == "tabpfn":
            (
                probability,
                model,
                feature_pipeline,
                details,
            ) = fit_tabpfn(
                train_df,
                train_y,
                test_df,
                device,
            )

        else:
            raise ValueError(model_name)

        joblib.dump(
            feature_pipeline,
            f"models/{model_name}_{scheme}_fold{fold}_features.joblib",
        )

        prediction = (
            probability >= 0.5
        ).astype(int)

        fold_result = calculate_metrics(
            y[testing_index],
            prediction,
            probability,
        )

        fold_result.update(
            {
                "fold": int(fold),
                "best_params": json.dumps(
                    details,
                    sort_keys=True,
                ),
            }
        )

        fold_metrics.append(fold_result)

        for position, row_index in enumerate(testing_index):
            prediction_rows.append(
                {
                    "id_lnp_id": df.id_lnp_id.iloc[row_index],
                    "group_paper_doi": (
                        df.group_paper_doi.iloc[row_index]
                    ),
                    "group_lipid_key": (
                        df.group_lipid_key.iloc[row_index]
                    ),
                    "y_true": int(y[row_index]),
                    "y_pred": int(prediction[position]),
                    "prob_active": float(
                        probability[position]
                    ),
                    "fold": int(fold),
                    "model": model_name,
                    "scheme": scheme,
                }
            )

        print(
            f"Finished {model_name}, {scheme}, fold {fold}",
            fold_result,
            flush=True,
        )

        del model
        torch.cuda.empty_cache()

    prediction_table = pd.DataFrame(
        prediction_rows
    ).sort_values("id_lnp_id")

    overall = calculate_metrics(
        prediction_table.y_true.to_numpy(),
        prediction_table.y_pred.to_numpy(),
        prediction_table.prob_active.to_numpy(),
    )

    overall.update(
        {
            "model": model_name,
            "scheme": scheme,
            "n": int(len(prediction_table)),
            "fold_metrics": fold_metrics,
        }
    )

    prediction_table.to_csv(
        f"results/predictions/{model_name}_{scheme}.csv",
        index=False,
    )

    write_json(
        overall,
        f"results/metrics/{model_name}_{scheme}.json",
    )

    print(
        f"COMPLETE: {model_name} {scheme}",
        overall,
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/processed/"
            "transfection_modelready_v1.csv"
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["ft_transformer", "tabpfn"],
    )

    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["grouped", "random"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    device = "cuda"

    df = pd.read_csv(
        args.input,
        low_memory=False,
    )

    y = (
        df.target_transfection_active
        .astype(int)
        .to_numpy()
    )

    Path("models").mkdir(exist_ok=True)
    Path("results/metrics").mkdir(
        parents=True,
        exist_ok=True,
    )
    Path("results/predictions").mkdir(
        parents=True,
        exist_ok=True,
    )

    for scheme in args.schemes:
        fold_path = (
            Path("data/processed")
            / f"folds_{scheme}_v1.csv"
        )

        fold_table = pd.read_csv(fold_path)

        if len(fold_table) != len(df):
            raise RuntimeError(
                f"Fold row count mismatch: {fold_path}"
            )

        if not np.array_equal(
            fold_table.id_lnp_id.astype(str).to_numpy(),
            df.id_lnp_id.astype(str).to_numpy(),
        ):
            raise RuntimeError(
                f"Fold row order mismatch: {fold_path}"
            )

        folds = fold_table.fold.to_numpy()

        for model_name in args.models:
            run_model(
                df,
                y,
                folds,
                model_name,
                scheme,
                args.seed,
                device,
            )


if __name__ == "__main__":
    main()
