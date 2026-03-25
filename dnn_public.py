"""Public DNN script for propensity-score estimation.

This version is designed for public sharing and reuse. All model architecture,
training, and data-path settings are exposed through command-line arguments so
users can adapt the workflow to their own simulation design.

Core workflow
-------------
1. Read shared simulation metadata from ``basic_info.txt``.
2. Load one or more simulation-condition folders.
3. Engineer tabular features from observed covariates.
4. Run K-fold cross-validation for the requested hyperparameter configuration.
5. Fit the final model on the full dataset and export predicted propensity scores.

The script intentionally avoids machine-specific file paths and private settings.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop


@dataclass
class BasicInfo:
    n_samples: int
    n_iterations: int
    n_covariates: int
    r_xx: List[float]
    r_xy: List[float]
    r_xt: List[float]
    r_ty: List[float]
    group_sizes: List[float]

    @property
    def folder_names(self) -> List[str]:
        folders: List[str] = []
        xx_names = [f"XX{int(v * 100):02d}_" for v in self.r_xx]
        ty_names = [f"TY{int(v * 100):02d}_" for v in self.r_ty]
        ts_names = [f"TS{int(v * 100):02d}" for v in self.group_sizes]
        for xx in xx_names:
            for ts in ts_names:
                for ty in ty_names:
                    folders.append(xx + ty + ts)
        return folders


def parse_basic_info(path: Path) -> BasicInfo:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mapping = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            mapping[key.strip()] = value.strip()

    return BasicInfo(
        n_samples=int(mapping["The number of samples"]),
        n_iterations=int(mapping["The number of iterations"]),
        n_covariates=int(mapping["The number of covariates"]),
        r_xx=[float(x) for x in mapping["r_XXs"].split()],
        r_xy=[float(x) for x in mapping["r_XYs"].split()],
        r_xt=[float(x) for x in mapping["r_XTs"].split()],
        r_ty=[float(x) for x in mapping["r_TYs"].split()],
        group_sizes=[float(x) for x in mapping["Group size"].split()],
    )


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def minmax_scale(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    if max_v == min_v:
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    return ((series - min_v) / (max_v - min_v)).astype(np.float32)


def engineer_features(
    df: pd.DataFrame,
    n_covariates: int,
    include_main: bool = True,
    include_squared: bool = True,
    include_interactions: bool = True,
) -> np.ndarray:
    """Create tabular features from raw covariates.

    Parameters
    ----------
    df:
        Input dataframe containing columns X1, X2, ..., Xp and treatment T.
    n_covariates:
        Number of observed covariates.
    include_main, include_squared, include_interactions:
        Feature-engineering switches exposed to end users.
    """
    covariate_names = [f"X{i}" for i in range(1, n_covariates + 1)]
    scaled = pd.DataFrame(index=df.index)
    base = pd.DataFrame({name: minmax_scale(df[name]) for name in covariate_names})

    if include_main:
        for name in covariate_names:
            scaled[name] = base[name]

    if include_squared:
        for name in covariate_names:
            scaled[f"{name}_sq"] = base[name] ** 2

    if include_interactions:
        for i, left in enumerate(covariate_names):
            for right in covariate_names[i + 1 :]:
                scaled[f"{left}_{right}"] = base[left] * base[right]

    if scaled.shape[1] == 0:
        raise ValueError("At least one feature family must be enabled.")

    return scaled.to_numpy(dtype=np.float32)


def parse_hidden_units(text: str) -> List[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--hidden-units must contain at least one integer.")
    return [int(v) for v in values]


def get_initializer(name: str):
    if name.lower() == "he_normal":
        return HeNormal()
    if name.lower() == "glorot_uniform":
        return GlorotUniform()
    raise ValueError(f"Unsupported initializer: {name}")


def get_optimizer(name: str, learning_rate: float):
    name = name.lower()
    if name == "sgd":
        return SGD(learning_rate=learning_rate)
    if name == "adam":
        return Adam(learning_rate=learning_rate)
    if name == "rmsprop":
        return RMSprop(learning_rate=learning_rate)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_dnn_model(
    input_dim: int,
    hidden_units: Sequence[int],
    activation: str,
    dropout_rate: float,
    initializer_name: str,
    optimizer_name: str,
    learning_rate: float,
) -> tf.keras.Model:
    """Construct a fully connected neural network classifier."""
    initializer = get_initializer(initializer_name)
    model = Sequential([Input(shape=(input_dim,))])

    for idx, units in enumerate(hidden_units):
        model.add(Dense(units, activation=activation, kernel_initializer=initializer, name=f"dense_{idx+1}"))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate, name=f"dropout_{idx+1}"))

    model.add(Dense(1, activation="sigmoid", name="ps_output"))
    model.compile(
        optimizer=get_optimizer(optimizer_name, learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.BinaryCrossentropy(name="cross_entropy")],
    )
    return model


def run_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    hidden_units: Sequence[int],
    activation: str,
    dropout_rate: float,
    initializer_name: str,
    optimizer_name: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    n_splits: int,
    patience: int,
    seed: int,
) -> List[dict]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results: List[dict] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x), start=1):
        model = build_dnn_model(
            input_dim=x.shape[1],
            hidden_units=hidden_units,
            activation=activation,
            dropout_rate=dropout_rate,
            initializer_name=initializer_name,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
        )
        callbacks = []
        if patience > 0:
            callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, verbose=0))
        history = model.fit(
            x[train_idx],
            y[train_idx],
            validation_data=(x[val_idx], y[val_idx]),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        metrics = model.evaluate(x[val_idx], y[val_idx], verbose=0)
        fold_results.append(
            {
                "fold": fold,
                "epochs_trained": len(history.history["loss"]),
                "train_accuracy_last": float(history.history["accuracy"][-1]),
                "val_accuracy_last": float(history.history["val_accuracy"][-1]),
                "train_loss_last": float(history.history["loss"][-1]),
                "val_loss_last": float(history.history["val_loss"][-1]),
                "eval_loss": float(metrics[0]),
                "eval_accuracy": float(metrics[1]),
                "eval_cross_entropy": float(metrics[2]),
            }
        )
        tf.keras.backend.clear_session()
        gc.collect()

    return fold_results


def fit_full_model_and_predict(
    x: np.ndarray,
    y: np.ndarray,
    hidden_units: Sequence[int],
    activation: str,
    dropout_rate: float,
    initializer_name: str,
    optimizer_name: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> np.ndarray:
    model = build_dnn_model(
        input_dim=x.shape[1],
        hidden_units=hidden_units,
        activation=activation,
        dropout_rate=dropout_rate,
        initializer_name=initializer_name,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
    )
    callbacks = []
    if patience > 0:
        callbacks.append(EarlyStopping(monitor="loss", mode="min", patience=patience, restore_best_weights=True, verbose=0))
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
    predictions = model.predict(x, verbose=0).reshape(-1)
    tf.keras.backend.clear_session()
    gc.collect()
    return predictions


def bundle_csv_files(folder: Path, bundle_size: int) -> List[pd.DataFrame]:
    csv_files = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".csv" and p.name.startswith("XX")])
    if not csv_files:
        raise FileNotFoundError(f"No simulation CSV files found in {folder}")
    if len(csv_files) % bundle_size != 0:
        raise ValueError(f"Number of files in {folder} is not divisible by bundle_size={bundle_size}.")

    bundles: List[pd.DataFrame] = []
    for start in range(0, len(csv_files), bundle_size):
        parts = [pd.read_csv(path) for path in csv_files[start : start + bundle_size]]
        bundles.append(pd.concat(parts, axis=0, ignore_index=True))
    return bundles


def save_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def build_run_name(args: argparse.Namespace, feature_dim: int) -> str:
    return (
        f"dnn_layers-{'-'.join(map(str, args.hidden_units))}"
        f"_features-{feature_dim}"
        f"_act-{args.activation}"
        f"_drop-{args.dropout}"
        f"_opt-{args.optimizer}"
        f"_lr-{args.learning_rate}"
        f"_batch-{args.batch_size}"
        f"_epochs-{args.epochs}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DNN propensity-score estimation script.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing basic_info.txt and simulation-data folders.")
    parser.add_argument("--results-root", type=Path, required=True, help="Directory where output files will be saved.")
    parser.add_argument("--bundle-size", type=int, default=50, help="Number of CSV files to concatenate per modeling bundle.")
    parser.add_argument("--use-noised-data", action="store_true", help="Read from datasets_noised instead of datasets.")
    parser.add_argument("--folder-index", type=int, default=None, help="Optional 0-based index for a single simulation condition.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for NumPy, TensorFlow, and cross-validation.")

    parser.add_argument("--include-main", action="store_true", default=True, help="Include main-effect covariates as model inputs.")
    parser.add_argument("--include-squared", action="store_true", default=True, help="Include squared covariate terms.")
    parser.add_argument("--include-interactions", action="store_true", default=True, help="Include pairwise interaction terms.")
    parser.add_argument("--hidden-units", type=parse_hidden_units, default=[189, 94], help="Comma-separated hidden-layer sizes, e.g., 189,94.")
    parser.add_argument("--activation", type=str, default="relu", help="Hidden-layer activation function.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate applied after each hidden layer when > 0.")
    parser.add_argument("--initializer", type=str, default="he_normal", choices=["he_normal", "glorot_uniform"], help="Weight initializer.")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "rmsprop"], help="Optimizer for training.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Optimizer learning rate.")
    parser.add_argument("--batch-size", type=int, default=100, help="Mini-batch size for model fitting.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping. Use 0 to disable.")

    args = parser.parse_args()
    set_global_seed(args.seed)

    basic = parse_basic_info(args.data_root / "basic_info.txt")
    source_dir = args.data_root / ("datasets_noised" if args.use_noised_data else "datasets")
    condition_indices = [args.folder_index] if args.folder_index is not None else list(range(len(basic.folder_names)))

    for condition_idx in condition_indices:
        folder_name = basic.folder_names[condition_idx]
        bundles = bundle_csv_files(source_dir / folder_name, args.bundle_size)

        # Use the first bundle to determine feature dimension for naming.
        preview_x = engineer_features(
            bundles[0],
            basic.n_covariates,
            include_main=args.include_main,
            include_squared=args.include_squared,
            include_interactions=args.include_interactions,
        )
        run_name = build_run_name(args, preview_x.shape[1])
        out_dir = args.results_root / "DNN" / folder_name / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "folder_name": folder_name,
            "seed": args.seed,
            "feature_options": {
                "include_main": args.include_main,
                "include_squared": args.include_squared,
                "include_interactions": args.include_interactions,
                "feature_dimension": int(preview_x.shape[1]),
            },
            "model_options": {
                "hidden_units": args.hidden_units,
                "activation": args.activation,
                "dropout": args.dropout,
                "initializer": args.initializer,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "cv_folds": args.cv_folds,
                "early_stopping_patience": args.early_stopping_patience,
            },
        }
        save_json(metadata, out_dir / "run_metadata.json")

        cv_summary = []
        for bundle_idx, df in enumerate(bundles):
            x = engineer_features(
                df,
                basic.n_covariates,
                include_main=args.include_main,
                include_squared=args.include_squared,
                include_interactions=args.include_interactions,
            )
            y = df["T"].to_numpy(dtype=np.float32)
            cv_results = run_cross_validation(
                x=x,
                y=y,
                hidden_units=args.hidden_units,
                activation=args.activation,
                dropout_rate=args.dropout,
                initializer_name=args.initializer,
                optimizer_name=args.optimizer,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                n_splits=args.cv_folds,
                patience=args.early_stopping_patience,
                seed=args.seed,
            )
            cv_summary.append({"bundle": bundle_idx, "results": cv_results})

            ps = fit_full_model_and_predict(
                x=x,
                y=y,
                hidden_units=args.hidden_units,
                activation=args.activation,
                dropout_rate=args.dropout,
                initializer_name=args.initializer,
                optimizer_name=args.optimizer,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.early_stopping_patience,
            )
            pd.DataFrame({"ps": ps}).to_csv(out_dir / f"bundle_{bundle_idx:03d}_predictions.csv", index=False)

        save_json(cv_summary, out_dir / "cross_validation_summary.json")


if __name__ == "__main__":
    main()
