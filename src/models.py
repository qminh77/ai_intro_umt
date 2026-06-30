from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    epochs: int
    batch_size: int
    train_limit: int | None
    early_stopping: bool
    description: str


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "underfit": ExperimentConfig(
        name="underfit",
        epochs=5,
        batch_size=64,
        train_limit=None,
        early_stopping=False,
        description="Small network and few epochs to demonstrate underfitting.",
    ),
    "overfit": ExperimentConfig(
        name="overfit",
        epochs=150,
        batch_size=32,
        train_limit=3000,
        early_stopping=False,
        description="Large network trained too long on limited data to demonstrate overfitting.",
    ),
    "goodfit": ExperimentConfig(
        name="goodfit",
        epochs=100,
        batch_size=64,
        train_limit=None,
        early_stopping=True,
        description="Moderate network with Dropout and EarlyStopping.",
    ),
}


def import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return tf


def build_model(experiment: str, input_dim: int):
    tf = import_tensorflow()

    if experiment == "underfit":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="ann_underfit_heuristic",
        )
    elif experiment == "overfit":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="ann_overfit_heuristic",
        )
    elif experiment == "goodfit":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="ann_goodfit_heuristic",
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def training_callbacks(experiment: str):
    tf = import_tensorflow()
    config = EXPERIMENTS[experiment]

    if not config.early_stopping:
        return []

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]

