from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import state_to_features
from .maze import Position
from .models import import_tensorflow


class AnnHeuristic:
    """Keras model wrapper that can be passed into A* as h(n)."""

    def __init__(self, maze: np.ndarray, model_path: Path) -> None:
        tf = import_tensorflow()
        self.maze = maze
        self.model = tf.keras.models.load_model(model_path)
        self.cache: dict[tuple[Position, Position], float] = {}

    def __call__(self, position: Position, goal: Position) -> float:
        key = (position, goal)
        if key not in self.cache:
            features = state_to_features(self.maze, position, goal).reshape(1, -1)
            prediction = float(self.model.predict(features, verbose=0)[0][0])
            self.cache[key] = max(0.0, prediction)
        return self.cache[key]

