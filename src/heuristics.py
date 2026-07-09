from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import state_to_features
from .maze import Position
from .models import import_tensorflow


class AnnHeuristic:
    """TensorFlow Lite model wrapper that can be passed into A* as h(n)."""

    def __init__(self, maze: np.ndarray, model_path: Path) -> None:
        tf = import_tensorflow()
        self.maze = maze
        
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.cache: dict[tuple[Position, Position], float] = {}

    def __call__(self, position: Position, goal: Position) -> float:
        key = (position, goal)
        if key not in self.cache:
            features = state_to_features(self.maze, position, goal).reshape(1, -1).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            self.interpreter.invoke()
            prediction = float(self.interpreter.get_tensor(self.output_details[0]['index'])[0][0])
            self.cache[key] = max(0.0, prediction)
        return self.cache[key]

