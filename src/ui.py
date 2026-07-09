from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from .config import MODELS_DIR
from .dataset import generate_training_maze
from .demo import choose_demo_start
from .heuristics import AnnHeuristic
from .maze import Position, WALL
from .search import SearchResult, astar, bfs_path, manhattan


BFS_NAME = "BFS"
MANHATTAN_NAME = "A* + Manhattan"
ANN_NAME = "A* + ANN"
ALGORITHM_ORDER = (BFS_NAME, MANHATTAN_NAME, ANN_NAME)


@dataclass(frozen=True)
class Button:
    rect: pygame.Rect
    label: str
    action: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive maze pathfinding UI.")
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--wall-prob", type=float, default=0.25)
    parser.add_argument("--min-goal-distance", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "ann_heuristic_goodfit.tflite",
        help="Trained TFLite model for A* + ANN.",
    )
    return parser.parse_args()


class MazeUi:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = np.random.default_rng(args.seed)

        pygame.init()
        pygame.display.set_caption("Maze ANN A* Interactive Demo")
        self.screen = pygame.display.set_mode((1180, 760))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.small_font = pygame.font.SysFont("arial", 15)
        self.title_font = pygame.font.SysFont("arial", 28, bold=True)
        self.bold_font = pygame.font.SysFont("arial", 18, bold=True)

        self.buttons = self._create_buttons()
        self.model_path = args.model
        self.model_available = self.model_path.exists()
        self.ann_error: str | None = None

        self.map_index = 0
        self.maze: np.ndarray
        self.start: Position
        self.goal: Position
        self.results: dict[str, SearchResult] = {}
        self.selected_algorithm: str | None = None
        self.ann_heuristic: AnnHeuristic | None = None
        self.message = "Ready. Press R to randomize or A to run all."
        self.random_map()

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def random_map(self) -> None:
        self.maze, self.goal, distances = generate_training_maze(
            height=self.args.height,
            width=self.args.width,
            wall_probability=self.args.wall_prob,
            min_goal_distance=self.args.min_goal_distance,
            rng=self.rng,
        )
        self.start = choose_demo_start(
            distances,
            min_distance=self.args.min_goal_distance,
            rng=self.rng,
        )
        self.map_index += 1
        self.results = {}
        self.selected_algorithm = None
        self.ann_heuristic = None
        self.message = f"Map #{self.map_index}: random maze generated."

    def run_algorithm(self, algorithm: str) -> None:
        if algorithm == BFS_NAME:
            result = bfs_path(self.maze, self.start, self.goal)
        elif algorithm == MANHATTAN_NAME:
            result = astar(
                self.maze,
                self.start,
                self.goal,
                manhattan,
                algorithm_name=MANHATTAN_NAME,
            )
        elif algorithm == ANN_NAME:
            if not self.model_available:
                self.message = f"ANN model not found: {self.model_path}"
                return
            try:
                if self.ann_heuristic is None:
                    self.ann_heuristic = AnnHeuristic(self.maze, self.model_path)
                result = astar(
                    self.maze,
                    self.start,
                    self.goal,
                    self.ann_heuristic,
                    algorithm_name=ANN_NAME,
                )
            except RuntimeError as exc:
                self.ann_error = str(exc)
                self.message = "Cannot load ANN model. Check TensorFlow installation."
                return
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.results[result.algorithm] = result
        self.selected_algorithm = result.algorithm
        status = "found" if result.found else "not found"
        self.message = (
            f"{result.algorithm}: {status}, length={result.path_length}, "
            f"nodes={result.explored_nodes}, time={result.elapsed_ms:.2f} ms"
        )

    def run_all(self) -> None:
        self.run_algorithm(BFS_NAME)
        self.run_algorithm(MANHATTAN_NAME)
        if self.model_available:
            self.run_algorithm(ANN_NAME)
        self.selected_algorithm = next(
            (name for name in reversed(ALGORITHM_ORDER) if name in self.results),
            None,
        )

    def clear_results(self) -> None:
        self.results = {}
        self.selected_algorithm = None
        self.message = "Cleared results for current map."

    def draw(self) -> None:
        self.screen.fill((241, 245, 249))
        self._draw_header()
        self._draw_maze()
        self._draw_panel()

    def _create_buttons(self) -> list[Button]:
        x1 = 680
        x2 = 910
        y = 88
        width = 210
        height = 38
        gap = 12
        rows = [
            ((x1, "Random Map [R]", "random"), (x2, "Clear [C]", "clear")),
            ((x1, "Run All [A]", "run_all"), (x2, "Run BFS [1]", "run_bfs")),
            ((x1, "Run A* Man [2]", "run_manhattan"), (x2, "Run ANN [3]", "run_ann")),
            ((x1, "Show BFS", "show_bfs"), (x2, "Show Manhattan", "show_manhattan")),
            ((x1, "Show ANN", "show_ann"),),
        ]

        buttons: list[Button] = []
        for row in rows:
            for x, label, action in row:
                buttons.append(Button(pygame.Rect(x, y, width, height), label, action))
            y += height + gap
        return buttons

    def _handle_click(self, position: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.rect.collidepoint(position) and self._button_enabled(button.action):
                self._dispatch(button.action)
                return

    def _handle_key(self, key: int) -> None:
        if key == pygame.K_r:
            self.random_map()
        elif key == pygame.K_a:
            self.run_all()
        elif key == pygame.K_1:
            self.run_algorithm(BFS_NAME)
        elif key == pygame.K_2:
            self.run_algorithm(MANHATTAN_NAME)
        elif key == pygame.K_3:
            self.run_algorithm(ANN_NAME)
        elif key == pygame.K_c:
            self.clear_results()

    def _dispatch(self, action: str) -> None:
        if action == "random":
            self.random_map()
        elif action == "clear":
            self.clear_results()
        elif action == "run_all":
            self.run_all()
        elif action == "run_bfs":
            self.run_algorithm(BFS_NAME)
        elif action == "run_manhattan":
            self.run_algorithm(MANHATTAN_NAME)
        elif action == "run_ann":
            self.run_algorithm(ANN_NAME)
        elif action == "show_bfs":
            self._select_result(BFS_NAME)
        elif action == "show_manhattan":
            self._select_result(MANHATTAN_NAME)
        elif action == "show_ann":
            self._select_result(ANN_NAME)

    def _select_result(self, algorithm: str) -> None:
        if algorithm in self.results:
            self.selected_algorithm = algorithm
            self.message = f"Showing {algorithm}."

    def _button_enabled(self, action: str) -> bool:
        if action == "run_ann":
            return self.model_available and self.ann_error is None
        if action == "show_bfs":
            return BFS_NAME in self.results
        if action == "show_manhattan":
            return MANHATTAN_NAME in self.results
        if action == "show_ann":
            return ANN_NAME in self.results
        return True

    def _draw_header(self) -> None:
        title = self.title_font.render("Maze ANN A* Interactive Demo", True, (15, 23, 42))
        self.screen.blit(title, (24, 22))
        subtitle = self.small_font.render(
            "Random map, run BFS / A* Manhattan / A* ANN, then inspect explored nodes and path.",
            True,
            (71, 85, 105),
        )
        self.screen.blit(subtitle, (24, 52))

    def _draw_maze(self) -> None:
        rows, cols = self.maze.shape
        cell = min(30, 620 // max(rows, cols))
        left = 24
        top = 92
        grid_width = cols * cell
        grid_height = rows * cell

        pygame.draw.rect(
            self.screen,
            (203, 213, 225),
            pygame.Rect(left - 2, top - 2, grid_width + 4, grid_height + 4),
            border_radius=4,
        )

        for row in range(rows):
            for col in range(cols):
                color = (255, 255, 255) if self.maze[row, col] != WALL else (15, 23, 42)
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(left + col * cell, top + row * cell, cell, cell),
                )

        result = self._selected_result()
        if result is not None:
            self._draw_explored_nodes(result, left, top, cell)
            self._draw_path(result, left, top, cell)

        for row in range(rows + 1):
            y = top + row * cell
            pygame.draw.line(self.screen, (226, 232, 240), (left, y), (left + grid_width, y))
        for col in range(cols + 1):
            x = left + col * cell
            pygame.draw.line(self.screen, (226, 232, 240), (x, top), (x, top + grid_height))

        self._draw_marker(self.start, left, top, cell, (34, 197, 94), "S")
        self._draw_marker(self.goal, left, top, cell, (59, 130, 246), "G")

    def _draw_explored_nodes(
        self,
        result: SearchResult,
        left: int,
        top: int,
        cell: int,
    ) -> None:
        path_cells = set(result.path)
        for row, col in result.explored_order:
            if (row, col) in path_cells or (row, col) in (self.start, self.goal):
                continue
            rect = pygame.Rect(left + col * cell + 3, top + row * cell + 3, cell - 6, cell - 6)
            pygame.draw.rect(self.screen, (147, 197, 253), rect, border_radius=3)

    def _draw_path(
        self,
        result: SearchResult,
        left: int,
        top: int,
        cell: int,
    ) -> None:
        if not result.path:
            return

        centers = [
            (left + col * cell + cell // 2, top + row * cell + cell // 2)
            for row, col in result.path
        ]
        if len(centers) > 1:
            pygame.draw.lines(self.screen, (239, 68, 68), False, centers, max(3, cell // 5))
        for row, col in result.path:
            rect = pygame.Rect(left + col * cell + 6, top + row * cell + 6, cell - 12, cell - 12)
            pygame.draw.rect(self.screen, (248, 113, 113), rect, border_radius=4)

    def _draw_marker(
        self,
        position: Position,
        left: int,
        top: int,
        cell: int,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        row, col = position
        center = (left + col * cell + cell // 2, top + row * cell + cell // 2)
        pygame.draw.circle(self.screen, color, center, cell // 2 - 3)
        text = self.bold_font.render(label, True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=center))

    def _draw_panel(self) -> None:
        panel = pygame.Rect(650, 24, 506, 712)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, border_radius=12)
        pygame.draw.rect(self.screen, (203, 213, 225), panel, width=1, border_radius=12)

        self._draw_text("Controls", 680, 54, self.bold_font, (15, 23, 42))
        for button in self.buttons:
            self._draw_button(button)

        y = 360
        self._draw_text("Map", 680, y, self.bold_font, (15, 23, 42))
        y += 30
        self._draw_text(f"Map #{self.map_index} | size: {self.args.height}x{self.args.width}", 680, y)
        y += 24
        self._draw_text(f"Start: {self.start} | Goal: {self.goal}", 680, y)
        y += 24
        selected = self.selected_algorithm or "None"
        self._draw_text(f"Selected view: {selected}", 680, y)

        y += 42
        self._draw_results_table(y)
        self._draw_legend(606)

        message_rect = pygame.Rect(680, 675, 438, 42)
        pygame.draw.rect(self.screen, (241, 245, 249), message_rect, border_radius=8)
        self._draw_wrapped_text(self.message, 692, 684, 410, self.small_font, (51, 65, 85))

        if not self.model_available:
            self._draw_text("ANN model missing: train goodfit first.", 680, 642, self.small_font, (185, 28, 28))

    def _draw_results_table(self, y: int) -> None:
        self._draw_text("Results", 680, y, self.bold_font, (15, 23, 42))
        y += 30
        headers = ("Algorithm", "Len", "Nodes", "Time")
        xs = (680, 835, 890, 965)
        for x, header in zip(xs, headers, strict=True):
            self._draw_text(header, x, y, self.small_font, (71, 85, 105))
        y += 22

        for algorithm in ALGORITHM_ORDER:
            row_rect = pygame.Rect(672, y - 4, 455, 28)
            if algorithm == self.selected_algorithm:
                pygame.draw.rect(self.screen, (219, 234, 254), row_rect, border_radius=6)

            result = self.results.get(algorithm)
            if result is None:
                values = (algorithm, "--", "--", "--")
            else:
                length = "--" if result.path_length is None else str(result.path_length)
                values = (
                    algorithm,
                    length,
                    str(result.explored_nodes),
                    f"{result.elapsed_ms:.2f} ms",
                )
            for x, value in zip(xs, values, strict=True):
                self._draw_text(value, x, y, self.small_font, (15, 23, 42))
            y += 30

    def _draw_legend(self, y: int) -> None:
        self._draw_text("Legend", 680, y, self.bold_font, (15, 23, 42))
        entries = [
            ((15, 23, 42), "Wall"),
            ((147, 197, 253), "Explored node"),
            ((248, 113, 113), "Final path"),
            ((34, 197, 94), "Start"),
            ((59, 130, 246), "Goal"),
        ]
        y += 28
        for index, (color, label) in enumerate(entries):
            x = 680 if index % 2 == 0 else 900
            row_y = y + (index // 2) * 24
            pygame.draw.rect(self.screen, color, pygame.Rect(x, row_y + 3, 16, 16), border_radius=3)
            self._draw_text(label, x + 24, row_y, self.small_font, (51, 65, 85))

    def _draw_button(self, button: Button) -> None:
        enabled = self._button_enabled(button.action)
        bg = (37, 99, 235) if enabled else (148, 163, 184)
        fg = (255, 255, 255) if enabled else (226, 232, 240)
        pygame.draw.rect(self.screen, bg, button.rect, border_radius=8)
        text = self.small_font.render(button.label, True, fg)
        self.screen.blit(text, text.get_rect(center=button.rect.center))

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font | None = None,
        color: tuple[int, int, int] = (51, 65, 85),
    ) -> None:
        rendered = (font or self.font).render(text, True, color)
        self.screen.blit(rendered, (x, y))

    def _draw_wrapped_text(
        self,
        text: str,
        x: int,
        y: int,
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        words = text.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if font.size(candidate)[0] <= width:
                line = candidate
            else:
                self._draw_text(line, x, y, font, color)
                y += 18
                line = word
        if line:
            self._draw_text(line, x, y, font, color)

    def _selected_result(self) -> SearchResult | None:
        if self.selected_algorithm is None:
            return None
        return self.results.get(self.selected_algorithm)


def main() -> None:
    app = MazeUi(parse_args())
    app.run()


if __name__ == "__main__":
    main()
