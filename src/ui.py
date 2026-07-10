"""Giao diện đồ họa tương tác Pygame cho demo tìm kiếm mê cung A* và mạng ANN.

Màn hình hiển thị cho phép người dùng sinh mê cung ngẫu nhiên mới, chạy từng hoặc tất cả các 
thuật toán tìm kiếm (BFS, A* Manhattan, A* ANN), xem thứ tự các nút được khám phá qua animation,
vẽ đường đi kết quả và hiển thị bảng so sánh hiệu năng.
"""

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

# Tên và cấu hình hiển thị cho các thuật toán
BFS_NAME = "BFS"
MANHATTAN_NAME = "A* + Manhattan"
ANN_NAME = "A* + ANN"
ALGORITHM_ORDER = (BFS_NAME, MANHATTAN_NAME, ANN_NAME)
ALL_VIEW = "All algorithms"

# Bảng màu đại diện cho các thuật toán (màu vẽ đường đi và nút mở rộng)
ALGORITHM_COLORS = {
    BFS_NAME: (37, 99, 235),       # Xanh dương
    MANHATTAN_NAME: (245, 158, 11),# Cam
    ANN_NAME: (124, 58, 237),      # Tím
}

# Màu sắc làm dịu/mờ để hiển thị các nút đã duyệt
ALGORITHM_SOFT_COLORS = {
    BFS_NAME: (191, 219, 254),
    MANHATTAN_NAME: (254, 215, 170),
    ANN_NAME: (221, 214, 254),
}

# Khoảng lệch tọa độ khi vẽ đè nhiều đường đi của các thuật toán trên cùng 1 ô (tránh đè trực tiếp)
PATH_OFFSETS = {
    BFS_NAME: (-5, -5),
    MANHATTAN_NAME: (0, 0),
    ANN_NAME: (5, 5),
}


@dataclass(frozen=True)
class Button:
    """Dataclass định nghĩa một đối tượng nút bấm trên giao diện Pygame.
    
    Attributes:
        rect: Khung chữ nhật bao quanh nút bấm (vị trí, kích thước).
        label: Nhãn chữ hiển thị trên nút.
        action: Chuỗi định danh hành động sẽ kích hoạt khi click nút.
    """
    rect: pygame.Rect
    label: str
    action: str


def parse_args() -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh đầu vào khi khởi chạy giao diện.
    
    Returns:
        argparse.Namespace chứa giá trị cấu hình khởi tạo giao diện.
    """
    parser = argparse.ArgumentParser(description="Giao diện tìm kiếm đường đi mê cung tương tác.")
    parser.add_argument("--height", type=int, default=20, help="Chiều cao mê cung mặc định.")
    parser.add_argument("--width", type=int, default=20, help="Chiều rộng mê cung mặc định.")
    parser.add_argument("--wall-prob", type=float, default=0.25, help="Xác suất xuất hiện tường.")
    parser.add_argument("--min-goal-distance", type=int, default=20, help="Khoảng cách tối thiểu Start-Goal.")
    parser.add_argument("--seed", type=int, default=None, help="Hạt giống số ngẫu nhiên.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "ann_heuristic_goodfit.tflite",
        help="Đường dẫn file mô hình .tflite.",
    )
    return parser.parse_args()


class MazeUi:
    """Lớp điều khiển toàn bộ trạng thái và vẽ giao diện đồ họa tương tác (GUI) Pygame."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Khởi tạo ứng dụng đồ họa Pygame, định hình cấu trúc cửa sổ, font chữ, nút bấm và sinh map ban đầu.
        
        Args:
            args: Namespace chứa các đối số cấu hình.
        """
        self.args = args
        self.rng = np.random.default_rng(args.seed)

        pygame.init()
        pygame.display.set_caption("Maze ANN A* Interactive Demo")
        self.screen = pygame.display.set_mode((1180, 760))
        self.clock = pygame.time.Clock()
        
        # Khởi tạo các font chữ kích thước khác nhau dùng trong hiển thị
        self.font = pygame.font.SysFont("arial", 18)
        self.small_font = pygame.font.SysFont("arial", 15)
        self.tiny_font = pygame.font.SysFont("arial", 10, bold=True)
        self.title_font = pygame.font.SysFont("arial", 28, bold=True)
        self.bold_font = pygame.font.SysFont("arial", 18, bold=True)

        # Định nghĩa các nút bấm điều hướng
        self.buttons = self._create_buttons()
        self.model_path = args.model
        self.model_available = self.model_path.exists()
        self.ann_error: str | None = None

        # Trạng thái điều khiển mô hình, tìm kiếm và animation
        self.map_index = 0
        self.maze: np.ndarray
        self.start: Position
        self.goal: Position
        self.results: dict[str, SearchResult] = {}
        self.selected_algorithm: str = ALL_VIEW
        self.animation_view: str | None = None
        self.animation_index = 0
        self.animation_cursor = 0.0
        self.animation_nodes_per_second = 35.0  # Tốc độ chạy animation (số ô mở rộng/giây)
        self.animating = False
        self.show_numbers = True
        self.ann_heuristic: AnnHeuristic | None = None
        self.message = "Sẵn sàng. Nhấn R để tạo map ngẫu nhiên hoặc A để chạy toàn bộ thuật toán."
        self.random_map()

    def run(self) -> None:
        """Vòng lặp chính (Game loop) của ứng dụng Pygame, xử lý sự kiện và vẽ giao diện liên tục."""
        running = True
        while running:
            dt_ms = self.clock.tick(60)  # Giới hạn 60 khung hình/giây
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            # Cập nhật tiến độ animation, vẽ lại màn hình
            self.update_animation(dt_ms)
            self.draw()
            pygame.display.flip()

        pygame.quit()

    def random_map(self) -> None:
        """Sinh ngẫu nhiên mê cung mới kèm theo điểm Start, Goal hợp lệ và đặt lại trạng thái."""
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
        self.selected_algorithm = ALL_VIEW
        self.ann_heuristic = None
        self._reset_animation()
        self.message = f"Bản đồ #{self.map_index}: Đã tạo mê cung ngẫu nhiên mới."

    def run_algorithm(self, algorithm: str, auto_animate: bool = True) -> None:
        """Kích hoạt và chạy một thuật toán tìm kiếm cụ thể trên mê cung hiện tại.
        
        Args:
            algorithm: Tên thuật toán cần chạy (BFS, A* + Manhattan, A* + ANN).
            auto_animate: Có tự động chạy hoạt cảnh duyệt nút sau khi giải xong hay không.
        """
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
                self.message = f"Không tìm thấy mô hình ANN: {self.model_path}"
                return
            try:
                # Lazy initialization cho AnnHeuristic để tối ưu thời gian khởi động
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
                self.message = "Không thể tải mô hình ANN. Vui lòng kiểm tra lại thư viện TensorFlow."
                return
        else:
            raise ValueError(f"Thuật toán không xác định: {algorithm}")

        # Lưu kết quả và thiết lập thông điệp kết quả
        self.results[result.algorithm] = result
        self.selected_algorithm = result.algorithm
        status = "Tìm thấy đường" if result.found else "Không tìm thấy đường"
        self.message = (
            f"{result.algorithm}: {status}, độ dài={result.path_length}, "
            f"node duyệt={result.explored_nodes}, thời gian={result.elapsed_ms:.2f} ms"
        )
        if auto_animate:
            self._start_animation(result.algorithm)

    def run_all(self) -> None:
        """Chạy tất cả các thuật toán tìm kiếm có sẵn và kích hoạt animation đồng thời."""
        self.run_algorithm(BFS_NAME, auto_animate=False)
        self.run_algorithm(MANHATTAN_NAME, auto_animate=False)
        if self.model_available:
            self.run_algorithm(ANN_NAME, auto_animate=False)
        self.selected_algorithm = ALL_VIEW
        self._start_animation(ALL_VIEW)
        self.message = "Đang chạy hoạt cảnh minh họa cho toàn bộ thuật toán."

    def clear_results(self) -> None:
        """Xóa toàn bộ kết quả tìm kiếm đã lưu trên bản đồ hiện tại."""
        self.results = {}
        self.selected_algorithm = ALL_VIEW
        self._reset_animation()
        self.message = "Đã xóa kết quả tìm kiếm của bản đồ hiện tại."

    def update_animation(self, dt_ms: int) -> None:
        """Cập nhật trạng thái chỉ số animation dựa trên thời gian thực tế đã trôi qua.
        
        Args:
            dt_ms: Số mili-giây trôi qua kể từ khung hình trước.
        """
        if not self.animating or self.animation_view is None:
            return

        limit = self._animation_limit(self.animation_view)
        if limit <= 0:
            self.animating = False
            return

        # Tính toán số lượng node được hiển thị tăng dần dựa trên tốc độ cấu hình
        self.animation_cursor += self.animation_nodes_per_second * (dt_ms / 1000)
        self.animation_index = min(limit, int(self.animation_cursor))
        if self.animation_index >= limit:
            self.animating = False
            self.message = f"Đã hoàn thành hoạt cảnh của: {self.animation_view}."

    def _start_animation(self, view: str) -> None:
        """Thiết lập và bắt đầu chạy hoạt cảnh cho một thuật toán hoặc tất cả thuật toán.
        
        Args:
            view: Tên thuật toán hiển thị hoạt cảnh.
        """
        limit = self._animation_limit(view)
        if limit <= 0:
            self._reset_animation()
            return

        self.animation_view = view
        self.animation_cursor = 0.0
        self.animation_index = 0
        self.animating = True

    def _reset_animation(self) -> None:
        """Đưa trạng thái hoạt cảnh về mặc định ban đầu (tắt hoạt cảnh)."""
        self.animation_view = None
        self.animation_index = 0
        self.animation_cursor = 0.0
        self.animating = False

    def draw(self) -> None:
        """Vẽ lại toàn bộ giao diện màn hình bao gồm: Header, Khung Mê cung và Bảng điều khiển."""
        self.screen.fill((241, 245, 249))  # Màu nền xám xanh nhạt
        self._draw_header()
        self._draw_maze()
        self._draw_panel()

    def _create_buttons(self) -> list[Button]:
        """Tạo lập danh sách các thực thể Button trên bảng điều khiển bên phải.
        
        Returns:
            Danh sách đối tượng Button.
        """
        x1 = 680
        x2 = 910
        y = 88
        width = 210
        height = 34
        gap = 8
        rows = [
            ((x1, "Tạo Map Mới [R]", "random"), (x2, "Xóa Kết Quả [C]", "clear")),
            ((x1, "Chạy Toàn Bộ [A]", "run_all"), (x2, "Chạy BFS [1]", "run_bfs")),
            ((x1, "Chạy A* Man [2]", "run_manhattan"), (x2, "Chạy ANN [3]", "run_ann")),
            ((x1, "Hiện Tất Cả [0]", "show_all"), (x2, "Hiện BFS", "show_bfs")),
            ((x1, "Hiện Manhattan", "show_manhattan"), (x2, "Hiện ANN", "show_ann")),
        ]

        buttons: list[Button] = []
        for row in rows:
            for x, label, action in row:
                buttons.append(Button(pygame.Rect(x, y, width, height), label, action))
            y += height + gap
        return buttons

    def _handle_click(self, position: tuple[int, int]) -> None:
        """Xử lý sự kiện khi người dùng click chuột trái vào màn hình.
        
        Args:
            position: Tọa độ click chuột (x, y).
        """
        for button in self.buttons:
            if button.rect.collidepoint(position) and self._button_enabled(button.action):
                self._dispatch(button.action)
                return

    def _handle_key(self, key: int) -> None:
        """Xử lý sự kiện nhấn phím tắt trên bàn phím.
        
        Args:
            key: Mã phím nhấn của Pygame.
        """
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
        elif key == pygame.K_0:
            self._select_result(ALL_VIEW)

    def _dispatch(self, action: str) -> None:
        """Điều hướng hành động của người dùng đến hàm xử lý nghiệp vụ tương ứng.
        
        Args:
            action: Chuỗi định danh hành động.
        """
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
        elif action == "show_all":
            self._select_result(ALL_VIEW)
        elif action == "show_bfs":
            self._select_result(BFS_NAME)
        elif action == "show_manhattan":
            self._select_result(MANHATTAN_NAME)
        elif action == "show_ann":
            self._select_result(ANN_NAME)

    def _select_result(self, algorithm: str) -> None:
        """Thay đổi góc nhìn (View) kết quả hiển thị trên bản đồ.
        
        Args:
            algorithm: Tên thuật toán muốn xem, hoặc ALL_VIEW.
        """
        if algorithm == ALL_VIEW:
            self.selected_algorithm = ALL_VIEW
            self._reset_animation()
            self.message = "Đang hiển thị toàn bộ đường đi tìm được."
        elif algorithm in self.results:
            self.selected_algorithm = algorithm
            self._reset_animation()
            self.message = f"Đang hiển thị kết quả của {algorithm}."

    def _button_enabled(self, action: str) -> bool:
        """Kiểm tra xem một nút bấm có khả năng nhấn được vào thời điểm hiện tại hay không.
        
        Args:
            action: Tên hành động của nút.
            
        Returns:
            True nếu có thể tương tác, ngược lại False.
        """
        if action == "run_ann":
            return self.model_available and self.ann_error is None
        if action == "show_all":
            return bool(self.results)
        if action == "show_bfs":
            return BFS_NAME in self.results
        if action == "show_manhattan":
            return MANHATTAN_NAME in self.results
        if action == "show_ann":
            return ANN_NAME in self.results
        return True

    def _draw_header(self) -> None:
        """Vẽ phần tiêu đề chính và hướng dẫn sử dụng ở góc trên bên trái màn hình."""
        title = self.title_font.render("Maze ANN A* Interactive Demo", True, (15, 23, 42))
        self.screen.blit(title, (24, 22))
        subtitle = self.small_font.render(
            "Tạo mê cung ngẫu nhiên, chạy BFS / A* Manhattan / A* ANN để so sánh số ô duyệt và đường đi.",
            True,
            (71, 85, 105),
        )
        self.screen.blit(subtitle, (24, 52))

    def _draw_maze(self) -> None:
        """Vẽ lưới mê cung lên màn hình kèm theo các điểm đánh dấu và đường đi tương ứng."""
        rows, cols = self.maze.shape
        cell = min(30, 620 // max(rows, cols))  # Tính toán kích thước ô dựa trên khung hình tối đa
        left = 24
        top = 92
        grid_width = cols * cell
        grid_height = rows * cell

        # Vẽ viền mê cung màu xám
        pygame.draw.rect(
            self.screen,
            (203, 213, 225),
            pygame.Rect(left - 2, top - 2, grid_width + 4, grid_height + 4),
            border_radius=4,
        )

        # Vẽ các ô trống và ô tường trong mê cung
        for row in range(rows):
            for col in range(cols):
                color = (248, 250, 252) if self.maze[row, col] != WALL else (30, 41, 59)
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(left + col * cell, top + row * cell, cell, cell),
                )

        # Vẽ dữ liệu đường đi và các nút được duyệt dựa trên thuật toán đang hiển thị
        if self.selected_algorithm == ALL_VIEW:
            self._draw_all_algorithm_view(left, top, cell)
        else:
            result = self.results.get(self.selected_algorithm)
            if result is not None:
                visible_count = self._visible_explored_count(self.selected_algorithm)
                self._draw_explored_nodes(
                    result,
                    self.selected_algorithm,
                    visible_count,
                    left,
                    top,
                    cell,
                    show_numbers=self.show_numbers,
                )
                if self._should_draw_path(self.selected_algorithm):
                    self._draw_path(result, self.selected_algorithm, left, top, cell)

        # Vẽ các đường lưới phân chia ô
        for row in range(rows + 1):
            y = top + row * cell
            pygame.draw.line(self.screen, (226, 232, 240), (left, y), (left + grid_width, y))
        for col in range(cols + 1):
            x = left + col * cell
            pygame.draw.line(self.screen, (226, 232, 240), (x, top), (x, top + grid_height))

        # Vẽ điểm đánh dấu Start (S) và Goal (G) đè lên trên cùng
        self._draw_marker(self.start, left, top, cell, (15, 23, 42), "S")
        self._draw_marker(self.goal, left, top, cell, (100, 116, 139), "G")

    def _draw_all_algorithm_view(self, left: int, top: int, cell: int) -> None:
        """Vẽ toàn bộ đường đi và hoạt cảnh duyệt nút của các thuật toán cùng một lúc.
        
        Args:
            left: Tọa độ X gốc của lưới mê cung.
            top: Tọa độ Y gốc của lưới mê cung.
            cell: Kích thước pixel mỗi ô.
        """
        animate_all = self.animation_view == ALL_VIEW and (
            self.animating or self.animation_index > 0
        )
        for algorithm in ALGORITHM_ORDER:
            result = self.results.get(algorithm)
            if result is None:
                continue
            if animate_all:
                visible_count = self._visible_explored_count(algorithm)
                self._draw_explored_nodes(
                    result,
                    algorithm,
                    visible_count,
                    left,
                    top,
                    cell,
                    show_numbers=False,
                )
            if self._should_draw_path(algorithm):
                self._draw_path(result, algorithm, left, top, cell, offset=True)

    def _draw_explored_nodes(
        self,
        result: SearchResult,
        algorithm: str,
        visible_count: int,
        left: int,
        top: int,
        cell: int,
        show_numbers: bool,
    ) -> None:
        """Vẽ các điểm đánh dấu (hình tròn) đại diện cho quá trình duyệt qua các node của thuật toán.
        
        Args:
            result: Kết quả tìm kiếm SearchResult.
            algorithm: Tên thuật toán hiện tại.
            visible_count: Số node đã duyệt hiển thị tại khung hình hiện tại.
            left: Điểm X gốc của mê cung.
            top: Điểm Y gốc của mê cung.
            cell: Độ rộng của ô.
            show_numbers: Có vẽ chữ số thứ tự duyệt node đè lên hay không.
        """
        explored = result.explored_order[:visible_count]
        soft_color = ALGORITHM_SOFT_COLORS[algorithm]
        strong_color = ALGORITHM_COLORS[algorithm]
        for index, (row, col) in enumerate(explored, start=1):
            if (row, col) in (self.start, self.goal):
                continue
            center = (left + col * cell + cell // 2, top + row * cell + cell // 2)
            radius = max(4, cell // 6)
            pygame.draw.circle(self.screen, soft_color, center, radius)
            
            # Vẽ viền tròn nổi bật cho node đang được duyệt ngoài cùng nếu đang chạy animation
            if index == visible_count and self.animating:
                pygame.draw.circle(self.screen, strong_color, center, radius + 3, width=2)

            # Vẽ số thứ tự duyệt node nếu hiển thị ở chế độ xem đơn lẻ
            if show_numbers and self.selected_algorithm != ALL_VIEW:
                label = self.tiny_font.render(str(index), True, (15, 23, 42))
                self.screen.blit(label, label.get_rect(center=center))

    def _draw_path(
        self,
        result: SearchResult,
        algorithm: str,
        left: int,
        top: int,
        cell: int,
        offset: bool = False,
    ) -> None:
        """Vẽ đường đi kết quả nối từ Start đến Goal bằng các đoạn thẳng liên tục.
        
        Args:
            result: Đối tượng kết quả tìm kiếm.
            algorithm: Tên thuật toán hiện tại.
            left: Điểm X gốc.
            top: Điểm Y gốc.
            cell: Kích thước pixel mỗi ô.
            offset: Có áp dụng độ lệch PATH_OFFSETS để tránh các đường đi đè lên nhau hay không.
        """
        if not result.path:
            return

        dx, dy = PATH_OFFSETS[algorithm] if offset else (0, 0)
        centers = [
            (left + col * cell + cell // 2 + dx, top + row * cell + cell // 2 + dy)
            for row, col in result.path
        ]
        if len(centers) > 1:
            pygame.draw.lines(
                self.screen,
                ALGORITHM_COLORS[algorithm],
                False,
                centers,
                max(4, cell // 6),
            )

    def _draw_marker(
        self,
        position: Position,
        left: int,
        top: int,
        cell: int,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        """Vẽ các điểm đánh dấu đặc biệt hình tròn chứa một ký tự chữ (Start / Goal).
        
        Args:
            position: Tọa độ ô cần vẽ.
            left: Điểm X gốc.
            top: Điểm Y gốc.
            cell: Kích thước ô.
            color: Màu nền của điểm tròn đánh dấu.
            label: Ký tự hiển thị (ví dụ "S" hoặc "G").
        """
        row, col = position
        center = (left + col * cell + cell // 2, top + row * cell + cell // 2)
        pygame.draw.circle(self.screen, color, center, cell // 2 - 3)
        text = self.bold_font.render(label, True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=center))

    def _draw_panel(self) -> None:
        """Vẽ khung bảng điều khiển bên phải, bao gồm các nút bấm, thông tin bản đồ, bảng kết quả và legend."""
        panel = pygame.Rect(650, 24, 506, 712)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, border_radius=12)
        pygame.draw.rect(self.screen, (203, 213, 225), panel, width=1, border_radius=12)

        self._draw_text("Điều khiển (Controls)", 680, 54, self.bold_font, (15, 23, 42))
        for button in self.buttons:
            self._draw_button(button)

        y = 326
        self._draw_text("Bản đồ (Map)", 680, y, self.bold_font, (15, 23, 42))
        y += 30
        self._draw_text(f"Mê cung #{self.map_index} | Kích thước: {self.args.height}x{self.args.width}", 680, y)
        y += 24
        self._draw_text(f"Start: {self.start} | Goal: {self.goal}", 680, y)
        y += 24
        self._draw_text(f"Chế độ hiển thị: {self.selected_algorithm}", 680, y)
        y += 24
        animation = "Đang chạy" if self.animating else "Dừng/Tắt"
        self._draw_text(f"Hoạt cảnh: {animation} | Tốc độ: Tự động", 680, y)

        y += 32
        self._draw_results_table(y)
        self._draw_legend(596)

        # Khung thông điệp thông báo trạng thái dưới cùng bảng điều khiển
        message_rect = pygame.Rect(680, 675, 438, 42)
        pygame.draw.rect(self.screen, (241, 245, 249), message_rect, border_radius=8)
        self._draw_wrapped_text(self.message, 692, 684, 410, self.small_font, (51, 65, 85))

        # Cảnh báo màu đỏ nổi bật nếu không tìm thấy file model ANN tốt nhất
        if not self.model_available:
            self._draw_text("Thiếu mô hình ANN: Hãy huấn luyện tốt nhất trước.", 680, 642, self.small_font, (185, 28, 28))

    def _draw_results_table(self, y: int) -> None:
        """Vẽ bảng so sánh các chỉ số độ dài đường đi, số node đã duyệt và thời gian chạy.
        
        Args:
            y: Tọa độ Y bắt đầu vẽ bảng.
        """
        self._draw_text("Bảng kết quả (Results)", 680, y, self.bold_font, (15, 23, 42))
        y += 30
        headers = ("Thuật toán", "Độ dài", "Node duyệt", "Thời gian")
        xs = (680, 835, 890, 965)
        for x, header in zip(xs, headers, strict=True):
            self._draw_text(header, x, y, self.small_font, (71, 85, 105))
        y += 22

        for algorithm in ALGORITHM_ORDER:
            row_rect = pygame.Rect(672, y - 4, 455, 28)
            # Làm nổi bật hàng của thuật toán đang được lựa chọn xem đơn lẻ
            if algorithm == self.selected_algorithm:
                pygame.draw.rect(self.screen, ALGORITHM_SOFT_COLORS[algorithm], row_rect, border_radius=6)

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
                color = ALGORITHM_COLORS[algorithm] if x == xs[0] else (15, 23, 42)
                self._draw_text(value, x, y, self.small_font, color)
            y += 30

    def _draw_legend(self, y: int) -> None:
        """Vẽ phần chú giải màu sắc các thành phần trên lưới mê cung.
        
        Args:
            y: Tọa độ Y bắt đầu vẽ chú giải.
        """
        self._draw_text("Chú giải (Legend)", 680, y, self.bold_font, (15, 23, 42))
        entries = [
            (ALGORITHM_COLORS[BFS_NAME], "Giải thuật BFS"),
            (ALGORITHM_COLORS[MANHATTAN_NAME], "A* Manhattan"),
            (ALGORITHM_COLORS[ANN_NAME], "A* ANN"),
            ((30, 41, 59), "Ô Tường (Wall)"),
        ]
        y += 28
        for index, (color, label) in enumerate(entries):
            x = 680 if index % 2 == 0 else 900
            row_y = y + (index // 2) * 24
            pygame.draw.rect(self.screen, color, pygame.Rect(x, row_y + 3, 16, 16), border_radius=3)
            self._draw_text(label, x + 24, row_y, self.small_font, (51, 65, 85))

    def _draw_button(self, button: Button) -> None:
        """Vẽ giao diện hiển thị một nút bấm và nhãn chữ của nó.
        
        Args:
            button: Đối tượng Button cần vẽ.
        """
        enabled = self._button_enabled(button.action)
        # Sử dụng tông màu xanh dương khi khả dụng, màu xám khi bị vô hiệu hóa
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
        """Hàm tiện ích vẽ chữ nhanh tại tọa độ tùy ý.
        
        Args:
            text: Nội dung chữ.
            x: Tọa độ X.
            y: Tọa độ Y.
            font: Font chữ tùy chọn (nếu None sử dụng self.font mặc định).
            color: Màu chữ RGB.
        """
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
        """Hàm tiện ích hỗ trợ tự động xuống dòng khi chiều dài dòng văn bản vượt quá giới hạn.
        
        Args:
            text: Nội dung văn bản cần hiển thị.
            x: Tọa độ X.
            y: Tọa độ Y.
            width: Độ rộng tối đa bằng pixel trước khi xuống dòng.
            font: Font chữ sử dụng.
            color: Màu chữ RGB.
        """
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

    def _visible_explored_count(self, algorithm: str) -> int:
        """Đo lường số lượng node duyệt được hiển thị trên hoạt cảnh tại thời điểm hiện tại.
        
        Args:
            algorithm: Tên giải thuật.
            
        Returns:
            Số lượng node được vẽ.
        """
        result = self.results.get(algorithm)
        if result is None:
            return 0
        # Nếu đang chạy hoạt cảnh cho chính thuật toán này hoặc toàn bộ thuật toán
        if self.animation_view in (algorithm, ALL_VIEW):
            return min(self.animation_index, len(result.explored_order))
        return len(result.explored_order)

    def _animation_limit(self, view: str) -> int:
        """Xác định số node tối đa là giới hạn kết thúc cho tiến trình hoạt cảnh.
        
        Args:
            view: Góc nhìn thuật toán chạy hoạt cảnh.
            
        Returns:
            Số node giới hạn.
        """
        if view == ALL_VIEW:
            return max(
                (len(result.explored_order) for result in self.results.values()),
                default=0,
            )
        result = self.results.get(view)
        if result is None:
            return 0
        return len(result.explored_order)

    def _should_draw_path(self, algorithm: str) -> bool:
        """Quyết định xem có vẽ đường đi kết quả của một thuật toán tại khung hình hiện tại hay không.
        
        Đường đi kết quả chỉ được vẽ sau khi hoạt cảnh mở rộng các node đã hoàn tất.
        
        Args:
            algorithm: Tên giải thuật.
            
        Returns:
            True nếu đồng ý vẽ đường đi, ngược lại False.
        """
        result = self.results.get(algorithm)
        if result is None:
            return False
        if self.animation_view in (algorithm, ALL_VIEW):
            return self.animation_index >= len(result.explored_order)
        return True


def main() -> None:
    """Hàm khởi tạo đối tượng ứng dụng MazeUi và thực thi tiến trình UI."""
    app = MazeUi(parse_args())
    app.run()


if __name__ == "__main__":
    main()

