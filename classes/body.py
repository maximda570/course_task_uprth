import numpy as np
from typing import List, Optional
from .material_point import MaterialPoint
class Body:
    def __init__(self, points: Optional[List[MaterialPoint]] = None, body_id: int = 0):
        # Сохраняем идентификатор тела
        self.id = body_id
        # Если points не предоставлен, создаем пустой список
        self.points = points if points is not None else []
    @classmethod
    def create_line_segment(cls, x_start: float, x_end: float, y: float, 
                           num_points: int, start_id: int = 0, mass: float = 1.0,
                           body_id: int = 0):
        if num_points < 2:
            raise ValueError("Количество точек должно быть не менее 2")
        # Создаем пустой список для точек
        points = []
        # Цикл по количеству точек
        for i in range(num_points):
            # Равномерно распределяем точки вдоль отрезка
            x = x_start + (x_end - x_start) * i / max(1, (num_points - 1))
            # Создаем материальную точку
            point = MaterialPoint(x, y, mass=mass, point_id=start_id + i)
            # Добавляем точку в список
            points.append(point)
        return cls(points, body_id=body_id)
    def __len__(self) -> int:
        return len(self.points)
    def __iter__(self):
        return iter(self.points)
    def get_positions_array(self):
        if not self.points:
            return np.array([]), np.array([])
        x_coords = np.array([point.x for point in self.points])
        y_coords = np.array([point.y for point in self.points])
        return x_coords, y_coords
    def get_initial_positions_array(self):
        if not self.points:
            return np.array([]), np.array([])
        x_coords = []
        y_coords = []
        for point in self.points:
            if point.trajectory.positions:
                initial_pos = point.trajectory.positions[0]
                x_coords.append(initial_pos.x)
                y_coords.append(initial_pos.y)
        return np.array(x_coords), np.array(y_coords)
    def add_point(self, point: MaterialPoint):
        self.points.append(point)
    def remove_point(self, point_id: int):
        self.points = [p for p in self.points if p.id != point_id]
