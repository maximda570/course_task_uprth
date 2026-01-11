import numpy as np
from typing import List, Optional
from .spatial_point import SpatialPoint
class Trajectory:
    def __init__(self):
        self.times: List[float] = []
        self.positions: List[SpatialPoint] = []
    def add(self, t: float, point: SpatialPoint):
        # Если уже есть точка с таким временем - заменяем её
        if self.times and abs(t - self.times[-1]) < 1e-10:
            self.positions[-1] = point
            return
        # Проверка упорядоченности времени
        if self.times and t < self.times[-1] - 1e-10:
            raise ValueError(f"Время t={t} должно быть больше последнего времени {self.times[-1]}")  
        self.times.append(t)
        self.positions.append(point)   
    def get_coords_arrays(self):
        if not self.times:
            return np.array([]), np.array([]), np.array([])    
        times_array = np.array(self.times)
        x_coords = np.array([p.x for p in self.positions])
        y_coords = np.array([p.y for p in self.positions]) 
        return times_array, x_coords, y_coords 
    def clear(self):
        """Очищает траекторию."""
        self.times.clear()
        self.positions.clear()
    def __len__(self) -> int:
        return len(self.times)
