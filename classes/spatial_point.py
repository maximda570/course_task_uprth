import numpy as np
from typing import Optional
class SpatialPoint:
    def __init__(self, x: float, y: float):
        # Сохраняем координату X в атрибут self.x объекта
        self.x = x
        # Сохраняем координату Y в атрибут self.y объекта
        self.y = y  
    def copy(self) -> 'SpatialPoint':
        return SpatialPoint(self.x, self.y)
