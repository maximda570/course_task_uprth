from .spatial_point import SpatialPoint
from .trajectory import Trajectory
class MaterialPoint:    
    def __init__(self, x: float, y: float, mass: float = 1.0, point_id: int = 0):
        self.id = point_id
        self.mass = mass
        self.position = SpatialPoint(x, y)
        self.trajectory = Trajectory()
        # Записываем начальное положение
        self.trajectory.add(0.0, SpatialPoint(x, y)) 
    def set_position(self, spatial_point: SpatialPoint):
        self.position = spatial_point   
    def record_position(self, t: float):
        self.trajectory.add(t, self.position.copy())
    @property
    def x(self) -> float:
        return self.position.x
    @property
    def y(self) -> float:
        return self.position.y
    def get_trajectory_coords(self):
        return self.trajectory.get_coords_arrays()
    def get_position_at_time(self, t: float):
        return self.trajectory.get_position_at_time(t)
    
  
