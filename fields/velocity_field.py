import numpy as np
from classes.spatial_point import SpatialPoint

class VelocityField:    
    def __init__(self, A_func, B_func):
        self.A = A_func  # A(t)
        self.B = B_func  # B(t)
    
    def at_point(self, t: float, point: SpatialPoint) -> np.ndarray:
        vx = -self.A(t) * point.x
        vy = self.B(t) * point.y
        return np.array([vx, vy])
    
    def analytical_solution(self, x0: float, y0: float, t0: float, t: float):
        integral = np.exp(t) - np.exp(t0)
        x_t = x0 * np.exp(-integral)
        y_t = y0 * np.exp(integral)
        return x_t, y_t
    
    def get_velocity_meshgrid(self, t: float, x_range: tuple, y_range: tuple, 
                            resolution: int = 20):
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        A_val = self.A(t)
        B_val = self.B(t)
        U = -A_val * X  # vx
        V = B_val * Y   # vy
        
        return X, Y, U, V

