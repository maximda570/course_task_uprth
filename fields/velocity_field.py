import numpy as np

class VelocityField:
    """Класс для представления поля скоростей"""
    
    def __init__(self, A_func, B_func):
        self.A = A_func
        self.B = B_func
    
    def at_point(self, t: float, point):
        """Вычисляет скорость в заданной точке и момент времени"""
        vx = -self.A(t) * point.x
        vy = self.B(t) * point.y
        return np.array([vx, vy])
    
    def stream_function(self, x, y):
        """Функция тока (для линий тока)"""
        return x * y  # т.к. dx/(-e^t x) = dy/(e^t y) => x*y = const
    
    def analytical_solution(self, x0, y0, t0, t):
        """Аналитическое решение для проверки"""
        exp_diff = np.exp(t) - np.exp(t0)
        x = x0 * np.exp(-exp_diff)
        y = y0 * np.exp(exp_diff)
        return x, y
