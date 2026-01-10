import numpy as np
class VelocityField:    
    def __init__(self, A_func, B_func):
        # Сохраняем функции A(t) и B(t)
        self.A = A_func
        self.B = B_func
    
    def at_point(self, t: float, point) -> np.ndarray:

        # Вычисляем компоненту скорости по X: vx = -A(t) * x
        vx = -self.A(t) * point.x
        # Вычисляем компоненту скорости по Y: vy = B(t) * y
        vy = self.B(t) * point.y
        # Возвращаем вектор скорости
        return np.array([vx, vy])
    
    def stream_function(self, x: float, y: float) -> float:

        return x * y
    
    def analytical_solution(self, x0: float, y0: float, t0: float, t: float) -> tuple:

        # Интегралы от A(t) и B(t) от t0 до t
        integral_A = self._integrate_func(self.A, t0, t)
        integral_B = self._integrate_func(self.B, t0, t)
        
        # Аналитическое решение:
        # x(t) = x0 * exp(-∫A(t)dt)
        x_t = x0 * np.exp(-integral_A)
        # y(t) = y0 * exp(∫B(t)dt)
        y_t = y0 * np.exp(integral_B)
        
        return x_t, y_t
    
    def _integrate_func(self, func, t0: float, t: float, n_points: int = 1000) -> float:

        # Если t0 == t, интеграл равен 0
        if t0 == t:
            return 0.0
        
        # Создаем равномерную сетку по времени
        ts = np.linspace(t0, t, n_points)
        # Вычисляем значения функции в узлах сетки
        values = func(ts)
        # Интегрируем методом трапеций (np.trapz)
        integral = np.trapz(values, ts)
        
        return integral
    
    def get_velocity_meshgrid(self, t: float, x_range: tuple, y_range: tuple, 
                            resolution: int = 20) -> tuple:
        # Создаем равномерные сетки по X и Y
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        # Создаем матричные сетки (meshgrid)
        X, Y = np.meshgrid(x, y)
        
        # Вычисляем компоненты скорости в каждой точке сетки
        U = -self.A(t) * X  # vx компонента
        V = self.B(t) * Y   # vy компонента
        
        return X, Y, U, V
 

