"""
Модуль поля скоростей.
Определяет поле скоростей v₁ = -A(t)x₁, v₂ = B(t)x₂.
"""

import numpy as np

class VelocityField:
    """
    Класс поля скоростей.
    
    Определяет векторное поле скоростей:
    v₁(x₁, x₂, t) = -A(t) x₁
    v₂(x₁, x₂, t) = B(t) x₂
    """
    
    def __init__(self, A_func, B_func):
        """
        Инициализация поля скоростей.
        
        Parameters:
        -----------
        A_func : callable
            Функция A(t)
        B_func : callable
            Функция B(t)
        """
        self.A = A_func
        self.B = B_func
    
    def at_point(self, t: float, point) -> np.ndarray:
        """
        Вычисляет скорость в заданной точке и момент времени.
        
        Parameters:
        -----------
        t : float
            Момент времени
        point : object with x, y attributes
            Точка, в которой вычисляется скорость
            
        Returns:
        --------
        np.ndarray
            Вектор скорости [vx, vy]
        """
        vx = -self.A(t) * point.x
        vy = self.B(t) * point.y
        return np.array([vx, vy])
    
    def stream_function(self, x: float, y: float) -> float:
        """
        Функция тока (stream function) для поля.
        
        Для данного поля линии тока определяются уравнением:
        dx/vx = dy/vy => dx/(-A(t)x) = dy/(B(t)y)
        После интегрирования: x*y = const (не зависит от t)
        
        Parameters:
        -----------
        x : float
            Координата x
        y : float
            Координата y
            
        Returns:
        --------
        float
            Значение функции тока
        """
        return x * y
    
    def analytical_solution(self, x0: float, y0: float, t0: float, t: float) -> tuple:
        """
        Аналитическое решение уравнений движения.
        
        Решение системы:
        dx/dt = -A(t)x, dy/dt = B(t)y
        
        Parameters:
        -----------
        x0, y0 : float
            Начальные координаты
        t0, t : float
            Начальное и конечное время
            
        Returns:
        --------
        tuple
            (x(t), y(t)) - координаты в момент t
        """
        # Интегралы от A(t) и B(t) от t0 до t
        integral_A = self._integrate_func(self.A, t0, t)
        integral_B = self._integrate_func(self.B, t0, t)
        
        x_t = x0 * np.exp(-integral_A)
        y_t = y0 * np.exp(integral_B)
        
        return x_t, y_t
    
    def _integrate_func(self, func, t0: float, t: float, n_points: int = 1000) -> float:
        """
        Численно интегрирует функцию от t0 до t.
        
        Parameters:
        -----------
        func : callable
            Функция для интегрирования
        t0, t : float
            Пределы интегрирования
        n_points : int, optional
            Количество точек для интегрирования
            
        Returns:
        --------
        float
            Значение интеграла
        """
        if t0 == t:
            return 0.0
        
        ts = np.linspace(t0, t, n_points)
        values = func(ts)
        integral = np.trapz(values, ts)
        
        return integral
    
    def get_velocity_meshgrid(self, t: float, x_range: tuple, y_range: tuple, 
                            resolution: int = 20) -> tuple:
        """
        Создает сетку скоростей для визуализации.
        
        Parameters:
        -----------
        t : float
            Момент времени
        x_range : tuple
            (x_min, x_max) - диапазон по x
        y_range : tuple
            (y_min, y_max) - диапазон по y
        resolution : int, optional
            Разрешение сетки
            
        Returns:
        --------
        tuple
            (X, Y, U, V) - сетки координат и скоростей
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        U = -self.A(t) * X
        V = self.B(t) * Y
        
        return X, Y, U, V
    
    def __repr__(self) -> str:
        """
        Строковое представление поля.
        
        Returns:
        --------
        str
            Оп
