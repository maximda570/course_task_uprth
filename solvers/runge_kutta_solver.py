import numpy as np
from solvers.butcher_table import ButcherTable

class RKSolver:
    """Реализация метода Рунге-Кутты с произвольной таблицей Бутчера"""
    
    def __init__(self, butcher_table: ButcherTable):
        self.bt = butcher_table
    
    def step(self, f, t: float, y: np.ndarray, h: float):
        """
        Выполняет один шаг метода Рунге-Кутты
        
        Parameters:
        -----------
        f : callable
            Функция правой части: dy/dt = f(t, y)
        t : float
            Текущее время
        y : np.ndarray
            Текущее значение решения
        h : float
            Шаг по времени
        
        Returns:
        --------
        y_new : np.ndarray
            Новое значение решения
        """
        s = self.bt.stages
        k = []
        
        for i in range(s):
            # Вычисляем аргумент для i-й стадии
            sum_a = np.zeros_like(y)
            for j in range(i):
                sum_a += self.bt.a[i, j] * k[j]
            
            y_arg = y + h * sum_a
            t_arg = t + self.bt.c[i] * h
            
            ki = f(t_arg, y_arg)
            k.append(ki)
        
        # Вычисляем приращение
        sum_b = np.zeros_like(y)
        for i in range(s):
            sum_b += self.bt.b[i] * k[i]
        
        y_new = y + h * sum_b
        return y_new
    
    def integrate(self, f, y0: np.ndarray, t_start: float, 
                  t_end: float, dt: float, record_callback=None):
        """
        Интегрирует систему ОДУ на интервале
        
        Parameters:
        -----------
        f : callable
            Функция правой части
        y0 : np.ndarray
            Начальное условие
        t_start, t_end : float
            Начальное и конечное время
        dt : float
            Шаг интегрирования
        record_callback : callable, optional
            Функция обратного вызова для записи результатов
            
        Returns:
        --------
        t_values : list
            Значения времени
        y_values : list
            Значения решения
        """
        t = t_start
        y = y0.copy()
        
        t_values = [t]
        y_values = [y.copy()]
        
        if record_callback:
            record_callback(t, y)
        
        while t < t_end:
            if t + dt > t_end:
                dt = t_end - t
            
            y = self.step(f, t, y, dt)
            t += dt
            
            t_values.append(t)
            y_values.append(y.copy())
            
            if record_callback:
                record_callback(t, y)
        
        return t_values, y_values
