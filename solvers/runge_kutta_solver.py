import numpy as np

from solvers.butcher_table import ButcherTable

class RKSolver:    
    def __init__(self, butcher_table: ButcherTable):
        self.bt = butcher_table
    
    def step(self, f, t: float, y: np.ndarray, h: float):
        # Получаем количество стадий метода
        s = self.bt.stages
        # Список для хранения значений k_i (наклонов)
        k = []
        
        # Цикл по всем стадиям метода
        for i in range(s):
            # Инициализируем сумму для аргумента стадии
            sum_a = np.zeros_like(y)            
            # Суммируем вклады от предыдущих стадий
            for j in range(i):
                # a[i][j] * k[j] - вклад j-й стадии в i-ю стадию
                sum_a += self.bt.a[i, j] * k[j]
   
            y_arg = y + h * sum_a
            # Вычисляем время для i-й стадии: t_arg = t + c_i * h
            t_arg = t + self.bt.c[i] * h
            
            # Вычисляем наклон k_i = f(t_arg, y_arg)
            ki = f(t_arg, y_arg)
            # Сохраняем наклон
            k.append(ki)
        
        # Вычисляем итоговое приращение: Σ b_i * k_i
        sum_b = np.zeros_like(y)
        for i in range(s):
            sum_b += self.bt.b[i] * k[i]
        
        # Новое значение: y_new = y + h * Σ b_i * k_i
        y_new = y + h * sum_b
        return y_new
    
    def integrate(self, f, y0: np.ndarray, t_start: float, 
                  t_end: float, dt: float, record_callback=None):

        # Начальные значения
        t = t_start
        y = y0.copy()  # Делаем копию, чтобы не изменять исходный массив
        
        # Списки для хранения истории (если нужно сохранить все точки)
        t_values = [t]
        y_values = [y.copy()]
        
        # Если задана функция обратного вызова, вызываем ее для начального состояния
        if record_callback:
            record_callback(t, y)
        
        # Основной цикл интегрирования
        while t < t_end:
            # Корректируем шаг, если выходим за t_end
            if t + dt > t_end:
                dt = t_end - t
            
            # Делаем один шаг метода Рунге-Кутты
            y = self.step(f, t, y, dt)
            # Увеличиваем время
            t += dt
            # Сохраняем текущее состояние в списки
            t_values.append(t)
            y_values.append(y.copy())




