import numpy as np

class RKSolver:
    def __init__(self, butcher_table):
        self.table = butcher_table
    
    def integrate(self, f, y0, t0, t_end, dt, record_callback=None):
        y = np.array(y0, dtype=float)
        t = t0
        
        # Записываем начальное состояние
        if record_callback:
            record_callback(t, y.copy())
        stages = self.table.get_stages()
        A = self.table.A
        b = self.table.b
        c = self.table.c  
        # Основной цикл интегрирования
        while t < t_end - 1e-10:
            # Определяем реальный шаг
            h = min(dt, t_end - t)
            
            # Вычисляем коэффициенты k_i
            k = np.zeros((stages, len(y)))
            
            for i in range(stages):
                # Сумма A[i][j] * k[j] для j < i
                sum_ak = np.zeros_like(y)
                for j in range(i):
                    sum_ak += A[i][j] * k[j]
                
                # Аргумент для вычисления k[i]
                t_arg = t + c[i] * h
                y_arg = y + h * sum_ak
                
                k[i] = f(t_arg, y_arg)
            
            # Сумма b[i] * k[i]
            sum_bk = np.zeros_like(y)
            for i in range(stages):
                sum_bk += b[i] * k[i]
            
            # Новое значение
            y = y + h * sum_bk
            t = t + h
            
            # Записываем результат
            if record_callback:
                record_callback(t, y.copy())
        
        return y
    




