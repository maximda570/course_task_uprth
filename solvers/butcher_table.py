import numpy as np

class ButcherTable:
    
    def __init__(self, a, b, c):
     
        # Преобразуем список a в двумерный массив
        self.a = np.array(a, dtype=float)
        # Преобразуем список b в одномерный
        self.b = np.array(b, dtype=float)
        # Преобразуем список c в одномерный 
        self.c = np.array(c, dtype=float)
        self.stages = len(b)

