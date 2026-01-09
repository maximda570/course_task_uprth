import numpy as np

class ButcherTable:
    """Класс для хранения таблицы Бутчера метода Рунге-Кутты"""
    
    def __init__(self, a: list, b: list, c: list):
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.stages = len(b)
    
    def is_explicit(self):
        """Проверяет, является ли метод явным"""
        return np.all(np.triu(self.a, k=1) == 0)
    
    def __repr__(self):
        return f"ButcherTable(stages={self.stages}, explicit={self.is_explicit()})"
