import numpy as np
class ButcherTable:
    def __init__(self, A, b, c):

        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
    
    def get_stages(self):

        return len(self.b)
