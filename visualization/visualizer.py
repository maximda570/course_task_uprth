import numpy as np
import matplotlib.pyplot as plt
class Visualizer:
    @staticmethod
    def plot_everything(body, times):
        # Создаем рисунок с 4 графиками
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        #Траектории 
        ax1 = axes[0, 0]
        for point in body.points:
            x_vals = [p.x for p in point.trajectory.positions]
            y_vals = [p.y for p in point.trajectory.positions]
            if len(x_vals) > 1:
                ax1.plot(x_vals, y_vals, 'b-', linewidth=1)
        ax1.set_title('Траектории точек')
        ax1.grid(True)
        
        #Начальная и конечная форма 
        ax2 = axes[0, 1]
        
        # Начальные координаты
        x_start = [p.trajectory.positions[0].x for p in body.points]
        y_start = [p.trajectory.positions[0].y for p in body.points]
        
        # Конечные координаты
        x_end = [p.position.x for p in body.points]
        y_end = [p.position.y for p in body.points]
        
        ax2.plot(x_start, y_start, 'go-', label='Начало (t=0)')
        ax2.plot(x_end, y_end, 'ro-', label=f'Конец (t={times[-1]})')
        
        ax2.set_title('Деформация тела')
        ax2.legend()
        ax2.grid(True)
        
        #Поле скоростей при t=0 
        ax3 = axes[1, 0]
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        X, Y = np.meshgrid(x, y)
        U0 = -np.exp(0) * X  # t=0
        V0 = np.exp(0) * Y
        
        ax3.quiver(X, Y, U0, V0, color='blue')
        ax3.set_title('Поле скоростей (t=0)')
        ax3.grid(True)
        
        #Поле скоростей при t=1 
        ax4 = axes[1, 1]
        t_last = times[-1] if len(times) > 0 else 1.0
        U1 = -np.exp(t_last) * X
        V1 = np.exp(t_last) * Y
        
        ax4.quiver(X, Y, U1, V1, color='red')
        ax4.set_title(f'Поле скоростей (t={t_last:.1f})')
        ax4.grid(True)
        
        fig.suptitle('Результаты моделирования', fontsize=14)
        # Настройка отступов
        plt.tight_layout(rect=[0, 0, 1, 0.96])        
        # Показываем
        plt.show()

