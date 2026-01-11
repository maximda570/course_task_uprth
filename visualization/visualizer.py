import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Dict, Optional, Tuple, Any
class Visualizer:
    def __init__(self, figsize=(10, 6), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10)) 
    def plot_trajectories(self, body, x_range: Tuple[float, float], 
                         y_range: Tuple[float, float], **kwargs):
        title = kwargs.get('title', 'Траектории движения')
        show_labels = kwargs.get('show_labels', False)      
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)      
        # Рисуем траектории всех точек
        for i, point in enumerate(body.points):
            traj_data = point.get_trajectory_coords()
            if len(traj_data) >= 3:
                _, x_coords, y_coords = traj_data
                if len(x_coords) > 1:
                    color = self.colors[i % len(self.colors)]
                    label = f'Точка {point.id}' if show_labels else None
                    ax.plot(x_coords, y_coords, '-', color=color, 
                           alpha=0.6, linewidth=1, label=label)     
        # Начальная и конечная форма
        x_init, y_init = body.get_initial_positions_array()
        x_final, y_final = body.get_positions_array()    
        if len(x_init) > 1:
            ax.plot(x_init, y_init, 'k--', linewidth=3, 
                   label='Начальная форма', alpha=0.8)
            ax.plot(x_final, y_final, 'r-', linewidth=3, 
                   label='Конечная форма', alpha=0.8)       
            # Отмечаем первую и последнюю точки
            ax.scatter(x_init[0], y_init[0], c='green', s=100, 
                      marker='o', zorder=5, label='Начало отрезка')
            ax.scatter(x_init[-1], y_init[-1], c='blue', s=100, 
                      marker='s', zorder=5, label='Конец отрезка')
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.axhline(y=0, color='black', alpha=0.2, linestyle='-')
        ax.axvline(x=0, color='black', alpha=0.2, linestyle='-')
        plt.tight_layout()
        plt.show()
    def plot_streamlines_with_distribution(self, velocity_field, t: float,
                                         x_range: Tuple[float, float], 
                                         y_range: Tuple[float, float],
                                         **kwargs):
        body = kwargs.get('body', None)
        show_body = kwargs.get('show_body', False)
        resolution = kwargs.get('resolution', 20)
        title = kwargs.get('title', f'Поле линий тока и распределение скоростей (t={t:.2f})')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.dpi) 
        try:
            X, Y, U, V = velocity_field.get_velocity_meshgrid(
                t, x_range, y_range, resolution
            )
            # 1. ЛИНИИ ТОКА
            stream = ax1.streamplot(X, Y, U, V, color='blue', 
                                   linewidth=1.0, density=2.0,
                                   arrowsize=0.8) 
            # Добавляем тело если нужно
            if show_body and body:
                x_coords, y_coords = [], []
                for point in body.points:
                    traj_data = point.get_trajectory_coords()
                    if len(traj_data) >= 3:
                        traj_times, x_vals, y_vals = traj_data
                        idx = np.argmin(np.abs(np.array(traj_times) - t))
                        x_coords.append(x_vals[idx])
                        y_coords.append(y_vals[idx])
                if x_coords:
                    sorted_idx = np.argsort(x_coords)
                    ax1.plot(np.array(x_coords)[sorted_idx], 
                            np.array(y_coords)[sorted_idx],
                            'ro-', linewidth=2, markersize=6,
                            label=f'Тело (t={t:.2f})')
                    ax1.legend(loc='upper right')  
            ax1.set_xlabel('x1', fontsize=12)
            ax1.set_ylabel('x2', fontsize=12)
            ax1.set_title('ПОЛЕ ЛИНИЙ ТОКА', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_xlim(x_range)
            ax1.set_ylim(y_range)
            # 2. РАСПРЕДЕЛЕНИЕ СКОРОСТЕЙ
            speed = np.sqrt(U**2 + V**2)
            im = ax2.contourf(X, Y, speed, levels=30, 
                             cmap='viridis', alpha=0.9) 
            # Контуры скорости
            contours = ax2.contour(X, Y, speed, levels=15, 
                                  colors='white', linewidths=0.7, 
                                  alpha=0.6)
            ax2.clabel(contours, inline=True, fontsize=8)        
            # Цветовая шкала
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label('Модуль скорости', fontsize=11)
            # Максимальная скорость
            max_speed = np.max(speed)
            ax2.text(0.02, 0.98, f'Макс. скорость: {max_speed:.3f}',
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax2.set_xlabel('x1', fontsize=12)
            ax2.set_ylabel('x2', fontsize=12)
            ax2.set_title('РАСПРЕДЕЛЕНИЕ СКОРОСТЕЙ', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_xlim(x_range)
            ax2.set_ylim(y_range)   
        except Exception as e:
            print(f"Ошибка при построении поля: {e}")
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, 'Ошибка построения',
                       transform=ax.transAxes, ha='center', va='center')
        plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
def prepare_deformation_data(body, times: List[float]) -> List[Dict]: 
    deformation_data = []
    for t in times:
        x_coords, y_coords = [], []
        for point in body.points:
            traj_data = point.get_trajectory_coords()
            if len(traj_data) >= 3:
                traj_times, x_vals, y_vals = traj_data
                idx = np.argmin(np.abs(np.array(traj_times) - t))
                x_coords.append(x_vals[idx])
                y_coords.append(y_vals[idx])
        if x_coords:
            sorted_idx = np.argsort(x_coords)
            deformation_data.append({
                'time': t,
                'x': np.array(x_coords)[sorted_idx],
                'y': np.array(y_coords)[sorted_idx]
            })
    return deformation_data
