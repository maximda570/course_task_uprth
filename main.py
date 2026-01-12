import numpy as np
import matplotlib.pyplot as plt
print("Визуализация движения тела")

def A_func(t):
    return np.exp(t)

def B_func(t):
    return np.exp(t)

BODY_START_X = 1.0
BODY_END_X = 2.0
BODY_Y = -0.1
NUM_POINTS = 10

TIME_START = 0.0
TIME_END = 1.0
DT = 0.01  # Шаг интегрирования

X_RANGE = (0, 3)
Y_RANGE = (-1, 1)

# Параметры метода Рунге-Кутты
BUTCHER_A = [[0, 0, 0, 0],
             [0.5, 0, 0, 0],
             [0, 0.5, 0, 0],
             [0, 0, 1, 0]]

BUTCHER_B = [1/6, 1/3, 1/3, 1/6]
BUTCHER_C = [0, 0.5, 0.5, 1]

from classes.body import Body
from classes.spatial_point import SpatialPoint
from fields.velocity_field import VelocityField
from solvers.butcher_table import ButcherTable
from solvers.runge_kutta_solver import RKSolver
from visualization.visualizer import Visualizer
    
def prepare_deformation_data(body, times):
    deformation_data = []
    for t in times:
        x_coords, y_coords = [], []
        for point in body.points:
            traj_data = point.get_trajectory_coords()
            if len(traj_data) >= 3:
                traj_times, x_vals, y_vals = traj_data
                # Находим ближайшую точку во времени
                idx = np.argmin(np.abs(np.array(traj_times) - t))
                x_coords.append(x_vals[idx])
                y_coords.append(y_vals[idx])
        
        if x_coords:
            # Сортируем по X для правильного соединения
            sorted_idx = np.argsort(x_coords)
            deformation_data.append({
                'time': t,
                'x': np.array(x_coords)[sorted_idx],
                'y': np.array(y_coords)[sorted_idx],
                'length': np.max(x_coords) - np.min(x_coords)
            })
    
    return deformation_data

def compare_solutions(body_numeric, body_analytic, velocity_field):
    errors_x = []
    errors_y = []
    
    for point_num, point_ana in zip(body_numeric.points, body_analytic.points):
        if point_num.trajectory.positions and point_ana.trajectory.positions:
            # Берем конечные положения
            num_pos = point_num.trajectory.positions[-1]
            ana_pos = point_ana.trajectory.positions[-1]
            
            error_x = abs(num_pos.x - ana_pos.x)
            error_y = abs(num_pos.y - ana_pos.y)
            
            errors_x.append(error_x)
            errors_y.append(error_y)
    
    if errors_x:
        print(f"Средняя ошибка по X: {np.mean(errors_x):.6f}")
        print(f"Максимальная ошибка по X: {np.max(errors_x):.6f}")
        print(f"Средняя ошибка по Y: {np.mean(errors_y):.6f}")
        print(f"Максимальная ошибка по Y: {np.max(errors_y):.6f}")

def main():
    # Создание тела и поля
    body = Body.create_line_segment(
        BODY_START_X,
        BODY_END_X,
        BODY_Y,
        NUM_POINTS
    )
    print(f"   Создано тело из {len(body)} точек")
    
    velocity_field = VelocityField(A_func, B_func)
    
    butcher_table = ButcherTable(BUTCHER_A, BUTCHER_B, BUTCHER_C)
    solver = RKSolver(butcher_table)
    print(f"Решатель Рунге-Кутты создан ({butcher_table.get_stages()} стадий)")
    # Функция скорости для интегрирования
    def velocity_func(t, y):
        return np.array([
            -A_func(t) * y[0], 
            B_func(t) * y[1]
        ])
    # Очищаем траектории
    for point in body.points:
        point.trajectory.times = []
        point.trajectory.positions = []
    for i, point in enumerate(body.points):
       
        
        initial_state = np.array([point.x, point.y])
        
        # Функция обратного вызова для записи результатов
        def create_callback(p):
            def callback(t, state):
                new_position = SpatialPoint(state[0], state[1])
                p.set_position(new_position)
                # Разрешаем запись с одинаковым временем (заменяем последнюю)
                if p.trajectory.times and abs(t - p.trajectory.times[-1]) < 1e-10:
                    p.trajectory.positions[-1] = new_position
                else:
                    p.trajectory.add(t, new_position)
            return callback
        

        solver.integrate(
            f=velocity_func,
            y0=initial_state,
            t0=TIME_START,
            t_end=TIME_END,
            dt=DT,
            record_callback=create_callback(point)
            )
    # Создание тела с аналитическим решением для сравнения 
    body_analytic = Body.create_line_segment(
        BODY_START_X,
        BODY_END_X,
        BODY_Y,
        NUM_POINTS
    )
    
    # Заполняем аналитическими решениями
    for point in body_analytic.points:
        point.trajectory.times = []
        point.trajectory.positions = []
        
        x0 = point.x
        y0 = point.y
        
        for t in np.linspace(TIME_START, TIME_END, 20):
            x_t, y_t = velocity_field.analytical_solution(x0, y0, TIME_START, t)
            point.trajectory.add(t, SpatialPoint(x_t, y_t))
            if t == TIME_END:
                point.set_position(SpatialPoint(x_t, y_t))

    visualizer = Visualizer()
    
    #Траектории движения
    visualizer.plot_trajectories(
        body=body,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        title=f"Траектории движения (RK4, шаг={DT})"
    )
    
    #Поле линий тока и распределения скоростей
    visualizer.plot_streamlines_with_distribution(
        velocity_field=velocity_field,
        t=TIME_END/2,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        body=body,
        show_body=True
    )
    
    # Создаем данные для сравнения
    comparison_times = [0.0, 0.5, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, t in enumerate(comparison_times):
        ax = axes[idx]
        
        # Численное решение
        x_num, y_num = [], []
        for point in body.points:
            traj_data = point.get_trajectory_coords()
            if len(traj_data) >= 3:
                traj_times, x_vals, y_vals = traj_data
                time_idx = np.argmin(np.abs(np.array(traj_times) - t))
                x_num.append(x_vals[time_idx])
                y_num.append(y_vals[time_idx])
        
        # Аналитическое решение
        x_ana, y_ana = [], []
        for point in body_analytic.points:
            traj_data = point.get_trajectory_coords()
            if len(traj_data) >= 3:
                traj_times, x_vals, y_vals = traj_data
                time_idx = np.argmin(np.abs(np.array(traj_times) - t))
                x_ana.append(x_vals[time_idx])
                y_ana.append(y_vals[time_idx])
        
        if x_num and x_ana:
            # Сортируем
            sorted_idx_num = np.argsort(x_num)
            sorted_idx_ana = np.argsort(x_ana)
            
            ax.plot(np.array(x_num)[sorted_idx_num], 
                   np.array(y_num)[sorted_idx_num],
                   'bo-', linewidth=2, markersize=6,
                   label='Численное (RK4)')
            
            ax.plot(np.array(x_ana)[sorted_idx_ana], 
                   np.array(y_ana)[sorted_idx_ana],
                   'r--', linewidth=2, markersize=6,
                   label='Аналитическое')
            
            # Вычисляем ошибку
            errors = []
            for xn, yn, xa, ya in zip(x_num, y_num, x_ana, y_ana):
                error = np.sqrt((xn - xa)**2 + (yn - ya)**2)
                errors.append(error)
            
            avg_error = np.mean(errors)
            ax.text(0.05, 0.95, f'Ср. ошибка: {avg_error:.2e}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'Сравнение при t={t:.2f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(X_RANGE)
        ax.set_ylim(Y_RANGE)
    
    plt.suptitle('СРАВНЕНИЕ ЧИСЛЕННОГО (RK4) И АНАЛИТИЧЕСКОГО РЕШЕНИЙ', 
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()
    
    print("Итоги:")
    
    # Сравнение решений
    compare_solutions(body, body_analytic, velocity_field)
    
    # Статистика деформации
    if body.points and body_analytic.points:
        # Начальная длина
        initial_x = [p.trajectory.positions[0].x for p in body.points 
                    if p.trajectory.positions]
        initial_length = max(initial_x) - min(initial_x) if initial_x else 0
        
        # Конечная длина (численная)
        final_x_num = [p.position.x for p in body.points]
        final_length_num = max(final_x_num) - min(final_x_num) if final_x_num else 0
        
        # Конечная длина (аналитическая)
        final_x_ana = [p.position.x for p in body_analytic.points]
        final_length_ana = max(final_x_ana) - min(final_x_ana) if final_x_ana else 0
        
        deformation_num = (final_length_num - initial_length) / initial_length
        deformation_ana = (final_length_ana - initial_length) / initial_length
        
        print("\nСТАТИСТИКА ДЕФОРМАЦИИ:")
        print(f"  Начальная длина: {initial_length:.6f}")
        print(f"  Конечная длина (числ.): {final_length_num:.6f}")
        print(f"  Конечная длина (аналит.): {final_length_ana:.6f}")
        print(f"  Деформация (числ.): {deformation_num:.6f} ({deformation_num:.2%})")
        print(f"  Деформация (аналит.): {deformation_ana:.6f} ({deformation_ana:.2%})")
        print(f"  Отличие в деформации: {abs(deformation_num - deformation_ana):.2e}")
    
  
main()
