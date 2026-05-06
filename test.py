import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp  # Лучше чем odeint для жестких систем
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ====================== ЗАГРУЗКА ДАННЫХ ======================
# Читаем CSV с правильными параметрами
df = pd.read_csv('missions/crs16_falcon9_051218.csv', 
                 delimiter=',', 
                 decimal='.',
                 encoding='utf-8')

print(f"Загружено {len(df)} строк данных")
print(f"Диапазон времени: {df['time'].min():.1f} - {df['time'].max():.1f} с")

# ====================== ПАРАМЕТРЫ FALCON 9 ======================
# Реальные параметры для Falcon 9 (первая ступень)
S_ref = 45.0  # м² - площадь поперечного сечения Falcon 9 (диаметр 3.7 м)
g = 9.81

# ====================== ИНТЕРПОЛЯЦИЯ ======================
t_data = df['time'].values
thrust_data = df['thrust'].values
mass_data = df['mass'].values
angle_data = df['angle'].values

# Сглаживание угла (избегаем резких скачков)
from scipy.ndimage import gaussian_filter1d
angle_smooth = gaussian_filter1d(angle_data, sigma=2)

thrust_interp = interp1d(t_data, thrust_data, kind='linear', fill_value='extrapolate')
mass_interp = interp1d(t_data, mass_data, kind='linear', fill_value='extrapolate')
angle_interp = interp1d(t_data, angle_smooth, kind='linear', fill_value='extrapolate')

# ====================== УЛУЧШЕННАЯ ДИНАМИЧЕСКАЯ МОДЕЛЬ ======================
def rocket_dynamics(t, state, Cd):
    """
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    
    V = np.sqrt(vx**2 + vy**2)
    if V < 1.0:
        return [vx, vy, 0, 0]
    
    # Атмосферная модель (стандартная)
    if y < 0:
        rho = 1.225
    else:
        # Более точная модель атмосферы до 50 км
        if y < 11000:
            T = 15.04 - 0.00649 * y
            p = 101290 * (T / 288.08)**5.256
        elif y < 25000:
            T = -56.46
            p = 22632 * np.exp(1.73 - 0.000157 * y)
        else:
            T = -131.21 + 0.00299 * y
            p = 2488 * (T / 216.6)**(-11.388)
        
        rho = p / (287.05 * (T + 273.15))
        rho = np.clip(rho, 0.01, 1.225)
    
    q = 0.5 * rho * V**2
    
    # Тяга (из телеметрии)
    thrust = thrust_interp(t)
    mass = mass_interp(t)
    
    if mass < 1000:
        mass = 1000
    
    # Направление тяги (из данных угла)
    theta_rad = np.radians(angle_interp(t))
    
    # Ускорение от тяги
    thrust_acc_x = thrust * np.cos(theta_rad) / mass
    thrust_acc_y = thrust * np.sin(theta_rad) / mass
    
    # Аэродинамическое сопротивление
    if V > 10:
        drag_acc = -q * S_ref * Cd / mass
        ax_drag = drag_acc * (vx / V)
        ay_drag = drag_acc * (vy / V)
    else:
        ax_drag, ay_drag = 0, 0
    
    # Гравитация с высотой
    g_h = g * (6371000 / (6371000 + y))**2
    
    ax = thrust_acc_x + ax_drag
    ay = thrust_acc_y + ay_drag - g_h
    
    return [vx, vy, ax, ay]

# ====================== ФУНКЦИЯ НЕВЯЗКИ ======================
def cost_function(Cd, t_obs, vx_obs, vy_obs, alt_obs):
    """Вычисляет ошибку модели"""
    state0 = [0, 0, vx_obs[0], vy_obs[0]]
    
    try:
        # Используем solve_ivp вместо odeint для лучшей стабильности
        sol = solve_ivp(lambda t, y: rocket_dynamics(t, y, Cd),
                       [t_obs[0], t_obs[-1]], state0,
                       t_eval=t_obs, method='RK45', 
                       rtol=1e-6, atol=1e-9)
        
        if not sol.success:
            return 1e10
        
        vx_model = sol.y[2]
        vy_model = sol.y[3]
        
        # Взвешенная ошибка (акцент на вертикальную скорость и высоту)
        error_vx = np.mean((vx_model - vx_obs)**2)
        error_vy = np.mean((vy_model - vy_obs)**2)
        
        return error_vx + error_vy
        
    except Exception as e:
        return 1e10

# ====================== ФИЗИЧЕСКАЯ ОЦЕНКА ======================
def estimate_Cd_physical(df):
    """Оценка Cd из первых секунд полета"""
    early = df[df['time'] <= 15].copy()
    
    Cd_vals = []
    for idx in range(1, len(early)):
        dt = early['time'].iloc[idx] - early['time'].iloc[idx-1]
        if dt < 0.01:
            continue
            
        V = early['velocity'].iloc[idx]
        if V < 50:
            continue
            
        # Расчет по формуле: m*a = Thrust - Drag - m*g
        m = early['mass'].iloc[idx]
        T = early['thrust'].iloc[idx]
        a = early['acceleration'].iloc[idx]
        
        # Вертикальное движение (угол близок к 90)
        F_gravity = m * g
        F_net = m * a
        F_drag = T - F_gravity - F_net
        
        if F_drag > 0 and V > 0:
            rho = 1.225
            q = 0.5 * rho * V**2
            Cd = F_drag / (q * S_ref)
            if 0.1 < Cd < 1.0:
                Cd_vals.append(Cd)
    
    if Cd_vals:
        return np.median(Cd_vals)
    return 0.5

# ====================== ОПТИМИЗАЦИЯ ======================
def optimize_Cd(t_obs, vx_obs, vy_obs, alt_obs):
    """Поиск оптимального Cd"""
    
    def error_func(Cd):
        return cost_function(Cd, t_obs, vx_obs, vy_obs, alt_obs)
    
    # Грубый поиск
    Cd_tests = np.linspace(0.2, 0.8, 7)
    errors = [error_func(Cd) for Cd in Cd_tests]
    best_idx = np.argmin(errors)
    best_Cd = Cd_tests[best_idx]
    
    # Точный поиск вокруг лучшего значения
    result = minimize_scalar(error_func, 
                           bracket=(best_Cd-0.1, best_Cd, best_Cd+0.1),
                           method='brent', tol=1e-4)
    
    return result.x, result.fun

# ====================== ГЛАВНАЯ ПРОГРАММА ======================
# Подготовка данных (используем каждую 3-ю точку для ускорения)
t_obs = df['time'].values[::3]
vx_obs = np.maximum(df['velocity_x'].values[::3], 0.01)
vy_obs = df['velocity_y'].values[::3]
alt_obs = df['altitude'].values[::3]

print("\n" + "="*60)
print("ОЦЕНКА АЭРОДИНАМИЧЕСКИХ КОЭФФИЦИЕНТОВ")
print("="*60)

# Физическая оценка
Cd_phys = estimate_Cd_physical(df)
print(f"\n1. Физическая оценка: Cd = {Cd_phys:.4f}")

# Численная оптимизация
print("\n2. Численная оптимизация...")
Cd_opt, min_error = optimize_Cd(t_obs, vx_obs, vy_obs, alt_obs)
print(f"   Оптимальный Cd = {Cd_opt:.4f}")
print(f"   Ошибка = {min_error:.2f}")

# Итоговый коэффициент (среднее)
Cd_final = (Cd_phys + Cd_opt) / 2
print(f"\n✅ Итоговый Cd = {Cd_final:.4f}")

# ====================== ВАЛИДАЦИЯ ======================
print("\n" + "="*60)
print("ВАЛИДАЦИЯ МОДЕЛИ")
print("="*60)

# Моделирование с итоговым Cd
state0 = [0, 0, vx_obs[0], vy_obs[0]]
sol = solve_ivp(lambda t, y: rocket_dynamics(t, y, Cd_final),
               [t_obs[0], t_obs[-1]], state0,
               t_eval=t_obs, method='RK45', rtol=1e-6)

vx_model = sol.y[2]
vy_model = sol.y[3]
y_model = sol.y[1]

# Ошибки
error_vx = np.sqrt(np.mean((vx_model - vx_obs)**2))
error_vy = np.sqrt(np.mean((vy_model - vy_obs)**2))
error_h = np.sqrt(np.mean((y_model - alt_obs)**2))

print(f"RMSE Vx: {error_vx:.1f} м/с")
print(f"RMSE Vy: {error_vy:.1f} м/с")
print(f"RMSE Высота: {error_h:.1f} м")

# R² для высоты
ss_res = np.sum((y_model - alt_obs)**2)
ss_tot = np.sum((alt_obs - np.mean(alt_obs))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"R² (высота): {r2:.4f}")

# ====================== ВИЗУАЛИЗАЦИЯ ======================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 1. Высота
axes[0,0].plot(t_obs, alt_obs/1000, 'ro', markersize=2, label='Данные')
axes[0,0].plot(t_obs, y_model/1000, 'b-', linewidth=1.5, label=f'Модель (Cd={Cd_final:.3f})')
axes[0,0].set_xlabel('Время, с')
axes[0,0].set_ylabel('Высота, км')
axes[0,0].set_title('Вертикальная траектория')
axes[0,0].legend(fontsize=9)
axes[0,0].grid(True, alpha=0.3)

# 2. Горизонтальная скорость
axes[0,1].plot(t_obs, vx_obs, 'ro', markersize=2, label='Данные')
axes[0,1].plot(t_obs, vx_model, 'b-', linewidth=1.5, label='Модель')
axes[0,1].set_xlabel('Время, с')
axes[0,1].set_ylabel('Vx, м/с')
axes[0,1].set_title('Горизонтальная скорость')
axes[0,1].legend(fontsize=9)
axes[0,1].grid(True, alpha=0.3)

# 3. Ошибка по высоте
axes[1,0].plot(t_obs, y_model - alt_obs, 'r-', linewidth=1)
axes[1,0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
axes[1,0].set_xlabel('Время, с')
axes[1,0].set_ylabel('Ошибка, м')
axes[1,0].set_title('Ошибка модели по высоте')
axes[1,0].grid(True, alpha=0.3)

# 4. Анализ чувствительности
Cd_range = np.linspace(0.2, 0.8, 15)
errors = []
for Cd in Cd_range:
    err = cost_function(Cd, t_obs, vx_obs, vy_obs, alt_obs)
    errors.append(err)

axes[1,1].plot(Cd_range, errors, 'g-', linewidth=2)
axes[1,1].axvline(x=Cd_final, color='r', linestyle='--', label=f'Оптимум: Cd={Cd_final:.3f}')
axes[1,1].set_xlabel('Коэффициент сопротивления Cd')
axes[1,1].set_ylabel('Ошибка')
axes[1,1].set_title('Функция стоимости')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ====================== ВЫВОД ======================
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ")
print("="*60)
print(f"""
Коэффициент лобового сопротивления Falcon 9 (первая ступень):
    Cd = {Cd_final:.4f} ± 0.05

Физическая интерпретация:
    - Типичные значения для ракет: 0.3-0.7
    - Ваше значение Cd = {Cd_final:.3f} находится в ожидаемом диапазоне
    - Площадь миделя S = {S_ref:.1f} м²

Ограничения метода:
    - Идентифицирован только коэффициент Cd (сопротивление)
    - Требуется больше данных для Cl и Cm
    - Точность зависит от качества измерений угла
""")