import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize, root_scalar
from scipy.interpolate import interp1d

# ====================== ЗАГРУЗКА ПОЛНЫХ ДАННЫХ ======================
# Замените на ваш полный CSV файл
df = pd.read_csv('missions/crs16_falcon9_051218.csv', sep=',')  # или другой разделитель

# Альтернатива: если данные вставлены как текст
# df = pd.read_csv(pd.compat.StringIO(data_str), sep='\t', decimal=',')

print(f"Загружено {len(df)} строк данных")
print(f"Диапазон времени: {df['time'].min():.1f} - {df['time'].max():.1f} с")

# ====================== ИНТЕРПОЛЯЦИЯ ПАРАМЕТРОВ ======================
t_data = df['time'].values
thrust_data = df['thrust'].values
mass_data = df['mass'].values
angle_data = df['angle'].values  # Угол меняется!

thrust_interp = interp1d(t_data, thrust_data, kind='linear', fill_value='extrapolate')
mass_interp = interp1d(t_data, mass_data, kind='linear', fill_value='extrapolate')
angle_interp = interp1d(t_data, angle_data, kind='linear', fill_value='extrapolate')

# ====================== УТОЧНЕННАЯ ДИНАМИЧЕСКАЯ МОДЕЛЬ ======================
def rocket_dynamics(state, t, Cd):
    """
    state = [x, y, vx, vy]
    Без угла - используем наблюдаемый угол из телеметрии
    """
    x, y, vx, vy = state
    
    V = np.sqrt(vx**2 + vy**2)
    if V < 0.1:
        return [vx, vy, 0, 0]
    
    # Параметры ракеты (укажите реальные!)
    S_ref = 10.75  # м² - ХАРАКТЕРНАЯ ПЛОЩАДЬ (проверьте!)
    rho0 = 1.225
    H_scale = 8500.0  # м - правильный масштаб для атмосферы
    
    rho = rho0 * np.exp(-y / H_scale)
    q = 0.5 * rho * V**2
    
    # Аэродинамическое ускорение (только сопротивление)
    aero_acc = -q * S_ref * Cd / mass_interp(t)
    ax_aero = aero_acc * (vx / V)
    ay_aero = aero_acc * (vy / V)
    
    # Тяга (УГОЛ ИЗ ДАННЫХ!)
    thrust = thrust_interp(t)
    theta_rad = np.radians(angle_interp(t))
    thrust_acc_x = thrust * np.cos(theta_rad) / mass_interp(t)
    thrust_acc_y = thrust * np.sin(theta_rad) / mass_interp(t)
    
    # Гравитация
    g = 9.81
    
    ax = ax_aero + thrust_acc_x
    ay = ay_aero + thrust_acc_y - g
    
    return [vx, vy, ax, ay]

# ====================== ФУНКЦИЯ НЕВЯЗКИ (ТОЛЬКО СКОРОСТЬ) ======================
def cost_function(Cd, t_obs, vx_obs, vy_obs):
    """Вычисляет среднеквадратичную ошибку между моделью и данными"""
    state0 = [0, 0, vx_obs[0], vy_obs[0]]
    
    try:
        sol = odeint(rocket_dynamics, state0, t_obs, args=(Cd,))
        vx_model = sol[:, 2]
        vy_model = sol[:, 3]
        
        # Взвешенная ошибка (Vx важнее, т.к. Vy шумный)
        error = np.mean((vx_model - vx_obs)**2) + 0.5 * np.mean((vy_model - vy_obs)**2)
        return error
    except:
        return 1e10  # Штраф за ошибку интегрирования

# ====================== МЕТОД 1: ОПТИМИЗАЦИЯ ЗОЛОТОГО СЕЧЕНИЯ ======================
def binary_search_Cd(t_obs, vx_obs, vy_obs, Cd_min=0.01, Cd_max=2.0, tol=1e-4):
    """Бинарный поиск оптимального Cd (более стабильный чем Ньютон-Рафсон)"""
    
    def error_at_Cd(Cd):
        return cost_function(Cd, t_obs, vx_obs, vy_obs)
    
    # Используем золотое сечение для одномерной оптимизации
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(error_at_Cd, bounds=(Cd_min, Cd_max), method='bounded')
    
    return result.x, result.fun

# ====================== МЕТОД 2: ПРОСТАЯ ФИЗИЧЕСКАЯ ОЦЕНКА ======================
def estimate_Cd_from_ballistic(df):
    """
    Упрощенная оценка Cd из уравнения движения:
    m*a = Thrust - 0.5*rho*V^2*S*Cd - m*g
    """
    # Берем момент, когда угол близок к вертикали (первые 10 секунд)
    early_data = df[df['time'] <= 10].copy()
    
    # Расчетное сопротивление
    rho0 = 1.225
    S_ref = 10.75
    
    Cd_estimates = []
    
    for idx in range(1, len(early_data)):
        dt = early_data['time'].iloc[idx] - early_data['time'].iloc[idx-1]
        if dt < 0.01:
            continue
            
        # Средние значения на интервале
        V_mean = early_data['velocity'].iloc[idx]  # м/с
        thrust_mean = early_data['thrust'].iloc[idx]
        mass_mean = early_data['mass'].iloc[idx]
        acc_meas = early_data['acceleration'].iloc[idx]
        
        # Из уравнения: m*a = F_thrust - F_drag - m*g
        F_thrust = thrust_mean
        F_gravity = mass_mean * 9.81
        F_net = mass_mean * acc_meas
        
        F_drag = F_thrust - F_gravity - F_net
        
        if F_drag > 0 and V_mean > 10:
            q = 0.5 * rho0 * V_mean**2
            Cd = F_drag / (q * S_ref)
            if 0.1 < Cd < 1.5:  # Фильтр выбросов
                Cd_estimates.append(Cd)
    
    if Cd_estimates:
        return np.median(Cd_estimates)
    else:
        return 0.5

# ====================== ГЛАВНАЯ ПРОГРАММА ======================
# Подготовка данных (используем все точки, но сглаживаем)
t_obs = df['time'].values
vx_obs = df['velocity_x'].values
vy_obs = df['velocity_y'].values

# Заменяем нулевые Vx на очень маленькие значения (чтобы избежать деления на ноль)
vx_obs = np.maximum(vx_obs, 0.01)

print("\n" + "="*60)
print("ОЦЕНКА АЭРОДИНАМИЧЕСКИХ КОЭФФИЦИЕНТОВ")
print("="*60)

# Метод 1: Простая физическая оценка
Cd_phys = estimate_Cd_from_ballistic(df)
print(f"\n1. Физическая оценка (из уравнения движения): Cd ≈ {Cd_phys:.3f}")

# Метод 2: Оптимизация (только Cd)
print("\n2. Численная оптимизация (поиск Cd)...")
Cd_opt, min_error = binary_search_Cd(t_obs, vx_obs, vy_obs, Cd_min=0.1, Cd_max=1.2)

print(f"   Оптимальный Cd = {Cd_opt:.4f}")
print(f"   Минимальная ошибка = {min_error:.6f}")

# Выбираем лучшую оценку
if Cd_phys > 0 and Cd_phys < 2:
    Cd_final = (Cd_phys + Cd_opt) / 2  # Усредняем
    print(f"\n✅ Итоговый коэффициент сопротивления: Cd = {Cd_final:.4f}")
else:
    Cd_final = Cd_opt
    print(f"\n✅ Итоговый коэффициент сопротивления: Cd = {Cd_final:.4f} (по оптимизации)")

# ====================== ВАЛИДАЦИЯ МОДЕЛИ ======================
print("\n" + "="*60)
print("ВАЛИДАЦИЯ МОДЕЛИ")
print("="*60)

# Моделирование с найденным Cd
state0 = [0, 0, vx_obs[0], vy_obs[0]]
sol = odeint(rocket_dynamics, state0, t_obs, args=(Cd_final,))

vx_model = sol[:, 2]
vy_model = sol[:, 3]
x_model = sol[:, 0]
y_model = sol[:, 1]

# Ошибки
error_vx = np.sqrt(np.mean((vx_model - vx_obs)**2))
error_vy = np.sqrt(np.mean((vy_model - vy_obs)**2))
error_h = np.sqrt(np.mean((y_model - df['altitude'].values)**2))

print(f"Среднеквадратичная ошибка Vx: {error_vx:.2f} м/с")
print(f"Среднеквадратичная ошибка Vy: {error_vy:.2f} м/с")
print(f"Среднеквадратичная ошибка высоты: {error_h:.2f} м")

# Коэффициент детерминации
ss_res_vx = np.sum((vx_model - vx_obs)**2)
ss_tot_vx = np.sum((vx_obs - np.mean(vx_obs))**2)
r2_vx = 1 - (ss_res_vx / ss_tot_vx)

print(f"R² для Vx: {r2_vx:.4f}")

# ====================== ВИЗУАЛИЗАЦИЯ ======================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 1. Высота
axes[0,0].plot(t_obs, df['altitude'].values/1000, 'ro', markersize=3, label='Данные')
axes[0,0].plot(t_obs, y_model/1000, 'b-', linewidth=2, label=f'Модель (Cd={Cd_final:.3f})')
axes[0,0].set_xlabel('Время, с')
axes[0,0].set_ylabel('Высота, км')
axes[0,0].set_title('Вертикальная траектория')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Горизонтальная скорость
axes[0,1].plot(t_obs, vx_obs, 'ro', markersize=3, label='Данные')
axes[0,1].plot(t_obs, vx_model, 'b-', linewidth=2, label='Модель')
axes[0,1].set_xlabel('Время, с')
axes[0,1].set_ylabel('Vx, м/с')
axes[0,1].set_title('Горизонтальная скорость')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Вертикальная скорость
axes[0,2].plot(t_obs, vy_obs, 'ro', markersize=3, label='Данные')
axes[0,2].plot(t_obs, vy_model, 'b-', linewidth=2, label='Модель')
axes[0,2].set_xlabel('Время, с')
axes[0,2].set_ylabel('Vy, м/с')
axes[0,2].set_title('Вертикальная скорость')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 4. Траектория в координатах X-Y
axes[1,0].plot(x_model/1000, y_model/1000, 'b-', linewidth=2)
axes[1,0].set_xlabel('Дальность, км')
axes[1,0].set_ylabel('Высота, км')
axes[1,0].set_title('Траектория полета')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].axis('equal')

# 5. Угол тангажа из данных
axes[1,1].plot(t_obs, df['angle'].values, 'g-', linewidth=2)
axes[1,1].set_xlabel('Время, с')
axes[1,1].set_ylabel('Угол, градусы')
axes[1,1].set_title('Угол тангажа (из телеметрии)')
axes[1,1].grid(True, alpha=0.3)

# 6. Невязка по Vx (ошибка модели)
axes[1,2].plot(t_obs, vx_model - vx_obs, 'r-', linewidth=1)
axes[1,2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
axes[1,2].set_xlabel('Время, с')
axes[1,2].set_ylabel('ΔVx, м/с')
axes[1,2].set_title('Ошибка модели по Vx')
axes[1,2].grid(True, alpha=0.3)
axes[1,2].set_ylim([-20, 20])

plt.tight_layout()
plt.show()

# ====================== АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ======================
print("\n" + "="*60)
print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
print("="*60)

Cd_range = np.linspace(0.2, 1.2, 20)
errors = [cost_function(Cd, t_obs, vx_obs, vy_obs) for Cd in Cd_range]

plt.figure(figsize=(8, 5))
plt.plot(Cd_range, errors, 'b-', linewidth=2)
plt.axvline(x=Cd_final, color='r', linestyle='--', label=f'Оптимум: Cd={Cd_final:.3f}')
plt.xlabel('Коэффициент сопротивления Cd')
plt.ylabel('Ошибка модели')
plt.title('Функция стоимости: зависимость от Cd')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n💡 Рекомендуемое значение: Cd = {Cd_final:.4f}")
print("\n⚠️ ВНИМАНИЕ:")
print("   - Полученный коэффициент зависит от площади S_ref (сейчас 10.75 м²)")
print("   - Для идентификации Cl и Cm нужны данные об угле крена и моменте")
print("   - Рекомендуется использовать полный CSV файл вместо усеченных данных")