import constants
import math
import numpy as np
import matplotlib.pyplot as plt

flight_file = "falcon9_020517.csv"
v_time     = constants.read_array_from_csv(flight_file, "Time")
v_speed    = constants.read_array_from_csv(flight_file, "Speed")
v_altitude = constants.read_array_from_csv(flight_file, "Altitude")

v_time     = v_time     [620:]
v_speed    = v_speed    [620:]
v_altitude = v_altitude [620:]

keep_mask = [True]

for i in range(1, len(v_speed)):
    speed_same = (v_speed[i] == v_speed[i-1])
    alt_same   = (v_altitude[i] == v_altitude[i-1])
    if speed_same or alt_same:
        keep_mask.append(False)
    else:
        keep_mask.append(True)

v_time     = [v_time[i] for i in range(len(v_time)) if keep_mask[i]]
v_speed    = [v_speed[i] for i in range(len(v_speed)) if keep_mask[i]]
v_altitude = [v_altitude[i] for i in range(len(v_altitude)) if keep_mask[i]]


# Конвертируем altitude в метры
v_altitude = [a*1000 for a in v_altitude]

v_time = np.array(v_time)
v_speed = np.array(v_speed)
v_altitude = np.array(v_altitude)


# Вычисляем вертикальную скорость
v_vertical = np.gradient(v_altitude, v_time)

# Проверяем, что скорость не нулевая (избегаем деления на ноль)
valid_mask = v_speed > 0

# Вычисляем sin угла (это уже полезная величина)
sin_gamma = np.zeros_like(v_speed)
sin_gamma[valid_mask] = v_vertical[valid_mask] / v_speed[valid_mask]

# Ограничиваем диапазон (численно может выходить за [-1, 1])
sin_gamma = np.clip(sin_gamma, -1, 1)

# Если нужно визуализировать угол (тогда arctan неизбежен)
gamma_rad = np.arcsin(sin_gamma)
gamma_deg = np.degrees(gamma_rad)

# Построим графики
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(v_time, sin_gamma, label='sin(γ)')
axes[0].set_ylabel('sin(γ)')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(v_time, gamma_deg, label='γ (degrees)', color='orange')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('γ (degrees)')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()