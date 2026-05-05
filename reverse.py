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


# Вычисляем угол напрямую, без средних точек
vert_speed_all = np.gradient(v_altitude, v_time)

v_tetta = []
for i in range(len(v_speed)):
    if v_speed[i] != 0:
        argument = vert_speed_all[i] / v_speed[i]
        argument = np.clip(argument, -1.0, 1.0)
        v_tetta.append(math.asin(argument))
    else:
        v_tetta.append(v_tetta[-1] if v_tetta else 0)

v_tetta_deg = np.degrees(v_tetta)

# Строим график в исходных временных точках
plt.plot(v_time, v_tetta_deg)
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Flight Path Angle vs Time')
plt.grid(True)
plt.show()