import constants
import matplotlib.pyplot as plt
import numpy as np
import atmosphere as atmo

flight_file = "missions/crs16_falcon9_051218.csv"
v_time = constants.read_array_from_csv(flight_file, "time")
v_velocity = constants.read_array_from_csv(flight_file, "velocity")
v_altitude = constants.read_array_from_csv(flight_file, "altitude")
v_tetta = constants.read_array_from_csv(flight_file, "angle")
v_acceleration = constants.read_array_from_csv(flight_file, "acceleration")
v_dypressure = constants.read_array_from_csv(flight_file, "q")
v_thrust = constants.read_array_from_csv(flight_file, "thrust")
v_mass = constants.read_array_from_csv(flight_file, "mass")

v_mach = []
for v,a in zip(v_velocity, v_altitude):
    at = atmo.atmosphere(a*1000)
    v_mach.append(v/at.get_SV())
# Преобразование
time = np.array(v_time)
velocity = np.array(v_velocity)
angle_deg = np.array(v_tetta)
thrust = np.array(v_thrust)
mass = np.array(v_mass)
acceleration = np.array(v_acceleration)
dyn_pressure = np.array(v_dypressure)

# Параметры
S = constants.cross_sectional_area(3.3)
g = 9.81

angle_rad = np.radians(angle_deg)
drag_force = thrust - mass * acceleration
angular_velocity_rad = np.radians(np.gradient(angle_deg, time))
lift_force = mass * velocity * angular_velocity_rad + mass * g * np.cos(angle_rad)

q_threshold = 100
Cx = np.full_like(dyn_pressure, np.nan)
Cy = np.full_like(dyn_pressure, np.nan)

valid = dyn_pressure > q_threshold
Cx[valid] = drag_force[valid] / (S * dyn_pressure[valid] * g)
Cy[valid] = lift_force[valid] / (S * dyn_pressure[valid] * g)


print("=" * 60)
print("АЭРОДИНАМИЧЕСКИЕ КОЭФФИЦИЕНТЫ (q в Па)")
print("=" * 60)
print(f"Диапазон Cx: {np.nanmin(Cx):.3f} ... {np.nanmax(Cx):.3f}")
print(f"Диапазон Cy: {np.nanmin(Cy):.3f} ... {np.nanmax(Cy):.3f}")

# ПЕЧАТЬ ЗНАЧЕНИЙ С 20 ДО 80 СЕКУНДЫ
print("\n" + "=" * 60)
print("ЗНАЧЕНИЯ Cx И Cy ОТ 20 ДО 80 СЕКУНДЫ")
print("=" * 60)
print(f"{'Time (s)':>10} {'Cx':>12} {'Cy':>12}")
print("-" * 40)

for i in range(len(time)):
    if 20 <= time[i] <= 80:
        cx_val = Cx[i] if not np.isnan(Cx[i]) else 0
        cy_val = Cy[i] if not np.isnan(Cy[i]) else 0
        print(f"{time[i]:10.1f} {cx_val:12.4f} {cy_val:12.4f}")

plt.figure(figsize=(12, 6))
plt.plot(v_mach, Cx, 'b-', linewidth=1.5, label='Cx (Drag)')
plt.plot(v_mach, Cy, 'r-', linewidth=1.5, label='Cy (Lift)')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.xlabel('Mach')
plt.ylabel('Coefficient')
plt.title('Aerodynamic Coefficients Cx and Cy vs Mach')
# plt.xlim(20,80)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-2, 2)
plt.show()