import constants
import matplotlib.pyplot as plt
import numpy as np
import atmosphere as atmo

flight_file = "missions/crs16_falcon9_051218.csv"
v_time = constants.read_array_from_csv(flight_file, "time")
v_velocity = constants.read_array_from_csv(flight_file, "velocity")
v_altitude = constants.read_array_from_csv(flight_file, "altitude")
v_tetta = constants.read_array_from_csv(flight_file, "angle")
v_dypressure = constants.read_array_from_csv(flight_file, "q")
v_thrust = constants.read_array_from_csv(flight_file, "thrust")
v_mass = constants.read_array_from_csv(flight_file, "mass")

v_mach = []
v_grav = []
v_altitude = [a*1000 for a in v_altitude]
v_dypressure = []
for v,a in zip(v_velocity, v_altitude):
    at = atmo.atmosphere(a)
    v_mach.append(v/at.get_SV())
    v_grav.append(at.get_AOG())
# Преобразование
time = np.array(v_time)
mach = np.array(v_mach)
velocity = np.array(v_velocity)
angle_deg = np.array(v_tetta)
thrust = np.array(v_thrust)
mass = np.array(v_mass)
acceleration = np.gradient(velocity, time)
dyn_pressure = np.array(v_dypressure)
altitude = np.array(v_altitude)

# Параметры
S = constants.cross_sectional_area(3.7)

angle_rad = np.radians(angle_deg)
angular_velocity_rad = np.radians(np.gradient(angle_deg, time))

# drag_force = thrust - mass * acceleration - mass * v_grav * np.sin(angle_rad)
# lift_force = mass * velocity * angular_velocity_rad + mass * v_grav * np.cos(angle_rad)

drag_force = thrust - mass * acceleration - mass * v_grav * np.sin(angle_rad)
lift_force = mass * velocity * angular_velocity_rad + mass * v_grav * np.cos(angle_rad) * (1 - velocity**2 / (v_grav * (altitude + constants.earth_radius)))

q_threshold = 100
Cx = np.full_like(dyn_pressure, np.nan)
Cy = np.full_like(dyn_pressure, np.nan)

valid = dyn_pressure > q_threshold
Cx[valid] = drag_force[valid] / (S * dyn_pressure[valid])
Cy[valid] = lift_force[valid] / (S * dyn_pressure[valid])


print("=" * 60)
print("АЭРОДИНАМИЧЕСКИЕ КОЭФФИЦИЕНТЫ (q в Па)")
print("=" * 60)
print(f"Диапазон Cx: {np.nanmin(Cx):.3f} ... {np.nanmax(Cx):.3f}")
print(f"Диапазон Cy: {np.nanmin(Cy):.3f} ... {np.nanmax(Cy):.3f}")

# ПЕЧАТЬ ЗНАЧЕНИЙ С 20 ДО 80 СЕКУНДЫ
print("\n" + "=" * 60)
print("ЗНАЧЕНИЯ Cx И Cy ОТ 20 ДО 80 СЕКУНДЫ")
print("=" * 60)
print(f"{'Mach':>10} {'Cx':>12} {'Cy':>12}")
print("-" * 40)

plt.figure(figsize=(12, 6))
# plt.plot(v_mach, Cx, 'b-', linewidth=1.5, label='Cx (Drag)')
# plt.plot(v_mach, Cy, 'r-', linewidth=1.5, label='Cy (Lift)')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.xlabel('Mach')
plt.ylabel('Coefficient')
plt.title('Aerodynamic Coefficients Cx vs Mach')
# plt.xlim(20,80)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-2, 5)

flight_file = "output/falcon_10stat.csv"
z_mach = constants.read_array_from_csv(flight_file, "mach")
z_cx = constants.read_array_from_csv(flight_file, "cx")

z_cx_start = []
for z in z_mach:
    z_cx_start.append(constants.get_coefficient_simple(z, v_mach, Cx))

plt.plot(z_mach, z_cx_start, 'b-', linewidth=1.5, label='Cx (reverse)')
plt.plot(z_mach, z_cx      , 'r-', linewidth=1.5, label='Cx (static)')

for i in range(len(z_cx)):
    if not np.isnan(z_cx_start[i]) and z_cx[i] > 0:
        err = abs(z_cx[i] - z_cx_start[i]) / z_cx[i] * 100
        print(f"Stat: {z_cx[i]:.4f}, Reverse: {z_cx_start[i]:.4f}, Err: {err:.2f}%")

plt.show()