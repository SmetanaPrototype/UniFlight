import constants
import matplotlib.pyplot as plt
import numpy as np
import atmosphere as atmo
import aerodynamics as aero
import rocket_parser as rp

flight_file = "missions/crs16_falcon9_051218.csv"
v_time = constants.read_array_from_csv(flight_file, "time")
v_velocity = constants.read_array_from_csv(flight_file, "velocity")
v_altitude = constants.read_array_from_csv(flight_file, "altitude")
v_tetta = constants.read_array_from_csv(flight_file, "angle")
v_dypressure = constants.read_array_from_csv(flight_file, "q")
v_fullac = constants.read_array_from_csv(flight_file, "acceleration")

parser = rp.rocket_parser()
G = aero.UnionStream()
G.set_elnumber(parser.get_block_number() + 1)
G.set_diameter(parser.get_part_diameters())
G.set_length(parser.get_part_length())

v_mach = []
v_grav = []
v_altitude = [a*1000 for a in v_altitude]
v_cx = []
for v,a in zip(v_velocity, v_altitude):
    at = atmo.atmosphere(a)
    v_mach.append(v/at.get_SV())
    v_grav.append(at.get_AOG())
    G.calculate_CXY(v,a,0)
    v_cx.append(G.CX)
# Преобразование
fullac = np.array(v_fullac)
grav = np.array(v_grav)
time = np.array(v_time)
mach = np.array(v_mach)
velocity = np.array(v_velocity)
angle_deg = np.array(v_tetta)

dyn_pressure = np.array(v_dypressure)
altitude = np.array(v_altitude)
cx = np.array(v_cx)

# Параметры
S = constants.cross_sectional_area(3.7)
mass0 = 549000
delta = 3100

angle_rad = np.radians(angle_deg)
angular_velocity_rad = np.radians(np.gradient(angle_deg, time))

acceleration = fullac - grav*np.sin(angle_rad)

acceleration_observed = np.gradient(velocity, time)
x_acc = fullac - acceleration_observed - grav * np.sin(angle_rad)
char_acc = fullac
v_char = np.cumsum(char_acc)
char_speed = np.array(v_char)
mass = mass0*np.exp(-char_speed/delta)

drag_force = mass*x_acc
cx = drag_force/dyn_pressure/S
# deltamass = np.gradient(mass, time)
# thrust = - deltamass*delta
# # print(deltamass)

# drag_force = thrust - mass * acceleration - mass * grav*np.sin(angle_rad)

# # # lift_force = mass * velocity * angular_velocity_rad + mass * v_grav * np.cos(angle_rad)


# # # drag_force = thrust - mass * acceleration - mass * v_grav * np.sin(angle_rad)
# # # lift_force = mass * velocity * angular_velocity_rad + mass * v_grav * np.cos(angle_rad) * (1 - velocity**2 / (v_grav * (altitude + constants.earth_radius)))

# # # th = mass * acceleration + dyn_pressure * cx * S + mass * v_grav * np.sin(angle_rad)

gold_cx = []
for i in mach:
    if i < 0.8:
        gold_cx.append(0.05)
    elif i < 1.1:
        gold_cx.append(0.7)
    elif i < 1.5:
        gold_cx.append(0.3)
    else:
        gold_cx.append(0.08)

gold_cx = np.array(gold_cx)

plt.plot(mach, cx)
plt.plot(mach, gold_cx)
plt.xlim(0.3, 2)
plt.ylim(-1,1)
plt.show()

print("mach:")
print(mach)
print("cx:")
print(cx)
# print(fullac - acceleration - grav*np.sin(angle_rad))