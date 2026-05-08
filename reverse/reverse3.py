import constants
import matplotlib.pyplot as plt
import numpy as np
import atmosphere as atmo
import aerodynamics as aero
import rocket_parser as rp


flight_file = "test.csv"
v_time     =      np.array(constants.read_array_from_csv(flight_file, "time"))
v_altitude =      np.array(constants.read_array_from_csv(flight_file, "altitude"))
v_vel      =      np.array(constants.read_array_from_csv(flight_file, "velocity"))
v_tetta    =      np.array(constants.read_array_from_csv(flight_file, "angle"))

parser = rp.rocket_parser()
G = aero.UnionStream()
G.set_elnumber(parser.get_block_number() + 1)
G.set_diameter(parser.get_part_diameters())
G.set_length(parser.get_part_length())

v_mach = []
v_grav = []
v_altitude = [a*1000 for a in v_altitude]
v_density = []
v_cx = []
for v,a in zip(v_vel, v_altitude):
    at = atmo.atmosphere(a)
    v_mach.append(v/at.get_SV())
    v_grav.append(at.get_AOG())
    G.calculate_CXY(v,a,0)
    v_cx.append(G.CX)
    v_density.append(at.get_density())

v_density = np.array(v_density)
v_Q = v_density * v_vel * v_vel /2

start_mass = 549000
final_mass = 140000
delta = 3100
S = 10.75

v_mass = np.linspace(start_mass, final_mass, len(v_time))
deltamass = np.gradient(v_mass, v_time, edge_order=2)
thrust = - deltamass*delta

acceleration = np.gradient(v_vel, v_time, edge_order=2)
angle_rad = np.radians(v_tetta)
drag_force = thrust - v_mass * acceleration - v_mass * v_grav*np.sin(angle_rad)
cx = drag_force/S/v_Q

plt.plot(v_time, cx)
plt.plot(v_time, v_cx)
plt.xlim(30, 130)
plt.ylim(-10,10)
plt.show()

print(cx)