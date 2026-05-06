import constants
import matplotlib.pyplot as plt
import numpy as np
import atmosphere as atmo

# flight_file = "output/falcon_ball_data.csv"
flight_file = "output/falcon_ball_data.csv"
v_time =       np.array(constants.read_array_from_csv(flight_file, "time"))
v_tetta =      np.array(constants.read_array_from_csv(flight_file, "tetta"))
v_dypressure = np.array(constants.read_array_from_csv(flight_file, "q"))
v_accel      = np.array(constants.read_array_from_csv(flight_file, "acceleration"))
v_thrust =     np.array(constants.read_array_from_csv(flight_file, "thrust"))
v_mass =       np.array(constants.read_array_from_csv(flight_file, "mass"))
v_aog =        np.array(constants.read_array_from_csv(flight_file, "g"))

S = 10.75

Cx = v_thrust - v_mass*v_accel - v_mass * v_aog * np.sin(np.radians(v_tetta))
# Cx = Cx/v_dypressure/S

plt.plot(v_time, Cx)
plt.xlim(0,150)
# plt.ylim(0,1)
plt.show()