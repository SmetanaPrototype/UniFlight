import basis
import matplotlib.pyplot as plt
import numpy as np
import atmosphere as atmo

# flight_file = "output/falcon_ball_data.csv"
flight_file = "output/falcon_ball_data.csv"
v_time =       np.array(basis.read_array_from_csv(flight_file, "time"))
v_tetta =      np.array(basis.read_array_from_csv(flight_file, "tetta"))
v_dypressure = np.array(basis.read_array_from_csv(flight_file, "q"))
v_accel      = np.array(basis.read_array_from_csv(flight_file, "acceleration"))
v_thrust =     np.array(basis.read_array_from_csv(flight_file, "thrust"))
v_mass =       np.array(basis.read_array_from_csv(flight_file, "mass"))
v_aog =        np.array(basis.read_array_from_csv(flight_file, "g"))

S = 10.75

Cx = v_thrust - v_mass*v_accel - v_mass * v_aog * np.sin(np.radians(v_tetta))
Cx = Cx/v_dypressure/S

cx_array = np.array([
    0,
    0,
    0,
    1.5659e-14,
    0,
    0,
    0,
    0,
    0,
    -9.35567e-15,
    0,
    0.000441804,
    0.002377348,
    0.004168118,
    0.010392215,
    0.022994409,
    0.033793334,
    0.050900922,
    0.074690794,
    0.084250948,
    0.081607696,
    0.084962984,
    0.065185634,
    0.053198691,
    0.057680048,
    0.045565577,
    0.019849633,
    -1.83897e-15,
    -1.81645e-15,
    0,
    -0.006583554,
    0,
    0,
    1.78109e-15,
    0.01699477,
    0.022465232,
    0.025720076,
    0.02972781,
    0.031154606,
    0.02634479,
    0.024870031,
    0.029749592,
    0.041440731,
    0.048594305,
    0.050206361,
    0.039815588,
    0.006376766,
    1.87171e-15,
    0.000896753,
    0.008580109,
    1.87385e-15,
    0,
    0.000802677,
    0.00824294,
    0.008330147,
    0.006679858,
    0.015044964,
    0.025672503,
    0.041820296,
    0.041733868,
    0.043443143
])

# perc = abs(Cx - cx_array)/cx_array*100
plt.plot(v_time, Cx)
plt.plot(v_time, cx_array)
plt.xlim(20,80)
plt.ylim(-0.2,0.6)
plt.show()
# plt.plot(v_time, perc)
plt.show()