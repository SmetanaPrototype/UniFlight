import basis
import matplotlib.pyplot as plt
import rocket_parser as rp
import numpy as np

rocketname = basis.current_rocket
parser = rp.rocket_parser()

oscillations_file = "output/"+rocketname+"_oscillations.csv"
f_stiffness = [0] * basis.mode_num
f_stiffness_diff = [0] * basis.mode_num

f_stiffness[0]      = basis.read_array_from_csv(oscillations_file, "form_1")
f_stiffness[1]      = basis.read_array_from_csv(oscillations_file, "form_2")
f_stiffness[2]      = basis.read_array_from_csv(oscillations_file, "form_3")
f_stiffness_diff[0] = basis.read_array_from_csv(oscillations_file, "difform_1")
f_stiffness_diff[1] = basis.read_array_from_csv(oscillations_file, "difform_2")
f_stiffness_diff[2] = basis.read_array_from_csv(oscillations_file, "difform_3")

freq_file = "output/"+rocketname+"_frequency.csv"
freq = [[] for _ in range(basis.mode_num)]
freq_time = basis.read_array_from_csv(freq_file, "time")
freq[0]   = basis.read_array_from_csv(freq_file, "freq_1")
freq[1]   = basis.read_array_from_csv(freq_file, "freq_2")
freq[2]   = basis.read_array_from_csv(freq_file, "freq_3")

dynamic_file = "output/"+rocketname+"_dynamic_coefs.csv"
const_time = []
Cwv = []
Cww = []
Cwb = []
Cvv = []
Cvb = []
Cvw = []
velocity = []
Wind = []
Csv = [[] for _ in range(basis.mode_num)]
Csw = [[] for _ in range(basis.mode_num)]
Csb = [[] for _ in range(basis.mode_num)]
CsvQ = []
CswQ = []
CssQ = []

const_time = basis.read_array_from_csv(dynamic_file, "time")
Cwv        = basis.read_array_from_csv(dynamic_file, "Cwy")
Cww        = basis.read_array_from_csv(dynamic_file, "Cww")
Cwb        = basis.read_array_from_csv(dynamic_file, "Cwb")
Cvv        = basis.read_array_from_csv(dynamic_file, "Cyy")
Cvb        = basis.read_array_from_csv(dynamic_file, "Cbs")
Cvw        = basis.read_array_from_csv(dynamic_file, "Cyw")
velocity   = basis.read_array_from_csv(dynamic_file, "velocity")
Wind       = basis.read_array_from_csv(dynamic_file, "wind")
Csw[0]     = basis.read_array_from_csv(dynamic_file, "Csw1")
Csw[1]     = basis.read_array_from_csv(dynamic_file, "Csw2")
Csw[2]     = basis.read_array_from_csv(dynamic_file, "Csw3")
Csv[0]     = basis.read_array_from_csv(dynamic_file, "Csy1")
Csv[1]     = basis.read_array_from_csv(dynamic_file, "Csy2")
Csv[2]     = basis.read_array_from_csv(dynamic_file, "Csy3")
Csb[0]     = basis.read_array_from_csv(dynamic_file, "Csb1")
Csb[1]     = basis.read_array_from_csv(dynamic_file, "Csb2")
Csb[2]     = basis.read_array_from_csv(dynamic_file, "Csb3")
CsvQ       = basis.read_array_from_csv(dynamic_file, "CsyQ")
CswQ       = basis.read_array_from_csv(dynamic_file, "CswQ")
CssQ       = basis.read_array_from_csv(dynamic_file, "CssQ")


def Cwv_t(time):
    return basis.get_y(time, const_time, Cwv)


def Cww_t(time):
    return basis.get_y(time, const_time, Cww)


def Cwb_t(time):
    return basis.get_y(time, const_time, Cwb)


def Cvv_t(time):
    return basis.get_y(time, const_time, Cvv)


def Cvb_t(time):
    return basis.get_y(time, const_time, Cvb)


def Cvw_t(time):
    return basis.get_y(time, const_time, Cvw)


def velocity_t(time):
    return basis.get_y(time, const_time, velocity)


def Wind_t(time):
    return basis.get_y(time, const_time, Wind)


def Csw_t(time, index):
    return basis.get_y(time, const_time, Csw[index])


def Csv_t(time, index):
    return basis.get_y(time, const_time, Csv[index])


def Csb_t(time, index):
    return basis.get_y(time, const_time, Csb[index])


def freq_t(time, index):
    return basis.get_y(time, freq_time, freq[index])


def CsvQ_t(time):
    return basis.get_y(time, const_time, CsvQ)


def CswQ_t(time):
    return basis.get_y(time, const_time, CswQ)


def CssQ_t(time):
    return basis.get_y(time, const_time, CssQ)


# control coefficients
a0 = 4
a1 = 2
a2 = 0.0004
a3 = 10 * a2
t1 = 0.38
t2 = 0.04

# constants
work_time = parser.get_work_time()
dempher = 0.06
h = 2 / (2 * np.pi * freq_t(work_time[0], 2) * np.sqrt(1 + dempher * dempher))


def calculate_parameters(include_oscillations, include_aerostiffness):

    if (include_aerostiffness) and not (include_oscillations):
        print("BOOL input failure! Please check arguments...")
        return

    uc = 0
    duc = 0
    dduc = 0
    v = 0
    y = 0
    dv = 0
    w = 0
    dw = 0
    ddw = 0
    s = [f_stiffness[0][0], f_stiffness[1][0], f_stiffness[2][0]]
    ds = [f_stiffness_diff[0][0], f_stiffness_diff[1][0], f_stiffness_diff[2][0]]
    dds = basis.mode_num * [0]
    t = 0

    pitch_vector = []
    velocity_vector = []
    movement_vector = []
    cotroL_sec_vector = []
    time_vector = []
    zero_line = []

    while t < work_time[0]:

        dv = -Cvw_t(t) * w - Cvv_t(t) * v - Cvb_t(t) * uc
        ddw = -Cww_t(t) * w - Cwv_t(t) * v - Cwb_t(t) * uc

        dv += Cvv_t(t) * Wind_t(t)
        ddw += Cwv_t(t) * Wind_t(t)

        if include_oscillations:
            for ci in range(basis.mode_num):
                dv += Csv_t(t, ci) * s[ci]
                ddw += Csw_t(t, ci) * s[ci]
                fre = freq_t(t, ci) * (2 * np.pi)
                dds[ci] = (
                    Csb_t(t, ci) * uc - pow(fre, 2) * s[ci] - 2 * dempher * fre * ds[ci]
                )
                if ci == 0 and include_aerostiffness:
                    dv += CsvQ_t(t) * s[0]
                    ddw += CswQ_t(t) * s[0]
                    dds[0] += CssQ_t(t) * (w - (v - Wind_t(t)) / velocity_t(t))

                ds[ci] += h * dds[ci]
                s[ci] += h * ds[ci]

        dduc = (-t1 * duc - uc + a0 * w + a1 * dw + a2 * y + a3 * v) / t2

        v += h * dv
        y += h * v
        # if (t<2):
        #     print(v)
        dw += h * ddw
        w += h * dw
        t += h
        duc += h * dduc
        uc += h * duc

        if uc > 7 / 57.3:
            uc = 7 / 57.3

        if uc < -7 / 57.3:
            uc = -7 / 57.3

        if t < 100:
            velocity_vector.append(float(v))
            pitch_vector.append(float(w * 57.3))
            movement_vector.append(float(y))
            cotroL_sec_vector.append(float(uc * 57.3))
            time_vector.append(float(t))

            zero_line.append(float(0))

        # destroying
        if w * 57.3 > 5:
            break

    plt.subplot(4, 1, 1)
    plt.plot(time_vector, velocity_vector)
    plt.ylabel("Скорость(t), м/c", color="gray")
    plt.plot(time_vector, zero_line)
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(time_vector, pitch_vector)
    plt.ylabel("Угол (t), град", color="gray")
    plt.plot(time_vector, zero_line)
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(time_vector, movement_vector)
    plt.ylabel("Перемещение(t), м", color="gray")
    plt.plot(time_vector, zero_line)
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(time_vector, cotroL_sec_vector)
    plt.ylabel("Поворот ОУ(t), град", color="gray")
    plt.plot(time_vector, zero_line)
    plt.grid(True)

# on/off oscillations & aerostiffness
calculate_parameters(False, False)
calculate_parameters(True, False)
calculate_parameters(True, True)
plt.show()
