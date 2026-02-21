import constants
import matplotlib.pyplot as plt
import path
import pandas as pd
import rocket_parser as rp
import math

rocket = "master"
parser = rp.rocket_parser(path.rocket_lib + rocket + ".json")


def read_aero_coefficients_from_csv(filename):
    """
    Чтение аэродинамических коэффициентов из CSV файла
    """
    # Инициализация массивов нулями
    const_time = []
    Cwv = []
    Cww = []
    Cwb = []
    Cvv = []
    Cvb = []
    Cvw = []
    velocity = []
    Wind = []
    Csv = [[] for _ in range(constants.mode_num)]
    Csw = [[] for _ in range(constants.mode_num)]
    Csb = [[] for _ in range(constants.mode_num)]
    CsvQ = []
    CswQ = []
    CssQ = []

    try:

        df = pd.read_csv(filename)

        const_time = df["time"]

        Cwv = df["Cwy"]
        Cww = df["Cww"]
        Cwb = df["Cwb"]
        Cvv = df["Cyy"]
        Cvb = -df["Cbs"]
        Cvw = df["Cyw"]
        velocity = df["velocity"]
        Wind = df["wind"]

        Csw[0] = df["Csw1"]
        Csw[1] = df["Csw2"]
        Csw[2] = df["Csw3"]
        Csv[0] = df["Csy1"]
        Csv[1] = df["Csy2"]
        Csv[2] = df["Csy3"]
        Csb[0] = df["Csb1"]
        Csb[1] = df["Csb2"]
        Csb[2] = df["Csb3"]

        CsvQ = df["CsyQ"]
        CswQ = df["CswQ"]
        CssQ = df["CssQ"]

    except FileNotFoundError:
        print(f"Файл {filename} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")

    return (
        const_time,
        Cwv,
        Cww,
        Cwb,
        Cvv,
        Cvb,
        Cvw,
        velocity,
        Wind,
        Csv,
        Csw,
        Csb,
        CsvQ,
        CswQ,
        CssQ,
    )


def read_freq_from_csv(filename):
    freq_time = []
    freq = [[] for _ in range(constants.mode_num)]

    df = pd.read_csv(filename)
    freq_time = df["time"]
    freq[0] = df["freq_1"]
    freq[1] = df["freq_2"]
    freq[2] = df["freq_3"]

    return freq_time, freq


def read_forms_from_csv(filename):
    f_stiffness = [[0], [0], [0]]
    f_stiffness_diff = [[0], [0], [0]]

    try:
        df = pd.read_csv(filename)

        column_f = [None] * constants.mode_num
        column_df = [None] * constants.mode_num

        column_f[0] = df["form_1"]
        column_f[1] = df["form_2"]
        column_f[2] = df["form_3"]

        column_df[0] = df["difform_1"]
        column_df[1] = df["difform_2"]
        column_df[2] = df["difform_3"]

        for i in range(constants.mode_num):
            f_stiffness[i] = column_f[i].to_numpy()
            f_stiffness_diff[i] = column_df[i].to_numpy()

    except FileNotFoundError:
        print(f"Файл {filename} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")

    return f_stiffness, f_stiffness_diff


f_stiffness = [0] * constants.mode_num
f_stiffness_diff = [0] * constants.mode_num
f_stiffness, f_stiffness_diff = read_forms_from_csv("output/Master_oscillations.csv")

# Инициализация массивов
(
    const_time,
    Cwv,
    Cww,
    Cwb,
    Cvv,
    Cvb,
    Cvw,
    velocity,
    Wind,
    Csv,
    Csw,
    Csb,
    CsvQ,
    CswQ,
    CssQ,
) = read_aero_coefficients_from_csv("output/Master_dynamic_coefs.csv")
freq_time, freq = read_freq_from_csv("output/Master_frequency.csv")


def Cwv_t(time):
    return constants.get_coefficient_simple(time, const_time, Cwv)


def Cww_t(time):
    return constants.get_coefficient_simple(time, const_time, Cww)


def Cwb_t(time):
    return constants.get_coefficient_simple(time, const_time, Cwb)


def Cvv_t(time):
    return constants.get_coefficient_simple(time, const_time, Cvv)


def Cvb_t(time):
    return constants.get_coefficient_simple(time, const_time, Cvb)


def Cvw_t(time):
    return constants.get_coefficient_simple(time, const_time, Cvw)


def velocity_t(time):
    return constants.get_coefficient_simple(time, const_time, velocity)


def Wind_t(time):
    return constants.get_coefficient_simple(time, const_time, Wind)


def Csw_t(time, index):
    return constants.get_coefficient_simple(time, const_time, Csw[index])


def Csv_t(time, index):
    return constants.get_coefficient_simple(time, const_time, Csv[index])


def Csb_t(time, index):
    return constants.get_coefficient_simple(time, const_time, Csb[index])


def freq_t(time, index):
    return constants.get_coefficient_simple(time, freq_time, freq[index])


def CsvQ_t(time):
    return constants.get_coefficient_simple(time, const_time, CsvQ)


def CswQ_t(time):
    return constants.get_coefficient_simple(time, const_time, CswQ)


def CssQ_t(time):
    return constants.get_coefficient_simple(time, const_time, CssQ)


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
h = 2 / (2 * math.pi * freq_t(work_time[0], 2) * math.sqrt(1 + dempher * dempher))


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
    dds = constants.mode_num * [0]
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
            for ci in range(constants.mode_num):
                dv += Csv_t(t, ci) * s[ci]
                ddw += Csw_t(t, ci) * s[ci]
                fre = freq_t(t, ci) * (2 * math.pi)
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
