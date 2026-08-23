
import rocket_parser as rp
import basis
import numpy as np

rocketname = basis.current_rocket
parser = rp.rocket_parser()
oscillations_file = "output/" + rocketname + "_oscillations.csv"
f_stiffness = [0] * basis.mode_num
f_stiffness_diff = [0] * basis.mode_num
f_stiffness_integral = [0] * basis.mode_num
freqmass = [0] * basis.mode_num

f_len = basis.read_array_from_csv(oscillations_file, "length")
f_stiffness[0] = basis.read_array_from_csv(oscillations_file, "form_1")
f_stiffness[1] = basis.read_array_from_csv(oscillations_file, "form_2")
f_stiffness[2] = basis.read_array_from_csv(oscillations_file, "form_3")
f_stiffness_diff[0] = basis.read_array_from_csv(oscillations_file, "difform_1")
f_stiffness_diff[1] = basis.read_array_from_csv(oscillations_file, "difform_2")
f_stiffness_diff[2] = basis.read_array_from_csv(oscillations_file, "difform_3")
f_stiffness_integral[0] = basis.read_array_from_csv(oscillations_file, "intform_1")
f_stiffness_integral[1] = basis.read_array_from_csv(oscillations_file, "intform_2")
f_stiffness_integral[2] = basis.read_array_from_csv(oscillations_file, "intform_3")

freq_file = "output/" + rocketname + "_frequency.csv"
freq = [[] for _ in range(basis.mode_num)]
freq_time = basis.read_array_from_csv(freq_file, "time")
freq[0] = basis.read_array_from_csv(freq_file, "freq_1")
freq[1] = basis.read_array_from_csv(freq_file, "freq_2")
freq[2] = basis.read_array_from_csv(freq_file, "freq_3")
freqmass[0] = basis.read_array_from_csv(freq_file, "freq_mass_1")
freqmass[1] = basis.read_array_from_csv(freq_file, "freq_mass_2")
freqmass[2] = basis.read_array_from_csv(freq_file, "freq_mass_3")

def freq_per_time(time, mode):
    return basis.get_y(time, freq_time, freq[mode])

def freqmass_per_time(time, mode):
    return basis.get_y(time, freq_time, freqmass[mode])

def form_per_len(mode, len_):
    return basis.get_y(len_, f_len, f_stiffness[mode])

def deform_per_len(mode, len_):
    return basis.get_y(len_, f_len, f_stiffness_diff[mode])

def intform(mode):
    return f_stiffness_integral[mode][0]

def aero_attack(cy, dypress, time_):
    #cy coef is required (rad)
    area = parser.maximum_area

    k_list = []
    for len_ in np.arange(0, parser.rocket_length, 0.1):
        freq_hz = [freq_per_time(time_, m) for m in range(basis.mode_num)]
        sform = [intform(m) for m in range(basis.mode_num)]
        dform = [deform_per_len(m, len_) for m in range(basis.mode_num)]
        smass = [freqmass_per_time(time_, m) for m in range(basis.mode_num)]

        freq_rad = [2 * np.pi * w for w in freq_hz]

        # Расчет интегралов
        integral_alpha = 0
        integral_cy = 0
        integral_xf_rel = 0
        for i in range(basis.mode_num):
            integral_alpha += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)
            integral_cy += dform[i] * sform[i] ** 2 / (smass[i] * freq_rad[i] ** 2)
            integral_xf_rel += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)

        k_list.append(-dypress * integral_alpha * area * cy)

    return   sum(k_list)

print(aero_attack(0.035, 20000, 20))