import numpy as np
import basis
import rocket_parser as rp
import atmosphere

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
    for k in range(len(freq_time)):
        if abs(freq_time[k] - time) < basis.timestep:
            return freq[mode][k]
    return None


def freqmass_per_time(time, mode):
    for k in range(len(freq_time)):
        if abs(freq_time[k] - time) < basis.timestep:
            return freqmass[mode][k]
    return None


def form_per_len(mode, len_):
    for k in range(len(f_len)):
        if abs(f_len[k] - len_) < basis.lenstep:
            return f_stiffness[mode][k]
    return None


def deform_per_len(mode, len_):
    for k in range(len(f_len)):
        if abs(f_len[k] - len_) < basis.lenstep:
            return f_stiffness_diff[mode][k]
    return None


def intform(mode):
    return f_stiffness_integral[mode][0]


print("ИСПРАВЛЕННЫЙ РАСЧЕТ С УЧЕТОМ РАЗМЕРНОСТЕЙ")
print("=" * 50)
cx_ = [0.33445, 0.51079, 0.46371]
cya = [0.035, 0.035, 0.035]
xf_ = [26.2, 26.2, 26.2]
p = 1.29
vel = [200, 1000, 2000]
alt = [5, 25, 45]
time = [10, 40, 80]
Qt = []
for i,a in enumerate(alt):
    atm = atmosphere.atmosphere(a*1000)
    Qt.append(atm.get_density()*vel[i]*vel[i]/2)


def aeroelastic_corrections(index, len_):
    cyy_deg, cx, xf, dypress, time_ = (
        cya[index],
        cx_[index],
        xf_[index],
        Qt[index],
        time[index],
    )
    # Исходные данные
    area = parser.maximum_area#parser.max_diameter * parser.rocket_length * 3.14
    alpha_flight_deg = 2  # полетный угол атаки, град

    # Константы
    rad_to_deg = 57.3
    deg_to_rad = 1 / 57.3

    freq_hz = [freq_per_time(time_, m) for m in range(basis.mode_num)]
    sform = [intform(m) for m in range(basis.mode_num)]
    dform = [deform_per_len(m, len_) for m in range(basis.mode_num)]
    smass = [freqmass_per_time(time_, m) for m in range(basis.mode_num)]

    freq_rad = [2 * np.pi * w for w in freq_hz]
    alpha_flight_rad = alpha_flight_deg * deg_to_rad
    cyy_rad = cyy_deg / rad_to_deg

    # Расчет интегралов
    integral_alpha = 0
    integral_cy = 0
    integral_xf_rel = 0
    for i in range(basis.mode_num):
        integral_alpha += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)
        integral_cy += dform[i] * sform[i] ** 2 / (smass[i] * freq_rad[i] ** 2)
        integral_xf_rel += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)

    # 1. Упругая добавка к углу атаки
    delta_alpha_rad = -dypress * integral_alpha * area * alpha_flight_rad * cyy_rad
    delta_alpha_deg = delta_alpha_rad * rad_to_deg
    delta_alpha_per = (delta_alpha_deg / alpha_flight_deg) * 100

    # 2. Изменение производной подъемной силы
    delta_cy_rel = -dypress * integral_cy * area * cyy_rad
    cy_elastic_deg = cyy_deg * (1 + delta_cy_rel)
    delta_cy_abs = cy_elastic_deg - cyy_deg
    delta_cy_per = delta_cy_rel * 100

    # 3. Изменение коэффициента лобового сопротивления
    cy_initial = cyy_rad * alpha_flight_rad
    A = (cx - 0.1) / (cy_initial**2) if cy_initial != 0 else 0.15
    delta_cy_abs_rad = cyy_rad * delta_cy_rel
    delta_cx_from_alpha = 2 * A * cy_initial * delta_cy_abs_rad * alpha_flight_rad
    k_cx = 0.04
    delta_cx_from_deform = k_cx * abs(delta_alpha_rad)
    delta_cx = delta_cx_from_alpha + delta_cx_from_deform
    cx_elastic = cx + delta_cx
    delta_cx_per = (delta_cx / cx) * 100

    # 4. Смещение фокуса
    delta_xf_rel = -dypress * integral_xf_rel * area * cyy_rad
    delta_xf_per = delta_xf_rel * 100
    xf_elastic = xf * (1 + delta_xf_rel)
    delta_xf_abs = xf_elastic - xf

    return {
        "delta_alpha_deg": delta_alpha_deg,
        "delta_alpha_percent": delta_alpha_per,
        "cy_elastic_deg": cy_elastic_deg,
        "delta_cy_percent": delta_cy_per,
        "delta_cy_abs": delta_cy_abs,
        "cx_elastic": cx_elastic,
        "delta_cx": delta_cx,
        "delta_cx_percent": delta_cx_per,
        "delta_xf_percent": delta_xf_per,
        "delta_xf_abs": delta_xf_abs,
        "xf_elastic": xf_elastic,
    }


print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ ДЛЯ ВСЕХ ТРЕХ РЕЖИМОВ")
print("=" * 70)

import matplotlib.pyplot as plt
cy_per = [[] for _ in range(basis.mode_num)]
cy_deg = [[] for _ in range(basis.mode_num)]
len_ar = [[] for _ in range(basis.mode_num)]
al_deg = [[] for _ in range(basis.mode_num)]
al_per = [[] for _ in range(basis.mode_num)]
cx_per = [[] for _ in range(basis.mode_num)]
cx_val = [[] for _ in range(basis.mode_num)]
xf_per = [[] for _ in range(basis.mode_num)]
xf_val = [[] for _ in range(basis.mode_num)]

for i in range(basis.mode_num):
    print(f"\nРЕЖИМ {i+1}: V = {vel[i]} м/с, t = {time[i]} с, q = {Qt[i]:.1f} Па")
    print("-" * 50)
    l = 0
    for l in f_len:
        len_ = l*basis.timestep
        results = aeroelastic_corrections(i, len_)
        cy_per[i].append(results['delta_cy_percent'])
        cy_deg[i].append(results['delta_cy_abs'])
        cx_per[i].append(results['delta_cx_percent'])
        cx_val[i].append(results['delta_cx'])
        al_deg[i].append(results['delta_alpha_deg'])
        al_per[i].append(results['delta_alpha_percent'])
        xf_per[i].append(results['delta_xf_percent'])
        xf_val[i].append(results['delta_xf_abs'])
        len_ar[i].append(len_/10)

    plt.plot(len_ar[i], cy_deg[i])

    print(
        f"Δα/α:   {sum(al_deg[i]):+.6f}° ({sum(al_per[i]):+.2f}%)"
    )
    print(f"ΔCx:    {sum(cx_val[i]):.3f} ({sum(cx_per[i]):+.2f}%)")
    print(
        f"ΔX_f:   {sum(xf_val[i]):+.3f} м ({sum(xf_per[i]):+.2f}%)"
    )
    print(f"ΔCyαsumm:      {sum(cy_deg[i]):+.5f}")
    print(f"ΔCyαsumm(%):   {sum(cy_per[i]):+.2f} %")

    # Предупреждения
    warnings = []
    if abs(results["delta_cy_percent"]) > 5:
        warnings.append("⚠️ Большое изменение производной (>5%)")
    if abs(results["delta_xf_percent"]) > 5:
        warnings.append("⚠️ Большое смещение фокуса (>5%)")
    if abs(results["delta_alpha_percent"]) > 10:
        warnings.append("⚠️ Большое изменение угла атаки (>10%)")

    if warnings:
        print("  " + "\n  ".join(warnings))

plt.xlabel("Профиль ракеты, м")
plt.ylabel("ΔCya, 1/град")
plt.grid(True)
plt.ticklabel_format(style='plain', axis='both')
plt.show()