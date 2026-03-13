import math
import constants
import rocket_parser as rp

rocketname = constants.current_rocket

oscillations_file = "output/"+rocketname+"_oscillations.csv"
f_stiffness = [0] * constants.mode_num
f_stiffness_diff = [0] * constants.mode_num
f_stiffness_integral = [0] * constants.mode_num
freqmass = [0] * constants.mode_num

f_len               = constants.read_array_from_csv(oscillations_file, "length")
f_stiffness[0]      = constants.read_array_from_csv(oscillations_file, "form_1")
f_stiffness[1]      = constants.read_array_from_csv(oscillations_file, "form_2")
f_stiffness[2]      = constants.read_array_from_csv(oscillations_file, "form_3")
f_stiffness_diff[0] = constants.read_array_from_csv(oscillations_file, "difform_1")
f_stiffness_diff[1] = constants.read_array_from_csv(oscillations_file, "difform_2")
f_stiffness_diff[2] = constants.read_array_from_csv(oscillations_file, "difform_3")
f_stiffness_integral[0] = constants.read_array_from_csv(oscillations_file, "intform_1")
f_stiffness_integral[1] = constants.read_array_from_csv(oscillations_file, "intform_2")
f_stiffness_integral[2] = constants.read_array_from_csv(oscillations_file, "intform_3")

freq_file = "output/"+rocketname+"_frequency.csv"
freq = [[] for _ in range(constants.mode_num)]
freq_time = constants.read_array_from_csv(freq_file, "time")
freq[0]   = constants.read_array_from_csv(freq_file, "freq_1")
freq[1]   = constants.read_array_from_csv(freq_file, "freq_2")
freq[2]   = constants.read_array_from_csv(freq_file, "freq_3")
freqmass[0]   = constants.read_array_from_csv(freq_file, "freq_mass_1")
freqmass[1]   = constants.read_array_from_csv(freq_file, "freq_mass_2")
freqmass[2]   = constants.read_array_from_csv(freq_file, "freq_mass_3")

def freq_per_time(time, mode):
    for k in range(len(freq_time)):
        if abs(freq_time[k] - time) < constants.timestep:
            return freq[mode][k]
    return None

def freqmass_per_time(time, mode):
    for k in range(len(freq_time)):
        if abs(freq_time[k] - time) < constants.timestep:
            return freqmass[mode][k]
    return None

def form_per_len(mode, len_):
    for k in range(len(f_len)):
        if abs(f_len[k] - len_) < constants.lenstep:
            return f_stiffness[mode][k]
    return None

def deform_per_len(mode, len_):
    for k in range(len(f_len)):
        if abs(f_len[k] - len_) < constants.lenstep:
            return f_stiffness_diff[mode][k]
    return None

def intform(mode):
    return f_stiffness_integral[mode][0]

def aeroelastic_corrections():
    # Исходные данные
    parser = rp.rocket_parser()
    area = parser.max_diameter * parser.rocket_length * 3.14
    cyy_deg = 0.15  # производная подъемной силы по углу атаки, 1/град
    dypress = 30000  # скоростной напор, Па
    alpha_flight_deg = 3  # полетный угол атаки, град
    xf = 20  # координата фокуса, м (абсолютная)
    cx = 0.4  # коэффициент лобового сопротивления

    # Константы
    rad_to_deg = 57.3
    deg_to_rad = 1 / 57.3

    time_ = 10
    len_  = 10

    freq_hz = [freq_per_time(time_, m) for m in range(constants.mode_num)]
    sform   = [intform(m) for m in range(constants.mode_num)]
    dform   = [deform_per_len(m, len_) for m in range(constants.mode_num)]
    smass   = [freqmass_per_time(time_, m) for m in range(constants.mode_num)]

    print("=== ИСХОДНЫЕ ДАННЫЕ ===")
    print("Частоты:")
    print(freq_hz)
    print("Интергалы форм")
    print(sform)
    print("Производные форм")
    print(dform)
    print("Приведенные массы")
    print(smass)

    freq_rad = [2 * math.pi * w for w in freq_hz]
    alpha_flight_rad = alpha_flight_deg * deg_to_rad
    cyy_rad = cyy_deg / rad_to_deg

    integral_alpha = 0
    for i in range(constants.mode_num):
        integral_alpha += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)

    delta_alpha_rad = -dypress * integral_alpha * area * alpha_flight_rad * cyy_rad
    delta_alpha_deg = delta_alpha_rad * rad_to_deg
    delta_alpha_per = (delta_alpha_deg/alpha_flight_deg)*100

    print("=== АЭРОУПРУГИЕ ПОПРАВКИ ===")
    print(
        f"Упругая добавка к углу атаки: {delta_alpha_rad:.6f} рад ({delta_alpha_deg:.6f}°) {delta_alpha_per:.6f} %"
    )

    integral_cy = 0
    for i in range(constants.mode_num):
        integral_cy += dform[i] * sform[i] ** 2 / (smass[i] * freq_rad[i] ** 2)

    delta_cy_rel = -dypress * integral_cy * area * cyy_rad
    delta_cy_abs_rad = cyy_rad * delta_cy_rel
    delta_cy_abs_deg = delta_cy_abs_rad / rad_to_deg

    cy_elastic_rad = cyy_rad * (1 + delta_cy_rel)
    cy_elastic_deg = cy_elastic_rad * rad_to_deg

    print(f"\nПроизводная подъемной силы:")
    print(f"  Исходная: {cyy_deg:.3f} 1/град")
    print(f"  Относит. изменение: {delta_cy_rel*100:.3f}%")
    print(f"  Абсолютное изменение: {delta_cy_abs_deg:.6f} 1/град")
    print(f"  С учетом упругости: {cy_elastic_deg:.6f} 1/град")

    cy_initial = cyy_rad * alpha_flight_rad
    A = (cx - 0.1) / (cy_initial**2) if cy_initial != 0 else 0.15

    delta_cx_from_alpha = 2 * A * cy_initial * delta_cy_abs_rad * alpha_flight_rad

    k_cx = 0
    delta_cx_from_deform = k_cx * abs(delta_alpha_rad)

    delta_cx = delta_cx_from_alpha + delta_cx_from_deform
    cx_elastic = cx + delta_cx

    print(f"\nКоэффициент лобового сопротивления:")
    print(f"  Исходный Cx: {cx:.3f}")
    print(f"  ΔCx от угла: {delta_cx_from_alpha:.6f}")
    print(f"  ΔCx от деформ: {delta_cx_from_deform:.6f}")
    print(f"  Суммарное ΔCx: {delta_cx:.6f} {delta_cx/cx*100:.6f} %")
    print(f"  Cx с упругостью: {cx_elastic:.3f}")

    integral_xf_rel = 0
    for i in range(3):
        integral_xf_rel += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)

    delta_xf_rel = -dypress * integral_xf_rel * area * cyy_rad

    for i in range(constants.mode_num):
        contrib_i = dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)
        print(f"Мода {i+1}: вклад {contrib_i/integral_alpha*100:.1f}%")

    print(f"\nПоложение фокуса:")
    print(f"  Исходный фокус (абс): {xf:.1f} м")
    print(f"  Относит. смещение фокуса: {delta_xf_rel*100:.4f}% САХ")


    if abs(delta_xf_rel) > 0.05:
        print("  ⚠️ Подозрительно большое смещение фокуса!")
    else:
        print("  ✓ Смещение фокуса в пределах нормы")

    if abs(delta_cy_rel) > 0.05:
        print("  ⚠️ Большое изменение производной!")
    else:
        print("  ✓ Изменение производной в пределах нормы")

    return {
        "delta_alpha_rad": delta_alpha_rad,
        "delta_alpha_deg": delta_alpha_deg,
        "cy_original_deg": cyy_deg,
        "cy_elastic_deg": cy_elastic_deg,
        "delta_cy_percent": delta_cy_rel * 100,
        "cx_original": cx,
        "cx_elastic": cx_elastic,
        "xf_original": xf,
        "delta_xf_rel_percent": delta_xf_rel * 100,
    }

print("ИСПРАВЛЕННЫЙ РАСЧЕТ С УЧЕТОМ РАЗМЕРНОСТЕЙ")
print("=" * 50)
results = aeroelastic_corrections()