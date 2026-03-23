import math
import constants
import rocket_parser as rp
import atmosphere
import matplotlib.pyplot as plt
import numpy as np

rocketname = constants.current_rocket
parser = rp.rocket_parser()
oscillations_file = "output/" + rocketname + "_oscillations.csv"
f_stiffness = [0] * constants.mode_num
f_stiffness_diff = [0] * constants.mode_num
f_stiffness_integral = [0] * constants.mode_num
freqmass = [0] * constants.mode_num

f_len = constants.read_array_from_csv(oscillations_file, "length")
f_stiffness[0] = constants.read_array_from_csv(oscillations_file, "form_1")
f_stiffness[1] = constants.read_array_from_csv(oscillations_file, "form_2")
f_stiffness[2] = constants.read_array_from_csv(oscillations_file, "form_3")
f_stiffness_diff[0] = constants.read_array_from_csv(oscillations_file, "difform_1")
f_stiffness_diff[1] = constants.read_array_from_csv(oscillations_file, "difform_2")
f_stiffness_diff[2] = constants.read_array_from_csv(oscillations_file, "difform_3")
f_stiffness_integral[0] = constants.read_array_from_csv(oscillations_file, "intform_1")
f_stiffness_integral[1] = constants.read_array_from_csv(oscillations_file, "intform_2")
f_stiffness_integral[2] = constants.read_array_from_csv(oscillations_file, "intform_3")

freq_file = "output/" + rocketname + "_frequency.csv"
freq = [[] for _ in range(constants.mode_num)]
freq_time = constants.read_array_from_csv(freq_file, "time")
freq[0] = constants.read_array_from_csv(freq_file, "freq_1")
freq[1] = constants.read_array_from_csv(freq_file, "freq_2")
freq[2] = constants.read_array_from_csv(freq_file, "freq_3")
freqmass[0] = constants.read_array_from_csv(freq_file, "freq_mass_1")
freqmass[1] = constants.read_array_from_csv(freq_file, "freq_mass_2")
freqmass[2] = constants.read_array_from_csv(freq_file, "freq_mass_3")


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


print("ИСПРАВЛЕННЫЙ РАСЧЕТ С УЧЕТОМ РАЗМЕРНОСТЕЙ")
print("=" * 50)
cx_ = [0.33445, 0.51079, 0.45153]
cya = [0.10337, 0.12385, 0.10298]
xf_ = [26.2, 26.2, 26.2]
p = 1.29
vel = [200, 1000, 2000]
alt = [5, 25, 40]
time = [10, 40, 80]
Qt = []
for i,a in enumerate(alt):
    atm = atmosphere.atmosphere(a*1000)
    Qt.append(atm.get_density()*vel[i]*vel[i]/2)


def calculate_integral_aeroelastic_corrections(index):
    """
    Рассчитывает суммарные (интегральные) аэроупругие поправки
    путем интегрирования по всей длине ракеты
    """
    cyy_deg, cx, xf, dypress, time_ = (
        cya[index],
        cx_[index],
        xf_[index],
        Qt[index],
        time[index],
    )
    
    # Исходные данные
    area = parser.maximum_area
    alpha_flight_deg = 0  # полетный угол атаки, град

    # Константы
    rad_to_deg = 57.3
    deg_to_rad = 1 / 57.3

    freq_hz = [freq_per_time(time_, m) for m in range(constants.mode_num)]
    smass = [freqmass_per_time(time_, m) for m in range(constants.mode_num)]
    freq_rad = [2 * math.pi * w for w in freq_hz]
    alpha_flight_rad = alpha_flight_deg * deg_to_rad
    cyy_rad = cyy_deg / rad_to_deg

    # Интегрирование по длине для получения суммарных характеристик
    # Используем метод трапеций для численного интегрирования
    
    integral_alpha = 0
    integral_cy = 0
    integral_xf_rel = 0
    
    # Создаем массивы для хранения подынтегральных выражений
    integrand_alpha = []
    integrand_cy = []
    integrand_xf = []
    lengths = []
    
    # Вычисляем подынтегральные выражения для каждого сечения
    for k in range(len(f_len)):
        len_ = f_len[k]
        lengths.append(len_)
        
        dform = [deform_per_len(m, len_) for m in range(constants.mode_num)]
        sform = [intform(m) for m in range(constants.mode_num)]
        
        # Подынтегральные выражения для текущего сечения
        int_alpha = 0
        int_cy = 0
        int_xf = 0
        
        for i in range(constants.mode_num):
            if smass[i] > 0 and freq_rad[i] > 0:
                int_alpha += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)
                int_cy += dform[i] * sform[i] ** 2 / (smass[i] * freq_rad[i] ** 2)
                int_xf += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)
        
        integrand_alpha.append(int_alpha)
        integrand_cy.append(int_cy)
        integrand_xf.append(int_xf)
    
    # Численное интегрирование методом трапеций
    for k in range(len(lengths)-1):
        dl = lengths[k+1] - lengths[k]
        integral_alpha += (integrand_alpha[k] + integrand_alpha[k+1]) * dl / 2
        integral_cy += (integrand_cy[k] + integrand_cy[k+1]) * dl / 2
        integral_xf_rel += (integrand_xf[k] + integrand_xf[k+1]) * dl / 2

    # 1. Упругая добавка к углу атаки
    delta_alpha_rad = -dypress * integral_alpha * area * alpha_flight_rad * cyy_rad
    delta_alpha_deg = delta_alpha_rad * rad_to_deg
    delta_alpha_per = (delta_alpha_deg / alpha_flight_deg) * 100

    # 2. Изменение производной подъемной силы
    delta_cy_rel = -dypress * integral_cy * area * cyy_rad
    cy_elastic_deg = cyy_deg * (1 + delta_cy_rel)
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
        "cx_elastic": cx_elastic,
        "delta_cx_percent": delta_cx_per,
        "delta_xf_percent": delta_xf_per,
        "delta_xf_abs": delta_xf_abs,
        "xf_elastic": xf_elastic,
        "integral_alpha": integral_alpha,
        "integral_cy": integral_cy,
        "integral_xf": integral_xf_rel,
    }


print("\n" + "=" * 70)
print("СУММАРНЫЕ АЭРОУПРУГИЕ ПОПРАВКИ (ПО ВСЕЙ ДЛИНЕ РАКЕТЫ)")
print("=" * 70)

# Создаем сводную таблицу результатов
results_summary = []

for i in range(constants.mode_num):
    print(f"\nРЕЖИМ {i+1}: V = {vel[i]} м/с, t = {time[i]} с, q = {Qt[i]:.1f} Па")
    print("-" * 60)
    
    results = calculate_integral_aeroelastic_corrections(i)
    results_summary.append(results)
    
    print(f"Интегралы по длине:")
    print(f"  J_α = {results['integral_alpha']:.6e}")
    print(f"  J_cy = {results['integral_cy']:.6e}")
    print(f"  J_xf = {results['integral_xf']:.6e}")
    print()
    print(f"Δα/α:   {results['delta_alpha_deg']:+.6f}° ({results['delta_alpha_percent']:+.2f}%)")
    print(f"ΔCyα:   {results['cy_elastic_deg']:.3f} 1/град ({results['delta_cy_percent']:+.2f}%)")
    print(f"ΔCx:    {results['cx_elastic']:.3f} ({results['delta_cx_percent']:+.2f}%)")
    print(f"ΔX_f:   {results['delta_xf_abs']:+.3f} м ({results['delta_xf_percent']:+.2f}%)")
    
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

# Создаем сводную таблицу
print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)
print(f"{'Режим':<8} {'V, м/с':<8} {'q, Па':<8} {'Δα/α,%':<8} {'ΔCyα,%':<8} {'ΔCx,%':<8} {'ΔX_f,%':<8}")
print("-" * 60)

for i, results in enumerate(results_summary):
    print(f"{i+1:<8} {vel[i]:<8} {Qt[i]:<8.1f} "
          f"{results['delta_alpha_percent']:>+6.2f}%  "
          f"{results['delta_cy_percent']:>+6.2f}%  "
          f"{results['delta_cx_percent']:>+6.2f}%  "
          f"{results['delta_xf_percent']:>+6.2f}%")

# Визуализация суммарных результатов
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Суммарные аэроупругие поправки по режимам полета', fontsize=14)

modes = [f'Режим {i+1}\nV={vel[i]} м/с' for i in range(constants.mode_num)]
x_pos = np.arange(len(modes))

# График изменения угла атаки
axes[0, 0].bar(x_pos, [r['delta_alpha_percent'] for r in results_summary])
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(modes)
axes[0, 0].set_ylabel('Δα/α, %')
axes[0, 0].set_title('Изменение угла атаки')
axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 0].grid(True, alpha=0.3)

# График изменения производной подъемной силы
axes[0, 1].bar(x_pos, [r['delta_cy_percent'] for r in results_summary])
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(modes)
axes[0, 1].set_ylabel('ΔCyα, %')
axes[0, 1].set_title('Изменение производной подъемной силы')
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 1].grid(True, alpha=0.3)

# График изменения коэффициента лобового сопротивления
axes[1, 0].bar(x_pos, [r['delta_cx_percent'] for r in results_summary])
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(modes)
axes[1, 0].set_ylabel('ΔCx, %')
axes[1, 0].set_title('Изменение коэффициента лобового сопротивления')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].grid(True, alpha=0.3)

# График смещения фокуса
axes[1, 1].bar(x_pos, [r['delta_xf_percent'] for r in results_summary])
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(modes)
axes[1, 1].set_ylabel('ΔX_f, %')
axes[1, 1].set_title('Смещение фокуса по длине')
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Дополнительно: показываем абсолютные значения
print("\n" + "=" * 70)
print("АБСОЛЮТНЫЕ ЗНАЧЕНИЯ АЭРОДИНАМИЧЕСКИХ ХАРАКТЕРИСТИК")
print("=" * 70)
print(f"{'Режим':<8} {'Cyα исх':<8} {'Cyα упр':<8} {'Cx исх':<8} {'Cx упр':<8} {'Xf исх, м':<10} {'Xf упр, м':<10}")
print("-" * 70)

for i, results in enumerate(results_summary):
    print(f"{i+1:<8} {cya[i]:<8.3f} {results['cy_elastic_deg']:<8.3f} "
          f"{cx_[i]:<8.3f} {results['cx_elastic']:<8.3f} "
          f"{xf_[i]:<10.2f} {results['xf_elastic']:<10.2f}")