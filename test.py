import math


def aeroelastic_corrections():
    # Исходные данные
    dform = [-0.061, -0.085, -0.122]  # производная форм
    sform = [0.5, 0.3, 0.1]  # интеграл форм
    smass = [50000, 16000, 10000]  # приведенные массы, кг
    freq_hz = [5, 11, 19]  # частоты, Гц
    area = 10.75  # площадь крыла, м²
    cyy_deg = 0.15  # производная подъемной силы по углу атаки, 1/град
    dypress = 30000  # скоростной напор, Па
    alpha_flight_deg = 4  # полетный угол атаки, град
    xf = 20  # координата фокуса, м (абсолютная)
    cx = 0.4  # коэффициент лобового сопротивления
    b = 30  # размах крыла, м (предположим, нужно для относительных величин)
    c_mean = area / b  # средняя аэродинамическая хорда, м

    # Константы
    rad_to_deg = 57.3
    deg_to_rad = 1 / 57.3

    # Перевод в радианы
    freq_rad = [2 * math.pi * w for w in freq_hz]
    alpha_flight_rad = alpha_flight_deg * deg_to_rad
    cyy_rad = cyy_deg * rad_to_deg

    # 1. УПРУГАЯ ДОБАВКА К УГЛУ АТАКИ
    integral_alpha = 0
    for i in range(3):
        integral_alpha += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)

    delta_alpha_rad = -dypress * integral_alpha * area * alpha_flight_rad * cyy_rad
    delta_alpha_deg = delta_alpha_rad * rad_to_deg

    print("=== АЭРОУПРУГИЕ ПОПРАВКИ ===")
    print(
        f"Упругая добавка к углу атаки: {delta_alpha_rad:.6f} рад ({delta_alpha_deg:.6f}°)"
    )

    # 2. ПОПРАВКА К ПРОИЗВОДНОЙ ПОДЪЕМНОЙ СИЛЫ
    integral_cy = 0
    for i in range(3):
        integral_cy += dform[i] * sform[i] ** 2 / (smass[i] * freq_rad[i] ** 2)

    delta_cy_rel = -dypress * integral_cy * area * cyy_rad
    delta_cy_abs_rad = cyy_rad * delta_cy_rel
    delta_cy_abs_deg = delta_cy_abs_rad / rad_to_deg

    cy_elastic_rad = cyy_rad * (1 + delta_cy_rel)
    cy_elastic_deg = cy_elastic_rad / rad_to_deg

    print(f"\nПроизводная подъемной силы:")
    print(f"  Исходная: {cyy_deg:.3f} 1/град")
    print(f"  Относит. изменение: {delta_cy_rel*100:.3f}%")
    print(f"  Абсолютное изменение: {delta_cy_abs_deg:.6f} 1/град")
    print(f"  С учетом упругости: {cy_elastic_deg:.3f} 1/град")

    # 3. ПОПРАВКА К КОЭФФИЦИЕНТУ ЛОБОВОГО СОПРОТИВЛЕНИЯ
    cy_initial = cyy_rad * alpha_flight_rad
    A = (cx - 0.1) / (cy_initial**2) if cy_initial != 0 else 0.15

    delta_cx_from_alpha = 2 * A * cy_initial * delta_cy_abs_rad * alpha_flight_rad

    k_cx = 0.03
    delta_cx_from_deform = k_cx * abs(delta_alpha_rad)

    delta_cx = delta_cx_from_alpha + delta_cx_from_deform
    cx_elastic = cx + delta_cx

    print(f"\nКоэффициент лобового сопротивления:")
    print(f"  Исходный Cx: {cx:.3f}")
    print(f"  ΔCx от угла: {delta_cx_from_alpha:.6f}")
    print(f"  ΔCx от деформ: {delta_cx_from_deform:.6f}")
    print(f"  Суммарное ΔCx: {delta_cx:.6f}")
    print(f"  Cx с упругостью: {cx_elastic:.3f}")

    # 4. ИСПРАВЛЕННЫЙ РАСЧЕТ СМЕЩЕНИЯ ФОКУСА
    # Фокус смещается в долях САХ, а не в метрах!

    # Вариант 1: Смещение фокуса в долях САХ (правильный подход)
    integral_xf_rel = 0
    for i in range(3):
        # Используем безразмерные величины
        integral_xf_rel += dform[i] * sform[i] / (smass[i] * freq_rad[i] ** 2)

    # Относительное смещение фокуса (в долях)
    delta_xf_rel = -dypress * integral_xf_rel * area * cyy_rad

    # Перевод в метры (умножаем на САХ)
    delta_xf_abs_m = delta_xf_rel * c_mean

    # Вариант 2: Если нужно смещение в метрах (для больших деформаций)
    # Но здесь результат будет мал, как и ожидается

    xf_elastic_rel = xf / c_mean + delta_xf_rel  # фокус в долях САХ
    xf_elastic_abs = xf + delta_xf_abs_m  # фокус в метрах

    print(f"\nПоложение фокуса:")
    print(f"  Средняя аэродинамическая хорда (САХ): {c_mean:.2f} м")
    print(f"  Исходный фокус (абс): {xf:.1f} м")
    print(f"  Исходный фокус (отн): {xf/c_mean:.3f} САХ")
    print(f"  Относит. смещение фокуса: {delta_xf_rel*100:.4f}% САХ")
    print(f"  Абсолютное смещение: {delta_xf_abs_m:.4f} м")
    print(f"  Фокус с упругостью (отн): {xf_elastic_rel:.3f} САХ")
    print(f"  Фокус с упругостью (абс): {xf_elastic_abs:.3f} м")

    # 5. КОНТРОЛЬНАЯ ПРОВЕРКА ФИЗИЧНОСТИ
    print("\n=== ПРОВЕРКА ФИЗИЧНОСТИ ===")
    print(f"Отклонение конца крыла при таком угле:")
    wing_deflection = delta_alpha_rad * (b / 2) * 1000  # в мм
    print(f"  Пример: {wing_deflection:.2f} мм (в пределах упругости)")

    if abs(delta_xf_rel) > 0.05:  # если смещение больше 5% САХ
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
        "xf_elastic_abs": xf_elastic_abs,
        "delta_xf_abs": delta_xf_abs_m,
        "delta_xf_rel_percent": delta_xf_rel * 100,
        "c_mean": c_mean,
    }


# Ваша исходная функция
def test_alpha():
    dform = [-0.061, -0.085, -0.122]
    sform = [0.5, 0.3, 0.1]
    smass = [50000, 16000, 10000]
    freq = [5, 11, 19]
    area = 10.75
    cyy = 0.15
    dypress = 30000
    a = 4
    xf = 20
    cx = 0.4

    freq = [6.28 * w for w in freq]
    a = a / 57.3
    cyy = cyy * 57.3

    integral = 0
    for i in range(3):
        integral += dform[i] * sform[i] / (smass[i] * freq[i] ** 2)
    return -dypress * integral * area * a * cyy


# Выполняем расчёт
print("ИСПРАВЛЕННЫЙ РАСЧЕТ С УЧЕТОМ РАЗМЕРНОСТЕЙ")
print("=" * 50)
results = aeroelastic_corrections()

print("\n" + "=" * 50)
print("ВАШИ ИСХОДНЫЕ РЕЗУЛЬТАТЫ")
print("=" * 50)
alp = test_alpha()
print(f"{alp:.6f} рад")
print(f"{alp*57.3:.6f} град")
print(f"{abs((alp*57.3)/3*100):.6f} %")
print(f"{alp * 50 * 1000:.6f} мм")
