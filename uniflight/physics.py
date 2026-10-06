import enum

import numpy as np

bar_values = (4.73, 7.853, 10.996, 14.137, 17.279)

class Density(enum.Enum):
    LOX = 1100.0      # Жидкий кислород
    RP1 = 810.0      # Очищенный керосин
    UDMH = 790.0      # Несимметричный диметилгидразин
    N2O4 = 1450.0     # Азотный тетраоксид
    HTPB = 1800.0     # Твердое топливо (гидроксил-терминированный полибутадиен)
    AP = 1950.0       # Перхлорат аммония (окислитель в твердом топливе)
    RG1 = 440.0      # Российский ракетный керосин
    UH25 = 880.0      # Смесь UDMH и гидразина
    CH4 = 422.0       # Сжиженный метан (жидкий природный газ)
    LH2 = 70.8        # Жидкий водород (Liquid Hydrogen)
    Naphthyl = 840.0  # Нафтил (модифицированный керосин для Союз-5, плотность между RP-1 и RG-1)

class FuelRatio(enum.Enum):
    LOX_RP1 = 2.56
    LOX_RG1 = 2.6
    LOX_Naphthyl = 2.5
    LOX_LH2 = 5.0
    N2O4_UDMH = 2.6
    N2O4_UH25 = 2.5
    LOX_CH4 = 3.5
    AP_HTPB = 6.0

engine_control_coefficients = {
    1: 1.0,
    2: 1.0,
    3: 0.33,
    4: 0.5,
    5: 0.4,
    6: 0.6,
}

young_modulus = 71000000000
earth_radius  = 6371000

def read_propellant_density(propellant_type):
    """Возвращает плотность для типа топлива"""
    try:
        return getattr(Density, propellant_type).value
    except AttributeError:
        return 0

def read_mixture_ratio(type1, type2):
    """Возвращает соотношение O/F для пары окислитель-горючее"""
    pair_name = f"{type1}_{type2}".upper()
    try:
        return getattr(FuelRatio, pair_name).value
    except AttributeError:
        pass
    pair_name = f"{type2}_{type1}".upper()
    try:
        return getattr(FuelRatio, pair_name).value
    except AttributeError:
        return 0

def read_control_coefficient(engine_count):
    """Возвращает коэффициент управления для заданного числа камер"""
    return engine_control_coefficients.get(engine_count, 0.0)

def calculate_stiffness(diameter):
    return young_modulus * np.pi * cross_sectional_area(diameter)


def cross_sectional_area(diameter):
    return np.pi * (diameter**2) / 4


def calculate_static(mass_, shoulder):
    return 0.5 * mass_ * shoulder


def calculate_inertia(mass_, shoulder, length, diameter):
    return mass_ * (0.25 * shoulder**2 + 0.333 * length**2 + (diameter/2)**2)

def fuel_distribution_by_level(ld, init_mass, burn_ratio,
                               reserve_fraction=0.0):
    """
    Возвращает список масс микроэлементов ступени в момент,
    когда сгорела доля burn_ratio ∈ [0, 1] от СГОРАЕМОГО топлива.

    Топливо и окислитель удаляются сверху вниз
    (от меньших cumlength к большим), как вода из бака.
    Construction, Tail, Head остаются неизменными.

    reserve_fraction — доля топлива (0..1), которая НЕ участвует
    в выгорании и остаётся в баке. При burn_ratio = 1 в баке
    останется reserve_fraction от полного топлива (в нижних слоях).
    """
    distro = list(init_mass)
    if burn_ratio <= 0.0:
        return distro

    burnable_fraction = 1.0 - reserve_fraction
    if burnable_fraction <= 0.0:
        return distro

    for cls in ("Fuel", "Oxidizer"):
        total_cls = sum(m for m, c in zip(init_mass, ld.classes) if c == cls)
        if total_cls <= 0.0:
            continue

        # ИЗМЕНЕНО: умножаем на burnable_fraction — предел выгорания
        to_remove = total_cls * burnable_fraction * min(1.0, burn_ratio)
        if to_remove <= 0.0:
            continue

        # Индексы класса, сверху вниз
        idxs = sorted(
            (i for i, c in enumerate(ld.classes) if c == cls),
            key=lambda i: ld.cumlengths[i],
        )
        for i in idxs:
            if to_remove <= 1e-12:
                break
            if distro[i] > 0.0:
                take = min(distro[i], to_remove)
                distro[i] -= take
                to_remove -= take

    return distro
