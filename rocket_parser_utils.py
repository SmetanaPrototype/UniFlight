import basis
import numpy as np

def read_propellant_density(propellant_type):
    """Возвращает плотность для типа топлива"""
    try:
        return getattr(basis.Density, propellant_type).value
    except AttributeError:
        return 0

def read_mixture_ratio(type1, type2):
    """Возвращает соотношение O/F для пары окислитель-горючее"""
    pair_name = f"{type1}_{type2}".upper()
    try:
        return getattr(basis.FuelRatio, pair_name).value
    except AttributeError:
        pass
    pair_name = f"{type2}_{type1}".upper()
    try:
        return getattr(basis.FuelRatio, pair_name).value
    except AttributeError:
        return 0

def read_control_coefficient(engine_count):
    """Возвращает коэффициент управления для заданного числа камер"""
    return basis.engine_control_coefficients.get(engine_count, 0.0)

#длина блока
def get_stage_length(current_stage, current_group):
    return sum(current_group.lengths[i] for i in range(len(current_group.stages)) if current_group.stages[i] == current_stage)
#длина участка/бака
def get_class_length(current_class, current_group):
    return sum(current_group.lengths[i] for i in range(len(current_group.classes)) if current_group.classes[i] == current_class)
def get_stageclass_length(current_stage, current_class, current_group):
    return sum(current_group.lengths[i] for i in range(len(current_group.classes)) if current_group.classes[i] == current_class and
                                                                                      current_group.stages[i] == current_stage)

def get_start_stageclass(current_stage, current_class, current_group):
    return min(current_group.cumlengths[i] for i in range(len(current_group.classes)) if current_group.classes[i] == current_class and 
                                                                                         current_group.stages[i] == current_stage)
def get_start_class(current_class, current_group):
    return min(current_group.cumlengths[i] for i in range(len(current_group.classes)) if current_group.stages[i] == current_class)

#Площади сечений и диаметры блоков
def get_stage_diameter(current_stage, current_group):
    indices = [i for i in range(len(current_group.stages)) if current_group.stages[i] == current_stage]
    return np.mean([current_group.diameters[i] for i in indices]) if indices else 0
def get_stage_area(current_stage, current_group):
    indices = [i for i in range(len(current_group.stages)) if current_group.stages[i] == current_stage]
    return np.mean([current_group.areas[i] for i in indices]) if indices else 0


def get_class_stage_mass(parser_obj, class_, stage_):
    res = 0
    eachtailmass = parser_obj.payload_mass * basis.tail_coefficient / (parser_obj.block_number + parser_obj.boosters_number - 1)
    if class_ == "Head" and stage_ == "Payload":
        res = parser_obj.payload_mass
    
    for b in range(parser_obj.block_number):
        if class_ == "Tail" and stage_ == parser_obj.stages[b]:
            res = eachtailmass
        elif class_ == "Oxidizer" and stage_ == parser_obj.stages[b]:
            res = parser_obj.mass_ox[b]
        elif class_ == "Fuel" and stage_ == parser_obj.stages[b]:
            res = parser_obj.mass_fu[b]
        elif class_ == "Construction" and stage_ == parser_obj.stages[b]:
            res = parser_obj.structural_mass[b] - eachtailmass
        
    return res

def stage_active_dt(self, index, time, dt):
    """Сколько секунд ступени index горит внутри интервала [time, time+dt]."""
    if index == 0 or (self.is_packet and index == 1):
        t_start = 0.0
    else:
        t_start = self.step_time[index - 1]
    t_end = self.step_time[index]

    lo = max(time, t_start)
    hi = min(time + dt, t_end)
    return max(0.0, hi - lo)

def fuel_distribution_by_level(self, ld, init_mass, burn_ratio,
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

def get_local_class(self, stage, local_li, start_):
    """Класс участка на расстоянии local_li от начала ступени stage."""
    acc = 0.0
    for L, st, cl in zip(start_.lengths, start_.stages, start_.classes):
        if st != stage:
            continue
        acc += L
        if local_li < acc:
            return cl
    return start_.classes[-1]
