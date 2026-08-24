import basis

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