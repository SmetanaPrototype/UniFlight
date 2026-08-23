

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