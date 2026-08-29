import pandas as pd
import numpy as np
import basis
import json
import matplotlib.pyplot as plt
import attack
import types
import rocket_parser_utils
from collections import defaultdict

class Distributed_dataset:
    def __init__(self):
        self.numbers = []
        self.lengths = []
        self.cumlengths = []
        self.diameters = []
        self.classes = []
        self.stages = []
        self.stiffnesses = []
        self.areas = []
        self.volumes = []
        self.masses = []

    def __iter__(self):
        for attr_name, attr_value in self.__dict__.items():
            yield attr_name, attr_value

class Flight_dataset:
    def __init__(self):
         self.thrusts = []
         self.masses = []
         self.times = []
         self.statics = []
         self.inertions = []
         self.centers = []

class Coord_element:
    def __init__(self, start, length):
        self.start = float(start)
        self.length = float(length)
        self.end = self.start + self.length

class Rocket_parser:
    def __init__(self):
        rocket = basis.current_rocket
        json_filename = "rockets/" + rocket + "/constant.json"

        with open(json_filename, "r") as r_file:
            r_data = json.load(r_file)

        # common data
        self.name = r_data["name"]
        self.payload_mass      = r_data["payload_mass"]
        self.block_mass        = r_data["block_mass"]
        self.exhaust_velocity  = r_data["exhaust_velocity"]
        self.thrust            = r_data["thrust"]
        self.structural_values = r_data["structural_values"]
        self.fuel_type         = r_data["fuel_type"]
        self.oxidizer_type     = r_data["oxidizer_type"]
        self.boosters_number   = r_data["booster_number"]
        self.attack_coefs      = r_data["attack_coefs"]
        self.chumbers_number   = r_data["chumbers_number"]

        # Проверка размеров массивов
        is_equal = len(set(map(len, 
            [self.oxidizer_type, 
             self.fuel_type, 
             self.structural_values, 
             self.thrust, 
             self.exhaust_velocity, 
             self.block_mass]))) == 1
        if not is_equal:
            raise ValueError("Не хватает данных для всех ступеней!")

        self.csv_filename = "rockets/" + rocket + "/distributed.csv"


        # Handling
        self.block_number = len(self.block_mass)
        self.is_packet = self.boosters_number > 1
        self.full_mass = self.payload_mass + sum(self.block_mass[1:]) + self.boosters_number * self.block_mass[0]
        print(f"Payload mass: {self.payload_mass}")
        print(f"Block mass:   {self.block_mass}")
        print(f"Full mass:    {self.full_mass}")

        # Deep Handling
        self._calculate_stage_parameters()
        self.partial = self._distributed_handler()
        self.flight_data = self._calculate_flight_dynamics()

    def _calculate_stage_parameters(self):
        """Расчет параметров ступеней"""
        control_coefficient = rocket_parser_utils.read_control_coefficient(self.chumbers_number)

        propellant_mass = []
        self.mass_ox = []
        self.mass_fu = []
        delta_mass = []
        self.work_time = []
        self.structural_mass = []
        self.delta_mass_ox = []
        self.delta_mass_fu = []
        self.fuel_density = []
        self.oxidizer_density = []
        mixture_ratio = []

        for k in range(self.block_number):
            self.fuel_density.append(rocket_parser_utils.read_propellant_density(self.fuel_type[k]))
            self.oxidizer_density.append(rocket_parser_utils.read_propellant_density(self.oxidizer_type[k]))
            mixture_ratio.append(rocket_parser_utils.read_mixture_ratio(self.fuel_type[k], self.oxidizer_type[k]))
            propellant_mass.append(self.block_mass[k] * self.structural_values[k] / (self.structural_values[k] + 1))
            self.mass_ox.append(propellant_mass[k] * mixture_ratio[k] / (mixture_ratio[k] + 1))
            self.mass_fu.append(propellant_mass[k] / (mixture_ratio[k] + 1))
            delta_mass.append(self.thrust[k] / self.exhaust_velocity[k])
            self.work_time.append(propellant_mass[k] / delta_mass[k])
            self.structural_mass.append(self.block_mass[k] - propellant_mass[k])
            self.delta_mass_ox.append(delta_mass[k] * mixture_ratio[k] / (mixture_ratio[k] + 1))
            self.delta_mass_fu.append(delta_mass[k] / (mixture_ratio[k] + 1))

        self.full_time = sum(self.work_time)
        if self.is_packet:
            self.full_time -= self.work_time[0]


    def _distributed_handler(self):
        """Расчет распределенных по длине параметров"""
        start_     = Distributed_dataset()
        # result_    = Distributed_dataset()

        df = pd.read_csv(self.csv_filename)
        start_.lengths = pd.to_numeric(df["L"], errors='coerce').tolist()
        start_.diameters = pd.to_numeric(df["D"], errors='coerce').tolist()
        start_.classes = df["Class"]
        start_.stages = df["Stage"]
        start_.cumlengths = np.cumsum(start_.lengths) #кумулятивная длина
        start_.numbers = list(range(len(start_.lengths)))

        # Проверка обработки исходных данных
        if len(start_.lengths) != len(set(start_.lengths)):
            raise ValueError("Найдены одинаковые по длине участки")

        if self.block_number not in [2,3,4,5]:
            raise ValueError("Current block number is not supported")

        # Распределение масс по ступеням
        self.stages = []
        if self.block_number == 2:
            self.stages = ["First", "Second", "Payload"]
        elif self.block_number == 3:
            self.stages = ["First", "Second", "Third", "Payload"]
        elif self.block_number == 4:
            self.stages = ["First", "Second", "Third", "Fourth", "Payload"]
        elif self.block_number == 5:
            self.stages = ["First", "Second", "Third", "Fourth", "Fifth", "Payload"]
        
        self.classes = ["Tail", "Fuel", "Oxidizer", "Construction", "Head"]

        def get_class_stage_mass(class_, stage_):
            res = 0
            eachtailmass = self.payload_mass * basis.tail_coefficient / (self.block_number + self.boosters_number - 1)
            if class_ == "Head" and stage_ == "Payload":
                res = self.payload_mass
            
            for b in range(self.block_number):
                if class_ == "Tail" and stage_ == self.stages[b]:
                    res = eachtailmass
                elif class_ == "Oxidizer" and stage_ == self.stages[b]:
                    res = self.mass_ox[b]
                elif class_ == "Fuel" and stage_ == self.stages[b]:
                    res = self.mass_fu[b]
                elif class_ == "Construction" and stage_ == self.stages[b]:
                    res = self.structural_mass[b] - eachtailmass
                
            return res

        stages_lengths = []
        class_stage_masses = defaultdict(dict)
        for s in self.stages: 
            stages_lengths.append(rocket_parser_utils.get_stage_length(s, start_))
            for c in self.classes:
                class_stage_masses[c][s] = get_class_stage_mass(c,s)
        
        shared_mass = sum(sum(inner_dict.values()) for inner_dict in class_stage_masses.values())
        booster_mass = sum(class_stage_masses[c].get("First", 0) for c in self.classes)
        shared_mass += booster_mass * (self.boosters_number - 1)
    
        # Проверка распределения масс по ступеням
        if abs(shared_mass - self.full_mass) > 0.01:
            raise ValueError("Неправильное распределение масс по ступеням!")

        # Создание наборов для каждого блока
        dis_partial_ = [Distributed_dataset() for _ in range(self.block_number)]
        # Разбиение на малые элементы

        ful_li = 0
        for b in range(len(dis_partial_)):
            li = 0
            num = 0

            while li < stages_lengths[b]:
                if li >= basis.lenstep / 2:
                    dis_partial_[b].lengths.append(basis.lenstep)
                dis_partial_[b].cumlengths.append(round(li, int(basis.lenstep)))
                dis_partial_[b].numbers.append(num)
                dis_partial_[b].stages.append(self.stages[b])
                dis_partial_[b].classes.append(rocket_parser_utils.get_cumlength_class(ful_li, start_))
                local_length = rocket_parser_utils.get_stageclass_length(dis_partial_[b].stages[-1], dis_partial_[b].classes[-1], start_)
                dis_partial_[b].masses.append(local_length > 0 and get_class_stage_mass(dis_partial_[b].classes[-1], dis_partial_[b].stages[-1]) * basis.lenstep / local_length or 0)
                dis_partial_[b].diameters.append(rocket_parser_utils.get_stage_diameter(self.stages[b], start_))
                dis_partial_[b].stiffnesses.append(basis.calculate_stiffness(dis_partial_[b].diameters[-1]))
                dis_partial_[b].areas.append(basis.cross_sectional_area(dis_partial_[b].diameters[-1]))
                dis_partial_[b].volumes.append(dis_partial_[b].areas[-1] * basis.lenstep)

                li += basis.lenstep
                ful_li += basis.lenstep
                num +=1
            
        # Проверка разбиения на малые элементы
        mass_with_micro = sum(sum(block.masses) for block in dis_partial_) + (self.boosters_number - 1) * sum(dis_partial_[0].masses)
        if abs(mass_with_micro - self.full_mass) < 0.01:
            raise ValueError("Неправильное разбиение на малые элементы")

        return dis_partial_

    def _calculate_flight_dynamics(self):
        """Расчет динамических параметров полета"""
        # max_area     = max(max(block.areas) for block in self.partial)
        # max_diameter = max(max(block.diameters) for block in self.partial)
        # print(max_area)

        

        delta_level_ox = []
        delta_level_fu = []
        for k in range(len(self.partial)):
            max_area = max(self.partial[k].areas)
            delta_level_ox.append(
                self.delta_mass_ox[k] / self.oxidizer_density[k] / max_area
            )
            delta_level_fu.append(
                self.delta_mass_fu[k] / self.fuel_density[k] / max_area
            )

        # Здесь ступени, а не блоки
        fly_partial_ = [Flight_dataset() for _ in range(self.block_number)]

        time = 0
        time_stages = np.cumsum(self.work_time)
        if self.is_packet:
            time_stages[1:] -= time_stages[0]

        while time < self.full_mass:
            for index, s in enumerate(fly_partial_):
                s.times.append(time)
            
            # Определяем текущий этап
            current_stage = None
            for i in range(len(time_stages)):
                if i == 0 and time < time_stages[0]:
                    current_stage = 0
                    break
                elif i > 0 and time_stages[i-1] <= time < time_stages[i]:
                    current_stage = i
                    break
            
            # Применяем thrust для всех ступеней
            if current_stage == 0:
                # Первый этап - специальная логика
                for s in fly_partial_:
                    thrust_ = self.thrust[0] * self.boosters_number
                    if self.is_packet:
                        thrust_ += self.thrust[1]
                    s.thrusts.append(thrust_)
            elif current_stage is not None:
                # Остальные этапы
                for i, s in enumerate(fly_partial_):
                    if i < current_stage:
                        s.thrusts.append(0)
                    else:
                        s.thrusts.append(self.thrust[current_stage])
            else:
                # После всех этапов
                for s in fly_partial_:
                    s.thrusts.append(0)
            
            time += basis.timestep
        
        return fly_partial_

        # for s in fly_partial_:
            
        # time = 0
        # current_mass = self.full_mass
        # stage_dropped = []
        # stage_separation_times = []
        # t_sep = 0
        # for i in range((self.block_number)):
        #     stage_dropped.append(False)
        #     t_sep += work_time[i]
        #     stage_separation_times.append(t_sep)

        # fdata.masses.append(current_mass)
        # fdata.times.append(time)
        # fdata.thrusts.append(thrust[0])

        # static_moment = basis.calculate_static(
        #     self.full_mass, full_length
        # ) * 1.3 # TODO: Fox error
        # inertia_moment = basis.calculate_inertia(
        #     self.full_mass, full_length, full_length, max_diameter
        # )
        # fdata.statics.append(static_moment)
        # fdata.inertions.append(inertia_moment)
        # fdata.centers.append(full_length - static_moment / current_mass) # TODO: Fix jump error

        # while time < stage_separation_times[-1]:
        #     fdata.statics.append(static_moment)
        #     fdata.inertions.append(inertia_moment)

        #     current_stage = None
        #     for i in range(self.block_number):
        #         if time < stage_separation_times[i]:
        #             current_stage = i
        #             break

        #     if current_stage is not None:
        #         current_mass -= delta_mass[current_stage] * basis.timestep
        #         lower_point = fuel_coordinates[current_stage].end
        #         upper_point = fuel_coordinates[current_stage].end - delta_level_fu[current_stage]
        #         static_moment -= (
        #             basis.calculate_static(delta_mass_fu[current_stage], lower_point + upper_point) * basis.timestep
        #         )
        #         lower_point = oxidyzer_coordinates[current_stage].end
        #         upper_point = oxidyzer_coordinates[current_stage].end - delta_level_ox[current_stage]
        #         static_moment -= (
        #             basis.calculate_static(delta_mass_ox[current_stage], lower_point + upper_point) * basis.timestep
        #         )
        #         lower_point = fuel_coordinates[current_stage].end
        #         upper_point = fuel_coordinates[current_stage].end - delta_level_fu[current_stage]
        #         inertia_moment -= (
        #             basis.calculate_inertia(
        #                 delta_mass_fu[current_stage],
        #                 lower_point + upper_point,
        #                 lower_point - upper_point,
        #                 max_diameter,
        #             )
        #             * basis.timestep
        #         )
        #         lower_point = oxidyzer_coordinates[current_stage].end
        #         upper_point = oxidyzer_coordinates[current_stage].end - delta_level_ox[current_stage]
        #         inertia_moment -= (
        #             basis.calculate_inertia(
        #                 delta_mass_ox[current_stage],
        #                 lower_point + upper_point,
        #                 lower_point - upper_point,
        #                 max_diameter,
        #             )
        #             * basis.timestep
        #         )

        #         thrust = thrust[current_stage]

        #     for i in range(self.block_number):
        #         if not stage_dropped[i] and time >= stage_separation_times[i]:
        #             current_mass -= structural_mass[i]
        #             static_moment -= basis.calculate_static(
        #                 structural_mass[i],
        #                 structural_coordinates[i].end
        #                 + structural_coordinates[i].start,
        #             )
        #             inertia_moment -= basis.calculate_inertia(
        #                 structural_mass[i],
        #                 structural_coordinates[i].end
        #                 + structural_coordinates[i].start,
        #                 structural_coordinates[i].length,
        #                 max_diameter,
        #             )
        #             stage_dropped[i] = True
        #             print(
        #                 f"Отделена {i+1}-я ступень в t={time} с, расчетное время: {stage_separation_times[i]} c, текущая масса: {current_mass}"
        #             )
        #     cent = static_moment / current_mass


        #     fdata.masses.append(current_mass)
        #     fdata.times.append(time)
        #     fdata.thrusts.append(thrust)
        #     fdata.centers.append(cent)
        #     time += basis.timestep
        # return fdata

                # этот скрипт писался для всей ракеты, теперь же надо адаптировать его под каждый блок
        # for b in range(self.block_number):
        #         found = False
        #         for i in range(len(dis_partial_[b].cumlengths) - 1):
        #             if (
        #                 li <= start_.cumlengths[i + 1]
        #                 and li > start_.cumlengths[i]
        #             ):
        #                 found = True
        #                 result_.classes.append(start_.classes[i + 1])
        #                 result_.stages.append(start_.stages[i + 1])
        #                 delta_diameter = (
        #                     start_.diameters[i + 1] - start_.diameters[i]
        #                 )
        #                 delta_length = (
        #                     start_.cumlengths[i + 1] - start_.cumlengths[i]
        #                 )
        #                 offset = li - start_.cumlengths[i]
        #                 if delta_length > 0:
        #                     current_diameter = (
        #                         start_.diameters[i]
        #                         + (delta_diameter / delta_length) * offset
        #                     )
        #                 else:
        #                     current_diameter = start_.diameters[i]
        #                 result_.diameters.append(current_diameter)
        #                 result_.stiffnesses.append(basis.calculate_stiffness(current_diameter))
        #                 result_.areas.append(basis.cross_sectional_area(current_diameter))
        #                 result_.volumes.append(result_.areas[-1] * basis.lenstep)
                  # этот скрипт писался для всей ракеты, теперь же надо адаптировать его под каждый блок






        # Получение данных для всей ракеты

        # result_.lengths.append(0)
        # result_.diameters.append(0)
        # result_.stiffnesses.append(0)
        # result_.areas.append(0)
        # result_.volumes.append(0)

        # result_.classes.append("Head")
        # result_.stages.append("Payload")

        # li = 0
        # num = 0
        # # надо пересчитывать для каждого блока с учетом разных типов топлива
        # # print(start_.lengths)
        # # Разбиение на малые элементы
        # while li < sum(start_.lengths):
        #     if li >= basis.lenstep / 2:
        #         result_.lengths.append(basis.lenstep)
        #     result_.cumlengths.append(round(li, 1))
        #     result_.numbers.append(num)
        #     found = False
        #     for i in range(len(start_.cumlengths) - 1):
        #         if (
        #             li <= start_.cumlengths[i + 1]
        #             and li > start_.cumlengths[i]
        #         ):
        #             found = True
        #             result_.classes.append(start_.classes[i + 1])
        #             result_.stages.append(start_.stages[i + 1])
        #             delta_diameter = (
        #                 start_.diameters[i + 1] - start_.diameters[i]
        #             )
        #             delta_length = (
        #                 start_.cumlengths[i + 1] - start_.cumlengths[i]
        #             )

        #             offset = li - start_.cumlengths[i]

        #             if delta_length > 0:
        #                 current_diameter = (
        #                     start_.diameters[i]
        #                     + (delta_diameter / delta_length) * offset
        #                 )
        #             else:
        #                 current_diameter = start_.diameters[i]

        #             result_.diameters.append(current_diameter)
        #             result_.stiffnesses.append(basis.calculate_stiffness(current_diameter))
        #             result_.areas.append(basis.cross_sectional_area(current_diameter))
        #             result_.volumes.append(result_.areas[-1] * basis.lenstep)

        #     li += basis.lenstep
        #     num += 1
        # # print(result_.diameters)
        # fuel_coordinates = []
        # oxidyzer_coordinates = []
        # structural_coordinates = []

        # # Поиск уровней
        # for s in self.stages:
        #     if rocket_parser_utils.get_stage_length(s, result_)>0 and s!="Payload":

        #         fuel_coordinates.append(
        #             Coord_element(
        #                 rocket_parser_utils.get_start_stageclass(s, "Fuel",  result_), 
        #                 rocket_parser_utils.get_stageclass_length(s, "Fuel", result_)
        #             )
        #         )

        #         oxidyzer_coordinates.append(
        #             Coord_element(
        #                 rocket_parser_utils.get_start_stageclass(s, "Oxidizer",  result_),
        #                 rocket_parser_utils.get_stageclass_length(s, "Oxidizer", result_),
        #             )
        #         )

        #         structural_coordinates.append(
        #             Coord_element(
        #                 rocket_parser_utils.get_start_class(s,  result_), 
        #                 rocket_parser_utils.get_class_length(s, result_))
        #         )

        # def get_class_density(current_class, current_group):
        #     if rocket_parser_utils.get_class_length(current_class, current_group)>0:
        #         return (class_masses[current_class] / sum(current_group.volumes[i]
        #             for i in range(len(current_group.classes)) 
        #             if current_group.classes[i] == current_class))
        #     else:
        #         return 0

        # class_densities = {}
        # # class_densities.update({"Construction": class_masses["Construction"] / sum(result_.volumes)})
        # # for cw in class_masses.keys():
        # #     if cw != "Construction":
        # #         class_densities.update({cw: get_class_density(cw,  result_)})

        # for j in range(len(result_.volumes)):
        #     # m = class_densities["Construction"] * result_.volumes[j]
        #     for cw in class_masses.keys():
        #         if result_.classes[j] == cw:
        #             m = get_class_density(cw,  result_) * result_.volumes[j]
        #     result_.masses.append(m)

        # print("Destr mass:", sum(result_.masses))
        # full_length = result_.cumlengths[-1]


        # # СОЗДАНИЕ НАБОРОВ ДАННЫХ ДЛЯ КАЖДОГО ТИПА STAGES ОТДЕЛЬНО
        # stage_datasets = {}
        # for stage_name in self.stages:
        #     stage_dataset = Distributed_dataset()
        #     # Инициализация пустых списков
        #     stage_dataset.lengths = []
        #     stage_dataset.diameters = []
        #     stage_dataset.stiffnesses = []
        #     stage_dataset.areas = []
        #     stage_dataset.volumes = []
        #     stage_dataset.classes = []
        #     stage_dataset.stages = []
        #     stage_dataset.numbers = []
        #     stage_dataset.cumlengths = []
        #     stage_dataset.masses = []
            
        #     # Заполнение данными только для текущего stage
        #     for idx in range(len(result_.classes)):
        #         if result_.stages[idx] == stage_name:
        #             stage_dataset.lengths.append(result_.lengths[idx])
        #             stage_dataset.diameters.append(result_.diameters[idx])
        #             stage_dataset.stiffnesses.append(result_.stiffnesses[idx])
        #             stage_dataset.areas.append(result_.areas[idx])
        #             stage_dataset.volumes.append(result_.volumes[idx])
        #             stage_dataset.classes.append(result_.classes[idx])
        #             stage_dataset.stages.append(result_.stages[idx])
        #             stage_dataset.numbers.append(result_.numbers[idx])
        #             stage_dataset.cumlengths.append(result_.cumlengths[idx])
        #             stage_dataset.masses.append(result_.masses[idx])
            
        #     # Сохраняем в словарь
        #     stage_datasets[stage_name] = stage_dataset
        # print(sum(stage_datasets["First"].masses))
        # print(sum(stage_datasets["Second"].masses))
        # print(sum(stage_datasets["Third"].masses))
        # print(sum(stage_datasets["Payload"].masses))
        
        # # if self.is_packet:
        # #     packet_result_ = Distributed_dataset()
        # #     for idx in range(len(result_.classes)):
        # #         if result_.stages[idx] != "First":
        # #             packet_result_.lengths.append(result_.lengths[idx])
        # #             packet_result_.numbers.append(result_.numbers[idx])
        # #             packet_result_.cumlengths.append(result_.cumlengths[idx])
        # #             packet_result_.masses.append(result_.masses[idx])
                
        #     #packet_result_.masses + # надо с последними элементами (кол-во равно кол-ву с First) сложить массы из First, умноженные на self.boosters_number

        # #     # я хочу обратную логику, отдельно должны вычисляться данные для stages
        # #     # если is packet, то first, умноженные на self.booster_number добавляется в конец 
        # # сейчас логика неправильная, из full_mass вычитаются остатки и размазываются по конструкции, что неверно
        # # массы должны складываться, площадь и диамеры должна быть средние D + d/2 + d/2 (для бустеров)
        # # а моменты инерции считаться по
        # # # Возвращаем основной результат и словарь с наборами данных по stages
        # return result_, stage_datasets

    def changed_mass(self, time_):
        """Расчет распределенной массы в заданный момент времени"""
        mass_t = distr_data.masses.copy()
        time_remaining = time_

        for stage in range(block_number):
            stage_time = work_time[stage]
            if time_remaining <= 0:
                break

            if time_remaining >= stage_time:
                # Ступень полностью отработала
                for i in range(len(mass_t)):
                    if distr_data.stages[i] in ["First", "Second", "Third"]:
                        if distr_data.classes[i] in ["Fuel", "Oxidizer"]:
                            # Оставляем только массу конструкции бака
                            mass_t[i] = distr_data.volumes[i] * class_densities["Construction"]
                        # Сегменты типа 'Construction', 'Tail' уже имеют правильную массу
                time_remaining -= stage_time

            else:

                burn_ratio = time_remaining / stage_time

                fuel_segments = []
                oxid_segments = []

                for i in range(len(mass_t)):
                    if distr_data.stages[i] in ["First", "Second", "Third"]:
                        if distr_data.classes[i] == "Fuel":
                            fuel_segments.append((i, distr_data.cumlengths[i]))
                        elif distr_data.classes[i] == "Oxidizer":
                            oxid_segments.append((i, distr_data.cumlengths[i]))

                fuel_segments.sort(key=lambda x: -x[1], reverse=True)
                oxid_segments.sort(key=lambda x: -x[1], reverse=True)

                fuel_mass_to_remove = mass_fu[stage] * burn_ratio
                oxid_mass_to_remove = mass_ox[stage] * burn_ratio

                for idx, _ in fuel_segments:
                    if fuel_mass_to_remove <= 0:
                        break
                    construction_mass = distr_data.volumes[idx] * class_densities["Construction"]
                    current_fuel_mass = mass_t[idx] - construction_mass
                    if current_fuel_mass > 0:
                        remove = min(current_fuel_mass, fuel_mass_to_remove)
                        mass_t[idx] -= remove
                        fuel_mass_to_remove -= remove

                for idx, _ in oxid_segments:
                    if oxid_mass_to_remove <= 0:
                        break
                    construction_mass = distr_data.volumes[idx] * class_densities["Construction"]
                    current_oxid_mass = mass_t[idx] - construction_mass
                    if current_oxid_mass > 0:
                        remove = min(current_oxid_mass, oxid_mass_to_remove)
                        mass_t[idx] -= remove
                        oxid_mass_to_remove -= remove

                time_remaining = 0
        return mass_t


    def attack_func(self, vel, time):
        """Получение угла атаки по коэффициентам из парсера"""
        alpha_obj = attack.alpha(
            self.attack_coefs[0], self.attack_coefs[1], self.work_time[0], False
        )
        alpha_val = alpha_obj.calculate_alpha(vel, time)
        return max(-30, min(30, alpha_val))

    def get_work_time(self):
        return self.work_time

    def get_data(self, is_booster):
        return self.distr_data

    def get_block_number(self):
        return self.block_number

    def get_delta_mass_fu(self):
        return self.delta_mass_fu

    def get_delta_mass_ox(self):
        return self.delta_mass_ox

    def get_coordinates_fu(self):
        return self.fuel_coordinates

    def get_coordinates_ox(self):
        return self.oxidyzer_coordinates

    def get_full_length(self):
        return self.full_length

    def get_changed_mass(self, time_):
        return self.changed_mass(time_)

rp = Rocket_parser()

plt.plot(rp.flight_data[0].times, rp.flight_data[0].thrusts, label="Тяга, Н", color="purple")
plt.plot(rp.flight_data[1].times, rp.flight_data[1].thrusts, label="Тяга, Н", color="purple")
plt.plot(rp.flight_data[2].times, rp.flight_data[2].thrusts, label="Тяга, Н", color="purple")
plt.xlabel("Время, с")
plt.ylabel("Тяга, Н")
plt.title("Тяга, Н")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()