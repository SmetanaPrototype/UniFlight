import pandas as pd
import numpy as np
import constants
import json
import matplotlib.pyplot as plt


def read_propellant_density(propellant_type):
    density_map = {
        "LOX" : constants.density.LOX.value,
        "RP-1": constants.density.RP_1.value,
        "RG_1": constants.density.RG_1.value,
        "UDMH": constants.density.UDMH.value,
        "N2O4": constants.density.N2O4.value,
        "HTPB": constants.density.HTPB.value,
        "AP"  : constants.density.AP.value,
        "UH25": constants.density.UH25.value,
        "CH4" : constants.density.CH4.value,
    }
    return density_map.get(propellant_type, 0)


class coord_element:
    def __init__(self, start, length):
        self.start = float(start)
        self.length = float(length)
        self.end = self.start + self.length


class rocket_parser:
    def __init__(self):
        rocket = constants.current_rocket
        json_filename = "rocket_lib/" + rocket + "/constant.json"
        csv_filename = "rocket_lib/" + rocket + "/distributed.csv"

        with open(json_filename, "r") as r_file:
            r_data = json.load(r_file)

        # initial data
        self.name = r_data["name"]
        self.block_number = r_data["block_number"]
        self.payload_mass = r_data["payload_mass"]
        self.block_mass = r_data["block_mass"]
        self.components_ratio = r_data["components_ratio"]
        self.thrust_ratio = r_data["thrust_ratio"]
        self.exhaust_velocity = r_data["exhaust_velocity"]
        self.thrust = r_data["thrust"]
        self.structural_values = r_data["structural_values"]
        self.fuel = r_data["fuel"]
        self.oxidizer = r_data["oxidizer"]
        self.attack_coefs = r_data["attack_coefs"]

        # handled data
        self.full_mass = self.payload_mass + sum(self.block_mass)
        print(f"Payload mass: {self.payload_mass}")
        print(f"Block mass: {self.block_mass}")
        print(f"Full mass: {self.full_mass}")
        self.fuel_density = read_propellant_density(self.fuel)
        self.oxidizer_density = read_propellant_density(self.oxidizer)

        self._calculate_stage_parameters()
        self.distr_data = self._distributed_handler(csv_filename)
        self._initialize_distributed_arrays()
        self._calculate_flight_dynamics()

    def _calculate_stage_parameters(self):
        """Расчет параметров ступеней"""
        self.propellant_mass = []
        self.delta_mass = []
        self.stage_mass = []
        self.structural_mass = []
        self.delta_mass_ox = []
        self.delta_mass_fu = []
        self.work_time = []
        self.mass_ox = []
        self.mass_fu = []

        self.Length_parts = []

        for k in range(self.block_number):
            self.propellant_mass.append(
                self.block_mass[k]
                * self.structural_values[k]
                / (self.structural_values[k] + 1)
            )
            self.mass_ox.append(
                self.propellant_mass[k]
                * self.components_ratio
                / (self.components_ratio + 1)
            )
            self.mass_fu.append(self.propellant_mass[k] / (self.components_ratio + 1))
            delta_mass = self.thrust[k] / self.exhaust_velocity[k]
            self.delta_mass.append(delta_mass)
            self.work_time.append(self.propellant_mass[k] / self.delta_mass[k])
            self.structural_mass.append(self.block_mass[k] - self.propellant_mass[k])
            self.delta_mass_ox.append(
                delta_mass * self.components_ratio / (self.components_ratio + 1)
            )
            self.delta_mass_fu.append(delta_mass / (self.components_ratio + 1))

        self.full_time = sum(self.work_time)
        m = self.propellant_mass[1]
        t = 0
        while m > 0:
            m -= self.delta_mass[1] * constants.timestep
            t += constants.timestep

    def _distributed_handler(self, filename):
        df = pd.read_csv(filename)
        Length_start_vector = df["L"]
        Diameter_start_vector = df["D"]
        Class_start_vector = df["Class"]
        Stage_start_vector = df["Stage"]

        self.Length_parts.append(
            sum(
                Length_start_vector[i]
                for i in range(len(Stage_start_vector))
                if Stage_start_vector[i] == "Payload"
            )
        )
        self.Length_parts.append(
            sum(
                Length_start_vector[i]
                for i in range(len(Stage_start_vector))
                if Stage_start_vector[i] == "First"
            )
        )
        self.Length_parts.append(
            sum(
                Length_start_vector[i]
                for i in range(len(Stage_start_vector))
                if Stage_start_vector[i] == "Second"
            )
        )

        # Classification handling
        class_counts = df["Class"].value_counts()
        class_counts_dict = class_counts.to_dict()
        tail_count = class_counts_dict.get("Tail", 0)
        head_count = class_counts_dict.get("Head", 0)

        # self.Length_parts.append(df[df['Stage'] == 'Payload']['L'].sum())
        # self.Length_parts.append(df[df['Stage'] == 'First']['L'].sum())
        # self.Length_parts.append(df[df['Stage'] == 'Second']['L'].sum())

        # Masses
        m_payload = self.payload_mass
        m_fuel = sum(self.mass_fu)
        m_oxidyzer = sum(self.mass_ox)
        m_full = self.full_mass
        m_engine = m_payload / 3
        self.max_diameter = max(Diameter_start_vector)
        m_construction = m_full - m_oxidyzer - m_fuel - m_payload - m_engine

        Sum_Length_start_vector = []
        cumlength = 0
        for le in Length_start_vector:
            cumlength += le
            Sum_Length_start_vector.append(cumlength)

        Length_final_vector = []
        Diameter_final_vector = []
        Class_final_vector = []
        Stage_final_vector = []
        Stiffness_final_vetor = []
        Area_final_vector = []
        Volume_final_vector = []
        Mass_final_vector = []

        Length_final_vector.append(0)
        Diameter_final_vector.append(0)
        Class_final_vector.append("Head")
        Stage_final_vector.append("Payload")
        Stiffness_final_vetor.append(0)
        Area_final_vector.append(0)
        Volume_final_vector.append(0)

        self.numeric = []
        raised_length = []

        li = 0
        num = 0

        while li < sum(Length_start_vector):
            if li >= constants.lenstep / 2:
                Length_final_vector.append(constants.lenstep)
            raised_length.append(round(li, 1))
            self.numeric.append(num)

            for i in range(len(Sum_Length_start_vector) - 1):
                if (
                    li < Sum_Length_start_vector[i + 1]
                    and li > Sum_Length_start_vector[i]
                ):
                    Class_final_vector.append(Class_start_vector[i + 1])
                    Stage_final_vector.append(Stage_start_vector[i + 1])
                    delta_diameter = (
                        Diameter_start_vector[i + 1] - Diameter_start_vector[i]
                    )
                    delta_length = (
                        Sum_Length_start_vector[i + 1] - Sum_Length_start_vector[i]
                    )

                    offset = li - Sum_Length_start_vector[i]

                    if delta_length > 0:
                        current_diameter = (
                            Diameter_start_vector[i]
                            + (delta_diameter / delta_length) * offset
                        )
                    else:
                        current_diameter = Diameter_start_vector[i]

                    Diameter_final_vector.append(current_diameter)
                    Stiffness_final_vetor.append(
                        constants.calculate_stiffness(current_diameter)
                    )
                    Area_final_vector.append(
                        constants.cross_sectional_area(current_diameter)
                    )
                    Volume_final_vector.append(
                        Area_final_vector[-1] * constants.lenstep
                    )

            li += constants.lenstep
            num += 1

        def mass_per_class(mass_, classname_):
            return mass_ / sum(
                Volume_final_vector[i]
                for i in range(len(Class_final_vector))
                if Class_final_vector[i] == classname_
            )

        def tanklen_per_class(type_, classname_):
            return sum(
                Length_final_vector[i]
                for i in range(len(Class_final_vector))
                if Class_final_vector[i] == type_
                and Stage_final_vector[i] == classname_
            )

        def tankpoint_per_class(type_, classname_):
            return min(
                raised_length[i]
                for i in range(len(Class_final_vector))
                if Class_final_vector[i] == type_
                and Stage_final_vector[i] == classname_
            )

        def len_per_class(classname_):
            return sum(
                Length_final_vector[i]
                for i in range(len(Class_final_vector))
                if Stage_final_vector[i] == classname_
            )

        def point_per_class(classname_):
            return min(
                raised_length[i]
                for i in range(len(Class_final_vector))
                if Stage_final_vector[i] == classname_
            )

        self.m_construction_per = m_construction / sum(Volume_final_vector)
        self.m_payload_per = mass_per_class(m_payload, "Head")
        self.m_oxidizer_per = mass_per_class(m_oxidyzer, "Oxidizer")
        self.m_fuel_per = mass_per_class(m_fuel, "Fuel")
        self.m_engine_per = mass_per_class(m_engine, "Tail")

        self.fuel_coordinates = []
        self.oxidyzer_coordinates = []
        self.structural_coordinates = []

        self.fuel_coordinates.append(
            coord_element(
                tankpoint_per_class("Fuel", "First"), tanklen_per_class("Fuel", "First")
            )
        )
        self.fuel_coordinates.append(
            coord_element(
                tankpoint_per_class("Fuel", "Second"),
                tanklen_per_class("Fuel", "Second"),
            )
        )

        self.oxidyzer_coordinates.append(
            coord_element(
                tankpoint_per_class("Oxidizer", "First"),
                tanklen_per_class("Oxidizer", "First"),
            )
        )
        self.oxidyzer_coordinates.append(
            coord_element(
                tankpoint_per_class("Oxidizer", "Second"),
                tanklen_per_class("Oxidizer", "Second"),
            )
        )

        self.structural_coordinates.append(
            coord_element(point_per_class("First"), len_per_class("First"))
        )
        self.structural_coordinates.append(
            coord_element(point_per_class("Second"), len_per_class("Second"))
        )

        for j in range(len(Volume_final_vector)):
            m = self.m_construction_per * Volume_final_vector[j]
            if Class_final_vector[j] == "Head":
                m += self.m_payload_per * Volume_final_vector[j]
            elif Class_final_vector[j] == "Oxidizer":
                m += self.m_oxidizer_per * Volume_final_vector[j]
            elif Class_final_vector[j] == "Fuel":
                m += self.m_fuel_per * Volume_final_vector[j]
            elif Class_final_vector[j] == "Tail":
                m += self.m_engine_per * Volume_final_vector[j]
            Mass_final_vector.append(m)

        # zero start

        print("Destr mass:", sum(Mass_final_vector))

        new_data = {
            "numeric": self.numeric,
            "step": Length_final_vector,
            "length": raised_length,
            "diameter": Diameter_final_vector,
            "area": Area_final_vector,
            "volume": Volume_final_vector,
            "mass": Mass_final_vector,
            "stiffness": Stiffness_final_vetor,
            "class": Class_final_vector,
            "stage": Stage_final_vector,
        }
        self.rocket_length = raised_length[-1]

        # print(len(numeric))
        # print(len(Length_final_vector))
        # print(len(raised_length))
        # print(len(Diameter_final_vector))
        # print(len(Area_final_vector))
        # print(len(Mass_final_vector))
        # print(len(Stiffness_final_vetor))
        # print(len(Class_final_vector))
        # print(len(Stage_final_vector))

        df_result = pd.DataFrame(new_data)
        df_result.to_csv("output/rocket_data.csv", index=False, encoding="utf-8")

        return new_data

    def _initialize_distributed_arrays(self):
        """Distributed parameters initialization"""
        self.asc_length = self.distr_data["length"]
        self.diameters = self.distr_data["diameter"]
        self.masses = self.distr_data["mass"]
        self.stiffnesses = self.distr_data["stiffness"]
        self.areas = self.distr_data["area"]
        self.volumes = self.distr_data["volume"]
        self.classes = self.distr_data["class"]
        self.stages = self.distr_data["stage"]
        self.steps = self.distr_data["step"]

    def changed_mass(self, time_):
        """Расчет распределенной массы в заданный момент времени"""
        mass_t = self.masses.copy()
        time_remaining = time_

        for stage in range(self.block_number):
            stage_time = self.work_time[stage]
            if time_remaining <= 0:
                break

            if time_remaining >= stage_time:
                # Ступень полностью отработала
                for i in range(len(mass_t)):
                    stage_name = ["Payload", "First", "Second"][stage + 1]
                    if self.stages[i] == stage_name:
                        if self.classes[i] in ["Fuel", "Oxidizer"]:
                            # Оставляем только массу конструкции бака
                            mass_t[i] = self.volumes[i] * self.m_construction_per
                        # Сегменты типа 'Construction', 'Tail' уже имеют правильную массу
                time_remaining -= stage_time

            else:

                burn_ratio = time_remaining / stage_time

                fuel_segments = []
                oxid_segments = []

                for i in range(len(mass_t)):
                    stage_name = ["Payload", "First", "Second"][stage + 1]
                    if self.stages[i] == stage_name:
                        if self.classes[i] == "Fuel":
                            fuel_segments.append((i, self.asc_length[i]))
                        elif self.classes[i] == "Oxidizer":
                            oxid_segments.append((i, self.asc_length[i]))

                fuel_segments.sort(key=lambda x: -x[1], reverse=True)
                oxid_segments.sort(key=lambda x: -x[1], reverse=True)

                fuel_mass_to_remove = self.mass_fu[stage] * burn_ratio
                oxid_mass_to_remove = self.mass_ox[stage] * burn_ratio

                for idx, _ in fuel_segments:
                    if fuel_mass_to_remove <= 0:
                        break
                    construction_mass = self.volumes[idx] * self.m_construction_per
                    current_fuel_mass = mass_t[idx] - construction_mass
                    if current_fuel_mass > 0:
                        remove = min(current_fuel_mass, fuel_mass_to_remove)
                        mass_t[idx] -= remove
                        fuel_mass_to_remove -= remove

                for idx, _ in oxid_segments:
                    if oxid_mass_to_remove <= 0:
                        break
                    construction_mass = self.volumes[idx] * self.m_construction_per
                    current_oxid_mass = mass_t[idx] - construction_mass
                    if current_oxid_mass > 0:
                        remove = min(current_oxid_mass, oxid_mass_to_remove)
                        mass_t[idx] -= remove
                        oxid_mass_to_remove -= remove

                time_remaining = 0

        return mass_t

    def effective_mass(self, time_, mode_index):

        f1lev = self.fuel_coordinates[0].length
        f2lev = self.fuel_coordinates[1].length
        o1lev = self.oxidyzer_coordinates[0].length
        o2lev = self.oxidyzer_coordinates[1].length

        if   mode_index == 0: E = 1.841
        elif mode_index == 1: E = 3.054
        elif mode_index == 2: E = 4.201

        first_stage_end = self.work_time[0]

        mass_total = self.changed_mass(time_)
        mass_effective = mass_total.copy()
        for i in range(len(mass_effective)):
            s = 0
            h = 0
            if self.classes[i] in ["Fuel"] and self.stages[i] in ["First"]:
                h = f1lev * (first_stage_end - time_)/first_stage_end
                s = self.structural_values[0]
            elif self.classes[i] in ["Oxidizer"] and self.stages[i] in ["First"]:
                h = o1lev * (first_stage_end - time_)/first_stage_end
                s = self.structural_values[0]
            elif self.classes[i] in ["Fuel"] and self.stages[i] in ["Second"]:
                h = f2lev
                s = self.structural_values[1]
            elif self.classes[i] in ["Oxidizer"] and self.stages[i] in ["Second"]:
                h = o2lev
                s = self.structural_values[1]

            if s!=0:
                construction_ratio = 1/s
                construction_mass = mass_effective[i] * construction_ratio
                fluid_mass = mass_effective[i] - construction_mass
                mass_effective[i] = construction_mass + fluid_mass*self.max_diameter/(E*h)*np.tanh(E*2*h/self.max_diameter)/(E*E-1)

        return mass_effective

    # def effective_mass(self, time_, mode_index):
    #     mass_total = self.changed_mass(time_)
    #     mass_effective = mass_total.copy()

    #     first_stage_end = self.work_time[0]
    #     for i in range(len(mass_effective)):
    #         if self.classes[i] in ["Fuel", "Oxidizer"]:
    #             construction_ratio = 1/self.structural_values[0]
    #             construction_mass = mass_total[i] * construction_ratio
    #             fluid_mass = mass_total[i] - construction_mass

    #             if mode_index == 0: E = 1.841
    #             if mode_index == 1: E = 3.054
    #             if mode_index == 2: E = 4.201

    #             if fluid_mass > 0:
    #                 h = 1 # нужны высоты столбов жидкости в моменте
    #                 kt = self.max_diameter/(E*h)*np.tanh(E*2*h/self.max_diameter)/(E*E-1)
    #                 # здесь надо получить высоту h заполнения каждого бака для рсчета коэффициентов участия

    #                 mass_effective[i] = construction_mass + kt * fluid_mass

    #     return mass_effective

    def _calculate_flight_dynamics(self):
        """Расчет динамических параметров полета"""
        self.maximum_area = max(self.distr_data["area"])
        self.delta_level_ox = []
        self.delta_level_fu = []
        for k in range(self.block_number):
            self.delta_level_ox.append(
                self.delta_mass_ox[k] / self.oxidizer_density / self.maximum_area
            )
            self.delta_level_fu.append(
                self.delta_mass_fu[k] / self.fuel_density / self.maximum_area
            )

        self.thrust_vector = []
        self.mass_vector = []
        self.time_vector = []
        self.static_vector = []
        self.inertia_vector = []
        self.center_vector = []

        time = 0
        current_mass = self.full_mass
        stage_dropped = []
        stage_separation_times = []
        t_sep = 0
        for i in range((self.block_number)):
            stage_dropped.append(False)
            t_sep += self.work_time[i]
            stage_separation_times.append(t_sep)

        self.mass_vector.append(current_mass)
        self.time_vector.append(time)
        self.thrust_vector.append(self.thrust[0])

        self.static_moment = constants.calculate_static(
            self.full_mass, self.rocket_length
        )
        self.inertia_moment = constants.calculate_inertia(
            self.full_mass, self.rocket_length, self.rocket_length, self.max_diameter
        )
        self.static_vector.append(self.static_moment)
        self.inertia_vector.append(self.inertia_moment)
        self.center_vector.append(
            self.rocket_length - self.static_moment / current_mass
        )
        thrust = 0
        cent = 0
        checkpoint = []
        checkpoint.append(self.full_mass - self.propellant_mass[0])
        checkpoint.append(
            self.full_mass
            - self.propellant_mass[0]
            - self.structural_mass[0]
            - self.propellant_mass[1]
        )
        prop_test = [0, 0]
        while time < stage_separation_times[-1] + 30:
            self.static_vector.append(self.static_moment)
            self.inertia_vector.append(self.inertia_moment)

            current_stage = None
            for i in range(self.block_number):
                if time < stage_separation_times[i]:
                    current_stage = i
                    break

            if current_stage is not None:
                current_mass -= self.delta_mass[current_stage] * constants.timestep
                self.static_moment -= (
                    constants.calculate_static(
                        self.delta_mass_fu[i],
                        self.fuel_coordinates[i].end
                        + self.fuel_coordinates[i].end
                        - self.delta_level_fu[i],
                    )
                    * constants.timestep
                )
                self.static_moment -= (
                    constants.calculate_static(
                        self.delta_mass_ox[i],
                        self.oxidyzer_coordinates[i].end
                        + self.oxidyzer_coordinates[i].end
                        - self.delta_level_ox[i],
                    )
                    * constants.timestep
                )
                self.inertia_moment -= (
                    constants.calculate_inertia(
                        self.delta_mass_fu[i],
                        self.fuel_coordinates[i].end
                        + self.fuel_coordinates[i].end
                        - self.delta_level_fu[i],
                        self.fuel_coordinates[i].end
                        - self.fuel_coordinates[i].end
                        - self.delta_level_fu[i],
                        self.max_diameter,
                    )
                    * constants.timestep
                )
                self.inertia_moment -= (
                    constants.calculate_inertia(
                        self.delta_mass_ox[i],
                        self.oxidyzer_coordinates[i].end
                        + self.oxidyzer_coordinates[i].end
                        - self.delta_level_ox[i],
                        self.fuel_coordinates[i].end
                        - self.fuel_coordinates[i].end
                        - self.delta_level_fu[i],
                        self.max_diameter,
                    )
                    * constants.timestep
                )
                thrust = self.thrust[i]
                prop_test[current_stage] += (
                    self.delta_mass[current_stage] * constants.timestep
                )
                thrust = self.thrust[current_stage]
                cent = self.static_moment / current_mass

            for i in range(self.block_number):
                if not stage_dropped[i] and time >= stage_separation_times[i]:
                    current_mass -= self.structural_mass[i]
                    self.static_moment -= constants.calculate_static(
                        self.structural_mass[i],
                        self.structural_coordinates[i].end
                        + self.structural_coordinates[i].start,
                    )
                    self.inertia_moment -= constants.calculate_inertia(
                        self.structural_mass[i],
                        self.structural_coordinates[i].end
                        + self.structural_coordinates[i].start,
                        self.structural_coordinates[i].length,
                        self.max_diameter,
                    )
                    stage_dropped[i] = True
                    print(
                        f"Отделена {i+1}-я ступень в t={time} с, расчетное время: {stage_separation_times[i]} c, текущая масса: {current_mass}"
                    )

            self.mass_vector.append(current_mass)
            self.time_vector.append(time)
            self.thrust_vector.append(thrust)
            self.center_vector.append(cent)
            time += constants.timestep

    def get_step_length(self):
        return self.steps

    def get_work_time(self):
        return self.work_time

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

    def get_full_time(self):
        return self.full_time

    def get_part_diameters(self):
        res = []
        for i in range(self.block_number + 1):
            res.append(self.diameters[-1])
        return res

    def get_part_length(self):
        return self.Length_parts

    def get_thrust_from_time(self, time):
        for k in range(len(self.time_vector)):
            if abs(self.time_vector[k] - time) < constants.timestep:
                return self.thrust_vector[k]
        return None

    def get_mass_from_time(self, time):
        for k in range(len(self.time_vector)):
            if abs(self.time_vector[k] - time) < constants.timestep:
                return self.mass_vector[k]
        return None

    def get_inertia_from_time(self, time):
        for k in range(len(self.inertia_vector)):
            if abs(self.time_vector[k] - time) < constants.timestep:
                return self.inertia_vector[k]
        return None

    def get_center_from_time(self, time):
        for k in range(len(self.center_vector)):
            if abs(self.time_vector[k] - time) < constants.timestep:
                return self.center_vector[k]
        return None

    def get_propellant_from_time(self, time):
        for k in range(len(self.time_vector)):
            if abs(self.time_vector[k] - time) < constants.timestep:
                return self.thrust_vector[k]
        return None


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Создаем объект ракеты
rp = rocket_parser()

# Выбираем моменты времени для анализа

ti = 0
time_points = []
while ti < rp.work_time[0]:
    time_points.append(ti)
    ti +=5
# time_points = [0, 10, 20, 30, 40, 50, 90, 100, rp.full_time - 20, rp.full_time - 10, rp.full_time]

# Создаем цветовую карту (от темного к светлому)
colors = cm.plasma(np.linspace(0, 0.9, len(time_points)))

# Создаем фигуру с двумя подграфиками
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

# Получаем координаты по длине ракеты
x_coords = rp.asc_length

color_pairs = [
    ([0.68, 0.85, 0.9], [0, 0, 1]),
    ([1, 0.68, 0.68], [1, 0, 0]),
    ([0.8, 0.8, 0.8], [0, 0, 0])
]

for i, t in enumerate(time_points):
    if t <= rp.full_time + 30:
        mass_distribution = rp.effective_mass(t,0)

        ax1.plot(x_coords, mass_distribution,
                color = constants.interpolate_color(color_pairs[0][0], color_pairs[0][1], i, len(time_points)),
                linewidth=2,
                alpha=0.8,
                label=f't = {t:.1f} с')

ax1.set_xlabel('Длина ракеты, м')
ax1.set_ylabel('Масса в сечении, кг')
ax1.set_title('Распределение массы по длине ракеты в разные моменты времени')
ax1.grid(True, alpha=0.3)

plt.show()
