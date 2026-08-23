import pandas as pd
import numpy as np
import basis
import json
import matplotlib.pyplot as plt
import attack
import types

def read_propellant_density(propellant_type):
    try:
        return getattr(basis.density, propellant_type).value
    except AttributeError:
        return 0

class Coord_element:
    def __init__(self, start, length):
        self.start = float(start)
        self.length = float(length)
        self.end = self.start + self.length

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

class Flight_dataset:
    def __init__(self):
         self.thrusts = []
         self.masses = []
         self.times = []
         self.statics = []
         self.inertions = []
         self.centers = []

class Rocket_parser:
    def __init__(self):
        rocket = basis.current_rocket
        json_filename = "rockets/" + rocket + "/constant.json"

        with open(json_filename, "r") as r_file:
            r_data = json.load(r_file)

        # common data
        self.name = r_data["name"]
        self.payload_mass    = r_data["payload_mass"]
        self.rocket_type     = r_data["type"]
        self.attack_coefs    = r_data["attack_coefs"]
        self.thrust_ratio    = r_data["thrust_ratio"]
        self.boosters_number = r_data.get("booster_number",0)
    
        # central stages data
        self.main  = types.SimpleNamespace()
        self.boost = types.SimpleNamespace()

        self.main.csv_filename = "rockets/" + rocket + "/distributed.csv"
        self.main.block_number = r_data["block_number"]
        self.main.block_mass = r_data["block_mass"]
        self.main.thrust = r_data["thrust"]
        self.main.structural_values = r_data["structural_values"]
        self.main.exhaust_velocity = r_data["exhaust_velocity"]
        self.main.fuel = r_data["fuel"]
        self.main.oxidizer = r_data["oxidizer"]
        self.main.components_ratio = r_data["components_ratio"]
        self.main.is_booster = False

        # boosters data
        self.boost.csv_filename = "rockets/" + rocket + "/distributed_boost.csv"
        self.boost.block_number = r_data.get("booster_block_number",0)
        self.boost.block_mass = [r_data.get("booster_mass",0.0)]
        self.boost.thrust = [r_data.get("booster_thrust",0)]
        self.boost.structural_values = [r_data.get("booster_structural_values",0.0)]
        self.boost.exhaust_velocity = [r_data.get("booster_exhaust_velocity",0.0)]
        self.boost.fuel = r_data.get("booster_fuel", "")
        self.boost.oxidizer = r_data.get("booster_oxidizer", "")
        self.boost.components_ratio = r_data.get("booster_components_ratio",0)
        self.boost.is_booster = True

        # handled data
        self.full_mass = self.payload_mass + sum(self.main.block_mass) + self.boosters_number * sum(self.boost.block_mass)
        print(f"Payload mass: {self.payload_mass}")
        print(f"Block mass:   {self.main.block_mass}")
        print(f"Booster mass: {self.boost.block_mass}")
        print(f"Full mass:    {self.full_mass}")

        self._initialize_partgroup(self.main)
        self._initialize_partgroup(self.boost)

        self.work_time = []

        if self.rocket_type != "tandem":
            self.work_time.append(max(self.boost.work_time))
            self.main.work_time[0]-= max(self.boost.work_time)
            self.main.full_mass += self.payload_mass
        self.work_time += self.main.work_time
        self.full_time = sum(self.work_time)

        self.complex = types.SimpleNamespace()
        self._merge_partgroup(self.main, self.boost, self.complex)
    
    def _initialize_partgroup(self, group):
        self._calculate_stage_parameters(group)
        group.distr_data  = self._distributed_handler(group)
        group.flight_data = self._calculate_flight_dynamics(group)

    def _merge_partgroup(self, main_, boost_, complex_):
        # distributed data
        complex_.distr_data = Distributed_dataset()
        if main_.distr_data.numbers < boost_.distr_data.numbers:
            main_, boost_ = boost_, main_
        complex_.distr_data.numbers = main_.distr_data.numbers
        complex_.distr_data.lengths = main_.distr_data.lengths
        complex_.distr_data.cumlengths = main_.distr_data.cumlengths

        # Диаметры
        n = len(boost_.distr_data.diameters)
        complex_.distr_data.diameters = (
            main_.distr_data.diameters[:-n] +
            [
                main_.distr_data.diameters[-n + i] + 2 * boost_.distr_data.diameters[i]
                for i in range(n)
            ]
        )

        # Массы
        n = len(boost_.distr_data.masses)
        complex_.distr_data.masses = (
            main_.distr_data.masses[:-n] +
            [
                main_.distr_data.masses[-n + i] + self.boosters_number * boost_.distr_data.masses[i]
                for i in range(n)
            ]
        )
        # flight data
        complex_.flight_data = Flight_dataset()

        # Тяга
        n = len(boost_.flight_data.thrusts)
        complex_.flight_data.thrusts = [
            main_.flight_data.thrusts[i] + self.boosters_number * boost_.flight_data.thrusts[i]
            for i in range(n)
        ] + main_.flight_data.thrusts[n:]

        # Масса
        n = len(boost_.flight_data.masses)
        complex_.flight_data.masses = [
            main_.flight_data.masses[i] + self.boosters_number * boost_.flight_data.masses[i]
            for i in range(n)
        ] + main_.flight_data.masses[n:]

        complex_.flight_data.times = main_.flight_data.times
        shoulder = main_.max_diameter/2 + boost_.max_diameter/2

        # Момент инерции
        n = len(boost_.flight_data.inertions)
        complex_.flight_data.inertions = [
            main_.flight_data.inertions[i] + self.boosters_number * boost_.flight_data.masses[i] * shoulder**2
            for i in range(n)
        ] + main_.flight_data.inertions[n:]

        # Статический момент
        n = len(boost_.flight_data.statics)
        complex_.flight_data.statics = [
            main_.flight_data.statics[i] + self.boosters_number * boost_.flight_data.masses[i] * shoulder
            for i in range(n)
        ] + main_.flight_data.statics[n:]

        # Центр масс
        complex_.flight_data.centers = [a/b for a,b in zip(complex_.flight_data.statics,complex_.flight_data.masses)]

        complex_.flight_data.inertions = [
            I - m * center**2  # Перенос к центру масс
            for I, m, center in zip(complex_.flight_data.inertions, 
                                    complex_.flight_data.masses,
                                    complex_.flight_data.centers)
        ]

    def _calculate_stage_parameters(self, group):
        """Расчет параметров ступеней"""

        group.fuel_density        = read_propellant_density(group.fuel)
        group.oxidizer_density    = read_propellant_density(group.oxidizer)

        group.propellant_mass = []
        group.mass_ox = []
        group.mass_fu = []
        group.delta_mass = []
        group.work_time = []
        group.structural_mass = []
        group.delta_mass_ox = []
        group.delta_mass_fu = []

        for k in range(group.block_number):
            group.propellant_mass.append(group.block_mass[k] * group.structural_values[k] / (group.structural_values[k] + 1))
            group.mass_ox.append(group.propellant_mass[k] * group.components_ratio / (group.components_ratio + 1))
            group.mass_fu.append(group.propellant_mass[k] / (group.components_ratio + 1))
            delta_mass = group.thrust[k] / group.exhaust_velocity[k]
            group.delta_mass.append(delta_mass)
            group.work_time.append(group.propellant_mass[k] / group.delta_mass[k])
            group.structural_mass.append(group.block_mass[k] - group.propellant_mass[k])
            group.delta_mass_ox.append(delta_mass * group.components_ratio / (group.components_ratio + 1))
            group.delta_mass_fu.append(delta_mass / (group.components_ratio + 1))
        group.full_mass = sum(group.block_mass)

        if not group.is_booster:
            group.full_mass += self.payload_mass

    def _distributed_handler(self, group):

        start_  = Distributed_dataset()
        result_ = Distributed_dataset()

        df = pd.read_csv(group.csv_filename)
        start_.lengths = df["L"]
        start_.diameters = df["D"]
        start_.classes = df["Class"]
        start_.stages = df["Stage"]
        start_.cumlengths = np.cumsum(start_.lengths)
        start_.numbers = list(range(len(start_.lengths)))

        self.stages = ["Payload", "First", "Second", "Third"]
        self.classes = ["Head", "Oxidizer", "Fuel", "Construction", "Tail"]

        group.stages_lengths = []
        for s in self.stages: 
            group.stages_lengths.append(get_stage_length(s, start_))

        # Masses
        class_masses = {}
        class_masses.update({"Oxidizer": sum(group.mass_ox)})
        class_masses.update({"Fuel": sum(group.mass_fu)})
        class_masses.update({"Tail": self.payload_mass / 3})
        class_masses.update({"Construction": group.full_mass - sum(class_masses.values())})

        m_full = group.full_mass

        result_.lengths.append(0)
        result_.diameters.append(0)
        result_.stiffnesses.append(0)
        result_.areas.append(0)
        result_.volumes.append(0)

        if get_stage_length("Payload", start_) > 0:
            class_masses.update({"Head": self.payload_mass})
            class_masses["Construction"] -= class_masses["Head"]
            result_.classes.append("Head")
            result_.stages.append("Payload")
        else:
            result_.classes.append("Construction")
            result_.stages.append("First")

        result_.numbers = []
        result_.cumlengths = []

        li = 0
        num = 0

        while li < sum(start_.lengths):
            if li >= basis.lenstep / 2:
                result_.lengths.append(basis.lenstep)
            result_.cumlengths.append(round(li, 1))
            result_.numbers.append(num)
            found = False
            for i in range(len(start_.cumlengths) - 1):
                if (
                    li <= start_.cumlengths[i + 1]
                    and li > start_.cumlengths[i]
                ):
                    found = True
                    result_.classes.append(start_.classes[i + 1])
                    result_.stages.append(start_.stages[i + 1])
                    delta_diameter = (
                        start_.diameters[i + 1] - start_.diameters[i]
                    )
                    delta_length = (
                        start_.cumlengths[i + 1] - start_.cumlengths[i]
                    )

                    offset = li - start_.cumlengths[i]

                    if delta_length > 0:
                        current_diameter = (
                            start_.diameters[i]
                            + (delta_diameter / delta_length) * offset
                        )
                    else:
                        current_diameter = start_.diameters[i]

                    result_.diameters.append(current_diameter)
                    result_.stiffnesses.append(basis.calculate_stiffness(current_diameter))
                    result_.areas.append(basis.cross_sectional_area(current_diameter))
                    result_.volumes.append(result_.areas[-1] * basis.lenstep)

            li += basis.lenstep
            num += 1

        group.fuel_coordinates = []
        group.oxidyzer_coordinates = []
        group.structural_coordinates = []

        for s in self.stages:
            if get_stage_length(s, result_)>0 and s!="Payload":

                group.fuel_coordinates.append(
                    Coord_element(
                        get_start_stageclass(s, "Fuel",  result_), 
                        get_stageclass_length(s, "Fuel", result_)
                    )
                )

                group.oxidyzer_coordinates.append(
                    Coord_element(
                        get_start_stageclass(s, "Oxidizer",  result_),
                        get_stageclass_length(s, "Oxidizer", result_),
                    )
                )

                group.structural_coordinates.append(
                    Coord_element(
                        get_start_class(s,  result_), 
                        get_class_length(s, result_))
                )

        group.class_densities = {}
        group.class_densities.update({"Construction": class_masses["Construction"] / sum(result_.volumes)})
        for cw in class_masses.keys():
            if cw != "Construction":
                group.class_densities.update({cw: get_class_density(cw,  result_)})

        for j in range(len(result_.volumes)):
            m = group.class_densities["Construction"] * result_.volumes[j]
            for cw in class_masses.keys():
                if result_.classes[j] == cw and cw != "Construction":
                    m += get_class_density(cw,  result_) * result_.volumes[j]
            result_.masses.append(m)

        print("Destr mass:", sum(result_.masses))
        group.full_length = result_.cumlengths[-1]

        return result_

    def changed_groupmass(self, time_, group):
        """Расчет распределенной массы в заданный момент времени"""
        mass_t = group.distr_data.masses.copy()
        time_remaining = time_

        for stage in range(group.block_number):
            stage_time = group.work_time[stage]
            if time_remaining <= 0:
                break

            if time_remaining >= stage_time:
                # Ступень полностью отработала
                for i in range(len(mass_t)):
                    if group.distr_data.stages[i] in ["First", "Second", "Third"]:
                        if group.distr_data.classes[i] in ["Fuel", "Oxidizer"]:
                            # Оставляем только массу конструкции бака
                            mass_t[i] = group.distr_data.volumes[i] * group.class_densities["Construction"]
                        # Сегменты типа 'Construction', 'Tail' уже имеют правильную массу
                time_remaining -= stage_time

            else:

                burn_ratio = time_remaining / stage_time

                fuel_segments = []
                oxid_segments = []

                for i in range(len(mass_t)):
                    if group.distr_data.stages[i] in ["First", "Second", "Third"]:
                        if group.distr_data.classes[i] == "Fuel":
                            fuel_segments.append((i, group.distr_data.cumlengths[i]))
                        elif group.distr_data.classes[i] == "Oxidizer":
                            oxid_segments.append((i, group.distr_data.cumlengths[i]))

                fuel_segments.sort(key=lambda x: -x[1], reverse=True)
                oxid_segments.sort(key=lambda x: -x[1], reverse=True)

                fuel_mass_to_remove = group.mass_fu[stage] * burn_ratio
                oxid_mass_to_remove = group.mass_ox[stage] * burn_ratio

                for idx, _ in fuel_segments:
                    if fuel_mass_to_remove <= 0:
                        break
                    construction_mass = group.distr_data.volumes[idx] * group.class_densities["Construction"]
                    current_fuel_mass = mass_t[idx] - construction_mass
                    if current_fuel_mass > 0:
                        remove = min(current_fuel_mass, fuel_mass_to_remove)
                        mass_t[idx] -= remove
                        fuel_mass_to_remove -= remove

                for idx, _ in oxid_segments:
                    if oxid_mass_to_remove <= 0:
                        break
                    construction_mass = group.distr_data.volumes[idx] * group.class_densities["Construction"]
                    current_oxid_mass = mass_t[idx] - construction_mass
                    if current_oxid_mass > 0:
                        remove = min(current_oxid_mass, oxid_mass_to_remove)
                        mass_t[idx] -= remove
                        oxid_mass_to_remove -= remove

                time_remaining = 0
        return mass_t

    # TODO: Make separation
    def changed_mass(self, time_):
        if time_ > self.work_time[0]:
           raise ValueError("Поддерживается только время полета первой ступени") 
        return self.changed_groupmass(time_, self.main) + self.boosters_number * self.changed_groupmass(time_, self.boost)

    def _calculate_flight_dynamics(self, group):
        """Расчет динамических параметров полета"""
        group.max_area     = max(group.distr_data.areas)
        group.max_diameter = max(group.distr_data.diameters)
        delta_level_ox = []
        delta_level_fu = []
        for k in range(group.block_number):
            delta_level_ox.append(
                group.delta_mass_ox[k] / group.oxidizer_density / group.max_area
            )
            delta_level_fu.append(
                group.delta_mass_fu[k] / group.fuel_density / group.max_area
            )

        fdata = Flight_dataset()

        time = 0
        current_mass = group.full_mass
        stage_dropped = []
        stage_separation_times = []
        t_sep = 0
        for i in range((group.block_number)):
            stage_dropped.append(False)
            t_sep += group.work_time[i]
            stage_separation_times.append(t_sep)

        fdata.masses.append(current_mass)
        fdata.times.append(time)
        fdata.thrusts.append(group.thrust[0])

        group.static_moment = basis.calculate_static(
            group.full_mass, group.full_length
        ) * 1.3 # TODO: Fox error
        group.inertia_moment = basis.calculate_inertia(
            group.full_mass, group.full_length, group.full_length, group.max_diameter
        )
        fdata.statics.append(group.static_moment)
        fdata.inertions.append(group.inertia_moment)
        fdata.centers.append(group.full_length - group.static_moment / current_mass) # TODO: Fix jump error

        while time < stage_separation_times[-1]:
            fdata.statics.append(group.static_moment)
            fdata.inertions.append(group.inertia_moment)

            current_stage = None
            for i in range(group.block_number):
                if time < stage_separation_times[i]:
                    current_stage = i
                    break

            if current_stage is not None:
                current_mass -= group.delta_mass[current_stage] * basis.timestep
                lower_point = group.fuel_coordinates[current_stage].end
                upper_point = group.fuel_coordinates[current_stage].end - delta_level_fu[current_stage]
                group.static_moment -= (
                    basis.calculate_static(group.delta_mass_fu[current_stage], lower_point + upper_point) * basis.timestep
                )
                lower_point = group.oxidyzer_coordinates[current_stage].end
                upper_point = group.oxidyzer_coordinates[current_stage].end - delta_level_ox[current_stage]
                group.static_moment -= (
                    basis.calculate_static(group.delta_mass_ox[current_stage], lower_point + upper_point) * basis.timestep
                )
                lower_point = group.fuel_coordinates[current_stage].end
                upper_point = group.fuel_coordinates[current_stage].end - delta_level_fu[current_stage]
                group.inertia_moment -= (
                    basis.calculate_inertia(
                        group.delta_mass_fu[current_stage],
                        lower_point + upper_point,
                        lower_point - upper_point,
                        group.max_diameter,
                    )
                    * basis.timestep
                )
                lower_point = group.oxidyzer_coordinates[current_stage].end
                upper_point = group.oxidyzer_coordinates[current_stage].end - delta_level_ox[current_stage]
                group.inertia_moment -= (
                    basis.calculate_inertia(
                        group.delta_mass_ox[current_stage],
                        lower_point + upper_point,
                        lower_point - upper_point,
                        group.max_diameter,
                    )
                    * basis.timestep
                )

                thrust = group.thrust[current_stage]

            for i in range(group.block_number):
                if not stage_dropped[i] and time >= stage_separation_times[i]:
                    current_mass -= group.structural_mass[i]
                    group.static_moment -= basis.calculate_static(
                        group.structural_mass[i],
                        group.structural_coordinates[i].end
                        + group.structural_coordinates[i].start,
                    )
                    group.inertia_moment -= basis.calculate_inertia(
                        group.structural_mass[i],
                        group.structural_coordinates[i].end
                        + group.structural_coordinates[i].start,
                        group.structural_coordinates[i].length,
                        group.max_diameter,
                    )
                    stage_dropped[i] = True
                    print(
                        f"Отделена {i+1}-я ступень в t={time} с, расчетное время: {stage_separation_times[i]} c, текущая масса: {current_mass}"
                    )
            cent = group.static_moment / current_mass


            fdata.masses.append(current_mass)
            fdata.times.append(time)
            fdata.thrusts.append(thrust)
            fdata.centers.append(cent)
            time += basis.timestep
        return fdata

    def attack_func(self, vel, time):
        """Получение угла атаки по коэффициентам из парсера"""
        alpha_obj = attack.alpha(
            self.attack_coefs[0], self.attack_coefs[1], self.work_time[0], False
        )
        alpha_val = alpha_obj.calculate_alpha(vel, time)
        return max(-30, min(30, alpha_val))

    def get_work_time(self):
        return self.work_time

    group_for = types.SimpleNamespace()

    def get_data(self, is_booster):
        if is_booster == True:
            self.group_for = self.boost
            return self.boost.distr_data
        else:
            self.group_for = self.main
            return self.main.distr_data

    def get_block_number(self):
        return self.group_for.block_number

    def get_delta_mass_fu(self):
        return self.group_for.delta_mass_fu

    def get_delta_mass_ox(self):
        return self.group_for.delta_mass_ox

    def get_coordinates_fu(self):
        return self.group_for.fuel_coordinates

    def get_coordinates_ox(self):
        return self.group_for.oxidyzer_coordinates

    def get_full_length(self):
        return self.group_for.full_length

    def get_changed_mass(self, time_):
        return self.changed_groupmass(time_, self.group_for)

rp = Rocket_parser()
