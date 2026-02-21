import json
import pandas as pd
import math
import numpy as np
import constants

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
        "CH4" : constants.density.CH4.value
    }
    return density_map.get(propellant_type, 0)

def calculate_static(mass_, shoulder):
    return 0.5 * mass_ * shoulder

def calculate_inertia(mass_, shoulder, shoulder_diff, diameter):
    return 0.25 * mass_ * (math.pow(shoulder, 2) + 0.333 * math.pow(shoulder_diff, 2) + math.pow((diameter/2), 2))

class rocket_parser:
    def __init__(self, json_filename, csv_filename='resources/rocket_data.csv'):
        with open(json_filename, 'r') as r_file:
            r_data = json.load(r_file)
        df = pd.read_csv(csv_filename)
        self.csv_data = df
        self.steps = df["step"].values
        self.lengths = df["length"].values
        self.diameters = df["diameter"].values
        self.areas = df["area"].values
        self.masses = df["mass"].values
        self.stiffness = df["stiffness"].values
        self.classes = df["class"].values
        self.stages = df["stage"].values
        self.max_diameter = max(self.diameters)
        self.maximum_area = (math.pi*(self.max_diameter**2))/4
        self.rocket_length = max(self.lengths)

        self.name = r_data["name"]
        self.payload_mass = r_data["payload_mass"]
        self.payload_str = r_data["payload_structure"]
        self.block_mass = r_data["block_mass"]
        self.full_mass = self.payload_mass + sum(self.block_mass)

        self.block_number = r_data["block_number"]
        self.thrust = r_data["thrust"]
        self.exhaust_velocity = r_data["exhaust_velocity"]
        self.components_ratio = r_data["components_ratio"]
        self.structural_values = r_data["structural_values"]
        self.interstep = r_data["integration_step"]
        self.thrust = r_data["thrust"]
        self.attack_coefs = r_data["attack_coefs"]
        self.thrust_ratio = r_data["thrust_ratio"]
        self.thickness = r_data.get("thickness", 4.0)

        self.fuel = r_data.get("fuel", "CH4")
        self.oxidizer = r_data.get("oxidizer", "LOX")
        self.fuel_density = read_propellant_density(self.fuel)
        self.oxidizer_density = read_propellant_density(self.oxidizer)

        self._identify_section_indices()

        self._calculate_stage_parameters()
        self._calculate_flight_dynamics()

    def _identify_section_indices(self):
        """Определяет индексы сечений для разных типов компонентов"""
        self.section_indices = {
            'payload': [],
            'construction': [],
            'oxidizer': [],
            'fuel': [],
            'tail': []
        }
        self.stage_boundaries = {}
        unique_stages = np.unique(self.stages)
        for stage in unique_stages:
            stage_indices = np.where(self.stages == stage)[0]
            self.stage_boundaries[stage] = {
                'start': min(stage_indices),
                'end': max(stage_indices)
            }


        for i, (cls, stage) in enumerate(zip(self.classes, self.stages)):
            key = f"{cls.lower()}_{stage.lower()}"
            if key not in self.section_indices:
                self.section_indices[key] = []
            self.section_indices[key].append(i)

            cls_lower = cls.lower()
            if cls_lower in self.section_indices:
                self.section_indices[cls_lower].append(i)

    def _calculate_stage_parameters(self):
        """Расчет параметров ступеней"""
        self.propellant_mass = []
        self.delta_mass = []
        self.stage_mass = []
        self.structural_mass = []
        self.delta_mass_ox = []
        self.delta_mass_fu = []
        self.work_time = []


        self.stage_masses_from_csv = {}
        for stage in np.unique(self.stages):
            stage_mask = self.stages == stage
            stage_mass = np.sum(self.masses[stage_mask])
            self.stage_masses_from_csv[stage] = stage_mass

        for k in range(self.block_number):

            self.propellant_mass.append(self.block_mass[k] * self.structural_values[k] / (self.structural_values[k] + 1))

            mass_ox = self.propellant_mass[k] * self.components_ratio / (self.components_ratio + 1)
            mass_fu = self.propellant_mass[k] / (self.components_ratio + 1)

            delta_mass = self.thrust[k] / self.exhaust_velocity[k]
            self.delta_mass.append(delta_mass)

            self.work_time.append(self.propellant_mass[k] / delta_mass)

            self.structural_mass.append(self.block_mass[k] - self.propellant_mass[k])

            self.delta_mass_ox.append(delta_mass * self.components_ratio / (self.components_ratio + 1))
            self.delta_mass_fu.append(delta_mass / (self.components_ratio + 1))

        self.full_time = sum(self.work_time)

    def _calculate_flight_dynamics(self):
        """Расчет динамических параметров полета"""
        self.thrust_vector = []
        self.mass_vector = []
        self.time_vector = []
        self.static_vector = []
        self.inertia_vector = []
        self.center_vector = []

        time = 0
        current_stage = 0
        current_mass = self.full_mass

        self.mass_vector.append(current_mass)
        self.time_vector.append(time)
        self.thrust_vector.append(self.thrust[current_stage])

        static_moment = self._calculate_static_moment(time)
        inertia_moment = self._calculate_inertia_moment(time)

        self.static_vector.append(static_moment)
        self.inertia_vector.append(inertia_moment)
        self.center_vector.append(static_moment / current_mass if current_mass > 0 else 0)

        while current_stage < self.block_number:
            stage_propellant_mass = self.propellant_mass[current_stage]
            stage_work_time = self.work_time[current_stage]
            time_step = self.interstep

            for t in np.arange(time_step, stage_work_time + time_step, time_step):
                if t > stage_work_time:
                    t = stage_work_time

                consumed_propellant = (t / stage_work_time) * stage_propellant_mass
                current_mass = self.full_mass - consumed_propellant

                current_time = time + t
                self.mass_vector.append(current_mass)
                self.time_vector.append(current_time)
                self.thrust_vector.append(self.thrust[current_stage])

                static_moment = self._calculate_static_moment(current_time)
                inertia_moment = self._calculate_inertia_moment(current_time)

                self.static_vector.append(static_moment)
                self.inertia_vector.append(inertia_moment)
                self.center_vector.append(static_moment / current_mass if current_mass > 0 else 0)

            # Переход к следующей ступени
            time += stage_work_time
            current_stage += 1

            if current_stage < self.block_number:
                # Сброс отработавшей ступени
                current_mass -= self.structural_mass[current_stage - 1]
                self.full_mass = current_mass

    def _calculate_static_moment(self, time):
        """Расчет статического момента для заданного времени"""
        static_moment = 0

        # Учитываем все сечения
        for i in range(len(self.lengths)):
            # Получаем массу сечения (может меняться со временем для топливных отсеков)
            section_mass = self._get_section_mass_at_time(i, time)

            # Плечо - координата центра сечения
            shoulder = self.lengths[i]

            static_moment += calculate_static(section_mass, shoulder)

        return static_moment

    def _calculate_inertia_moment(self, time):
        """Расчет момента инерции для заданного времени"""
        inertia_moment = 0

        for i in range(len(self.lengths)):
            section_mass = self._get_section_mass_at_time(i, time)
            shoulder = self.lengths[i]
            diameter = self.diameters[i]

            # Для упрощения считаем shoulder_diff = 0 (сечение считается точечной массой)
            shoulder_diff = 0

            inertia_moment += calculate_inertia(section_mass, shoulder, shoulder_diff, diameter)

        return inertia_moment

    def _get_section_mass_at_time(self, section_idx, time):
        """Получить массу сечения в заданный момент времени"""
        section_class = self.classes[section_idx].lower()
        base_mass = self.masses[section_idx]

        if section_class in ['oxidizer', 'fuel']:
            active_stage = self._get_active_stage_at_time(time)

            if active_stage is not None:
                stage_work_time = self.work_time[active_stage]
                elapsed_time = time - sum(self.work_time[:active_stage])

                if elapsed_time < 0:
                    return base_mass
                elif elapsed_time > stage_work_time:
                    return base_mass * 0.1
                else:
                    consumption_ratio = elapsed_time / stage_work_time
                    return base_mass * (1 - consumption_ratio)

        return base_mass

    def _get_active_stage_at_time(self, time):
        """Определить активную ступень в заданное время"""
        cumulative_time = 0
        for i, work_time in enumerate(self.work_time):
            cumulative_time += work_time
            if time <= cumulative_time:
                return i
        return None

    def get_block_number(self):
        return self.block_number

    def get_rocket_length(self):
        return self.rocket_length

    def get_structural_values(self):
        return self.structural_values

    def get_structural_mass(self):
        return self.structural_mass

    def get_payload(self):
        return self.payload_mass

    def get_full_mass(self):
        return self.full_mass

    def get_propellant_mass(self):
        return self.propellant_mass

    def get_delta_mass(self):
        return self.delta_mass

    def get_stage_mass(self):
        return self.stage_mass

    def get_delta_mass_ox(self):
        return self.delta_mass_ox

    def get_delta_mass_fu(self):
        return self.delta_mass_fu

    def get_interstep(self):
        return self.interstep

    def get_work_time(self):
        return self.work_time

    def get_full_time(self):
        return self.full_time

    def get_part_diameters(self):
        return self.diameters

    def get_thrust(self):
        return self.thrust

    def get_csv_data(self):
        return self.csv_data

    def get_section_data(self):
        """Возвращает данные о всех сечениях"""
        return {
            'lengths': self.lengths,
            'diameters': self.diameters,
            'areas': self.areas,
            'masses': self.masses,
            'stiffness': self.stiffness,
            'classes': self.classes,
            'stages': self.stages
        }

    def vector_time(self):
        return self.time_vector

    def vector_mass(self):
        return self.mass_vector

    def vector_static(self):
        return self.static_vector

    def vector_inertia(self):
        return self.inertia_vector

    def vector_center(self):
        return self.center_vector

    def vector_thrust(self):
        return self.thrust_vector

    def get_thrust_from_time(self, time):
        for k in range(len(self.time_vector)):
            if abs(self.time_vector[k] - time) < self.interstep:
                return self.thrust_vector[k]
        return None

    def get_mass_from_time(self, time):
        for k in range(len(self.time_vector)):
            if abs(self.time_vector[k] - time) < self.interstep:
                return self.mass_vector[k]
        return None

    def get_inertia_from_time(self, time):
        for k in range(len(self.inertia_vector)):
            if abs(self.time_vector[k] - time) < self.interstep:
                return self.inertia_vector[k]
        return None

    def get_center_from_time(self, time):
        for k in range(len(self.center_vector)):
            if abs(self.time_vector[k] - time) < self.interstep:
                return self.center_vector[k]
        return None

    def get_propellant_from_time(self, time):
        for k in range(len(self.time_vector)):
            if abs(self.time_vector[k] - time) < self.interstep:
                return self.thrust_vector[k]
        return None