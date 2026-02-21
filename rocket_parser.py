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
        "CH4" : constants.density.CH4.value
    }
    return density_map.get(propellant_type, 0)

class coord_element:
    def __init__(self, start, end, length, propellant_type=None):
        self.start = float(start)
        self.end = float(end)
        self.length = float(length)
        self.propellant_type = propellant_type

class rocket_parser:
    def __init__(self, rocket):
        json_filename  = "rocket_lib/" + rocket + "/constant.json"
        csv_filename = "rocket_lib/" + rocket + "/distributed.csv"

        with open(json_filename, 'r') as r_file:
            r_data = json.load(r_file)

        # initial data
        self.name = r_data['name']
        self.block_number = r_data['block_number']
        self.payload_mass = r_data['payload_mass']
        self.block_mass = r_data['block_mass']
        self.components_ratio = r_data['components_ratio']
        self.thrust_ratio = r_data['thrust_ratio']
        self.exhaust_velocity = r_data['exhaust_velocity']
        self.thrust = r_data['thrust']
        self.structural_values = r_data['structural_values']
        self.fuel = r_data['fuel']
        self.oxidizer = r_data['oxidizer']
        self.attack_coefs = r_data['attack_coefs']

        self.Length_secs  = []
        self.Length_parts = []

        # handled data
        self.full_mass = self.payload_mass + sum(self.block_mass)
        print(f"Payload mass: {self.payload_mass}")
        print(f"Block mass: {self.block_mass}")
        print(f"Full mass: {self.full_mass}")
        self.fuel_density = read_propellant_density(self.fuel)
        self.oxidizer_density = read_propellant_density(self.oxidizer)

        self._calculate_stage_parameters()
        self.distr_data = self._distributed_handler(csv_filename)
        self.rocket_length = self.distr_data["length"][-1]
        self._initialize_distributed_arrays()
        self._calculate_flight_dynamics()

    def _initialize_distributed_arrays(self):
        """Distributed parameters initialization"""
        self.asc_length = self.distr_data['length']
        self.diameters = self.distr_data['diameter']
        self.masses = self.distr_data['mass']
        self.stiffnesses = self.distr_data['stiffness']
        self.areas = self.distr_data['area']
        self.classes = self.distr_data['class']
        self.stages = self.distr_data['stage']
        self.steps = self.distr_data['step']

    def _distributed_handler(self, filename):
        """Distributed parameters handling & Further calculation"""
        df = pd.read_csv(filename)

        L_sec_vector = df['L']
        D_sec_vector = df['D']
        Class_sec_vector = df['Class']
        Stage_sec_vector = df['Stage']

        # Classification handling
        class_counts = df['Class'].value_counts()
        class_counts_dict = class_counts.to_dict()
        tail_count = class_counts_dict.get("Tail", 0)
        head_count = class_counts_dict.get("Head", 0)

        self.Length_parts.append(df[df['Stage'] == 'Payload']['L'].sum())
        self.Length_parts.append(df[df['Stage'] == 'First']['L'].sum())
        self.Length_parts.append(df[df['Stage'] == 'Second']['L'].sum())

        #Masses
        m_payload = self.payload_mass
        m_fuel = sum(self.mass_fu)
        m_oxidyzer = sum(self.mass_ox)
        m_full = self.full_mass
        m_engine = m_payload/3
        self.max_diameter = max(D_sec_vector)
        m_construction = m_full - m_oxidyzer - m_fuel - m_payload - m_engine

        # Vector init
        stiff_vector = []
        area_vector = []
        mass_vector = []
        rocket_volume = 0
        step = constants.lenstep
        len_t = 0

        # Новые векторы для дискретизации
        L_sec_vector_sum_new = []
        D_sec_vector_new = []
        m_vector_new = []
        s_vector_new = []
        A_vector_new = []
        Class_sec_vector_new = []
        Stage_sec_vector_new = []

        # Координаты компонентов
        fuel_coords = []
        oxidyzer_coords = []
        stage_coords = {}  # Словарь для хранения координат ступеней
        current_stage = None
        current_stage_start = 0

        # Расчет жесткости и площадей
        for i, L in enumerate(L_sec_vector):
            stiff_vector.append(constants.calculate_stiffness(D_sec_vector[i]))
            area_vector.append(constants.cross_sectional_area(D_sec_vector[i]))
            rocket_volume += L * area_vector[-1]

        # Плотность конструкции
        dencity_free = m_construction / rocket_volume

        # Счетчики для распределения массы по ступеням
        f_count = self.block_number - 1
        o_count = self.block_number - 1

        current_position_from_nose = 0

        # Обработка каждого элемента
        for i, L in enumerate(L_sec_vector):
            d_delta = 0
            # Определение начала новой ступени
            stage_name = Stage_sec_vector[i]
            if stage_name != current_stage:
                if current_stage is not None:
                    # Сохраняем координаты предыдущей ступени
                    stage_coords[current_stage] = {
                        'start': current_stage_start,
                        'end': current_position_from_nose,
                        'length': current_position_from_nose - current_stage_start,
                        'components': []
                    }
                current_stage = stage_name
                current_stage_start = current_position_from_nose

            # Расчет массы элемента
            m_temp = dencity_free * area_vector[i] * L
            if Class_sec_vector[i] == "Head":
                m_temp += m_payload / (head_count if head_count > 0 else 1)
            elif Class_sec_vector[i] == "Oxidizer":
                m_temp += self.mass_ox[o_count]
                oxidyzer_start = current_position_from_nose
                oxidyzer_end = current_position_from_nose + L
                oxidyzer_coords.append((oxidyzer_start, oxidyzer_end, Stage_sec_vector[i]))
                o_count -= 1
            elif Class_sec_vector[i] == "Fuel":
                m_temp += self.mass_fu[f_count]
                fuel_start = current_position_from_nose
                fuel_end = current_position_from_nose + L
                fuel_coords.append((fuel_start, fuel_end, Stage_sec_vector[i]))
                f_count -= 1
            elif Class_sec_vector[i] == "Tail":
                m_temp += m_engine / (tail_count if tail_count > 0 else 1)
            elif Class_sec_vector[i] == "Construction":
                m_temp += 0

            mass_vector.append(m_temp)

            # Запись информации о компоненте ступени
            if current_stage in stage_coords:
                stage_coords[current_stage]['components'].append({
                    'class': Class_sec_vector[i],
                    'length': L,
                    'diameter': D_sec_vector[i],
                    'start': current_position_from_nose,
                    'end': current_position_from_nose + L
                })

            # Дискретизация элемента на шаги
            if i + 1 < len(D_sec_vector):
                d_delta = (D_sec_vector.iloc[i+1] - D_sec_vector.iloc[i]) / L

            element_len = 0

            target_length = L
            while element_len < target_length - 1e-10:
                current_pos = len_t + element_len

                D_current = D_sec_vector.iloc[i] + d_delta * element_len
                mass_per_length = mass_vector[i] / L if L > 0 else 0
                mass_element = mass_per_length * step * (D_current/self.max_diameter if self.max_diameter > 0 else 1)

                self.Length_secs.append(step)
                L_sec_vector_sum_new.append(current_pos)
                D_sec_vector_new.append(D_current)
                m_vector_new.append(mass_element)
                s_vector_new.append(constants.calculate_stiffness(D_current))
                A_vector_new.append(constants.cross_sectional_area(D_current))
                Class_sec_vector_new.append(Class_sec_vector[i])
                Stage_sec_vector_new.append(Stage_sec_vector[i])

                element_len += step

                if element_len > target_length:
                    remaining = target_length - (element_len - step)
                    if remaining > 1e-10:
                        pass
                    break

            current_position_from_nose += L
            len_t += L
        # Сохраняем координаты последней ступени
        if current_stage is not None:
            stage_coords[current_stage] = {
                'start': current_stage_start,
                'end': current_position_from_nose,
                'length': current_position_from_nose - current_stage_start,
                'components': stage_coords.get(current_stage, {}).get('components', [])
            }

        # Сохранение координат топлива и окислителя
        self.fuel_coordinates = []
        self.oxidyzer_coordinates = []
        self.structural_coordinates = []
        print("\n=== Координаты топлива от носа ракеты ===")
        for i, (start, end, stage) in enumerate(fuel_coords):
            print(f"Горючее {i+1} (ступень {stage}):")
            print(f"  Начало: {start:.2f} м от носа")
            print(f"  Конец:  {end:.2f} м от носа")
            print(f"  Длина:  {end-start:.2f} м")
            self.fuel_coordinates.append(coord_element(start, end, end-start))

        for i, (start, end, stage) in enumerate(oxidyzer_coords):
            print(f"Окислитель {i+1} (ступень {stage}):")
            print(f"  Начало: {start:.2f} м от носа")
            print(f"  Конец:  {end:.2f} м от носа")
            print(f"  Длина:  {end-start:.2f} м")
            self.oxidyzer_coordinates.append(coord_element(start, end, end-start))

        # Вывод информации о ступенях
        print("\n=== Координаты блоков ступеней ===")
        self.stage_coordinates = {}
        for stage_name, stage_info in sorted(stage_coords.items()):
            print(f"\nСтупень: {stage_name}")
            print(f"  Начало: {stage_info['start']:.2f} м от носа")
            print(f"  Конец:  {stage_info['end']:.2f} м от носа")
            print(f"  Длина:  {stage_info['length']:.2f} м")
            self.structural_coordinates.append(coord_element(stage_info['start'], stage_info['end'], stage_info['length']))

            # Вывод информации о компонентах ступени
            print(f"  Компоненты:")
            for comp in stage_info['components']:
                print(f"    - {comp['class']}: {comp['length']:.1f} м, Ø{comp['diameter']:.1f} м "
                    f"(от {comp['start']:.1f} до {comp['end']:.1f} м)")

        # Корректировка массы
        final_total_mass = sum(m_vector_new)
        dif = - ( final_total_mass - m_full )
        for i in range(len(m_vector_new)):
            m_vector_new[i] += dif/len(m_vector_new)

        final_total_mass = sum(m_vector_new)
        dif = - ( final_total_mass - m_full )
        print(f"\nFinal total mass: {final_total_mass:.4f} kg")
        print(f"Target mass: {m_full} kg")
        print(f"Difference: {dif:.4f} kg")

        # Создание DataFrame с результатами
        new_data = {
            'step': self.Length_secs,
            'length': L_sec_vector_sum_new,
            'diameter': D_sec_vector_new,
            'area': A_vector_new,
            'mass': m_vector_new,
            'stiffness': s_vector_new,
            'class': Class_sec_vector_new,
            'stage': Stage_sec_vector_new
        }

        df_result = pd.DataFrame(new_data)
        df_result.to_csv('resources/rocket_data.csv', index=False, encoding='utf-8')
        # Добавляем координаты ступеней в возвращаемые данные
        new_data['stage_coordinates'] = self.stage_coordinates
        return new_data

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

        for k in range(self.block_number):
            self.propellant_mass.append(self.block_mass[k] * self.structural_values[k] / (self.structural_values[k] + 1))
            self.mass_ox.append(self.propellant_mass[k] * self.components_ratio / (self.components_ratio + 1))
            self.mass_fu.append(self.propellant_mass[k] / (self.components_ratio + 1))
            delta_mass = self.thrust[k] / self.exhaust_velocity[k]
            self.delta_mass.append(delta_mass)
            self.work_time.append(self.propellant_mass[k] / self.delta_mass[k])
            self.structural_mass.append(self.block_mass[k] - self.propellant_mass[k])
            self.delta_mass_ox.append(delta_mass * self.components_ratio / (self.components_ratio + 1))
            self.delta_mass_fu.append(delta_mass / (self.components_ratio + 1))

        self.full_time = sum(self.work_time)
        print("test")
        m = self.propellant_mass[1]
        t = 0
        while m > 0:
            m -= self.delta_mass[1]*constants.timestep
            t +=constants.timestep
        print(t)
        print(self.work_time[1])

    def _calculate_flight_dynamics(self):
        """Расчет динамических параметров полета"""
        self.maximum_area = max(self.distr_data["area"])
        self.delta_level_ox = []
        self.delta_level_fu = []
        for k in range(self.block_number):
            self.delta_level_ox.append(self.delta_mass_ox[k]/self.oxidizer_density/self.maximum_area)
            self.delta_level_fu.append(self.delta_mass_fu[k]/self.fuel_density/self.maximum_area)

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

        print(stage_separation_times)

        self.mass_vector.append(current_mass)
        self.time_vector.append(time)
        self.thrust_vector.append(self.thrust[0])

        self.static_moment = constants.calculate_static(self.full_mass, self.rocket_length)
        self.inertia_moment = constants.calculate_inertia(self.full_mass, self.rocket_length, self.rocket_length, self.max_diameter)
        print(self.static_moment)
        print(self.inertia_moment)
        self.static_vector.append(self.static_moment)
        self.inertia_vector.append(self.inertia_moment)
        self.center_vector.append(self.rocket_length - self.static_moment/current_mass)
        thrust = 0
        cent = 0
        checkpoint = []
        checkpoint.append(self.full_mass - self.propellant_mass[0])
        checkpoint.append(self.full_mass - self.propellant_mass[0] - self.structural_mass[0] - self.propellant_mass[1])
        prop_test = [0,0]
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
                self.static_moment -=constants.calculate_static (self.delta_mass_fu[i], self.fuel_coordinates[i].end + self.fuel_coordinates[i].end-self.delta_level_fu[i])* constants.timestep
                self.static_moment -=constants.calculate_static (self.delta_mass_ox[i], self.oxidyzer_coordinates[i].end + self.oxidyzer_coordinates[i].end-self.delta_level_ox[i])* constants.timestep
                self.inertia_moment-=constants.calculate_inertia(self.delta_mass_fu[i], self.fuel_coordinates[i].end + self.fuel_coordinates[i].end-self.delta_level_fu[i], self.fuel_coordinates[i].end - self.fuel_coordinates[i].end-self.delta_level_fu[i], self.max_diameter)* constants.timestep
                self.inertia_moment-=constants.calculate_inertia(self.delta_mass_ox[i], self.oxidyzer_coordinates[i].end + self.oxidyzer_coordinates[i].end-self.delta_level_ox[i], self.fuel_coordinates[i].end - self.fuel_coordinates[i].end-self.delta_level_fu[i], self.max_diameter)* constants.timestep
                thrust = self.thrust[i]
                prop_test[current_stage] += self.delta_mass[current_stage] * constants.timestep
                thrust = self.thrust[current_stage]
                cent = self.static_moment / current_mass

            for i in range(self.block_number):
                if not stage_dropped[i] and time >= stage_separation_times[i]:
                    print(self.structural_mass[i])
                    current_mass -= self.structural_mass[i]
                    self.static_moment -=constants.calculate_static (self.structural_mass[i],self.structural_coordinates[i].end + self.structural_coordinates[i].start)
                    self.inertia_moment-=constants.calculate_inertia(self.structural_mass[i],self.structural_coordinates[i].end + self.structural_coordinates[i].start, self.structural_coordinates[i].length, self.max_diameter)
                    stage_dropped[i] = True
                    print(f"Отделена {i+1}-я ступень в t={time} с, расчетное время: {stage_separation_times[i]} c, текущая масса: {current_mass}")

            self.mass_vector.append(current_mass)
            self.time_vector.append(time)
            self.thrust_vector.append(thrust)
            self.center_vector.append(cent)
            time += constants.timestep

    def get_step_length(self):
        self.Length_secs[0] = 0
        return self.Length_secs

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
        for i in range(self.block_number+1):
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

# r = rocket_parser("falcon")
# print(f"Rocket name: {r.name}")

# plt.figure(figsize=(15, 10))

# plt.subplot(2, 1, 1)
# plt.plot(r.distr_data["length"], r.distr_data["mass"], label="Mass distribution", color="green")
# plt.xlabel("Rocket length, m")
# plt.ylabel("Mass, kg")
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(r.distr_data["length"], r.distr_data["stiffness"], label="Stiffness distribution", color="black")
# plt.xlabel("Rocket length, m")
# plt.ylabel("Stiffness")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Дополнительная визуализация динамических параметров
# # plt.figure(figsize=(15, 10))

# plt.subplot(2, 2, 1)
# plt.plot(r.time_vector, r.mass_vector, label="Total mass", color="blue")
# plt.xlabel("Time, s")
# plt.ylabel("Mass, kg")
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 2, 2)
# plt.plot(r.time_vector, r.center_vector, label="Thrust", color="red")
# plt.xlabel("Time, s")
# plt.ylabel("Center position, m")
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 2, 3)
# plt.plot(r.time_vector, r.static_vector, label="Static moment", color="purple")
# plt.xlabel("Time, s")
# plt.ylabel("Static moment")
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 2, 4)
# plt.plot(r.time_vector, r.inertia_vector, label="Moment of inertia", color="orange")
# plt.xlabel("Time, s")
# plt.ylabel("Inertia, kg·m²")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()