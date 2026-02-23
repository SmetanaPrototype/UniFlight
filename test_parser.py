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
    def __init__(self, start, end, length):
        self.start = float(start)
        self.end = float(end)
        self.length = float(length)

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
        m = self.propellant_mass[1]
        t = 0
        while m > 0:
            m -= self.delta_mass[1]*constants.timestep
            t +=constants.timestep

    def _distributed_handler(self, filename):
        df = pd.read_csv(filename)
        Length_start_vector     = df['L']
        Diameter_start_vector   = df['D']
        Class_start_vector      = df['Class']
        Stage_start_vector      = df['Stage']

        # Classification handling
        class_counts = df['Class'].value_counts()
        class_counts_dict = class_counts.to_dict()
        tail_count = class_counts_dict.get("Tail", 0)
        head_count = class_counts_dict.get("Head", 0)

        # self.Length_parts.append(df[df['Stage'] == 'Payload']['L'].sum())
        # self.Length_parts.append(df[df['Stage'] == 'First']['L'].sum())
        # self.Length_parts.append(df[df['Stage'] == 'Second']['L'].sum())

        #Masses
        m_payload = self.payload_mass
        m_fuel = sum(self.mass_fu)
        m_oxidyzer = sum(self.mass_ox)
        m_full = self.full_mass
        m_engine = m_payload/3
        self.max_diameter = max(Diameter_start_vector)
        m_construction = m_full - m_oxidyzer - m_fuel - m_payload - m_engine

        Sum_Length_start_vector = []
        cumlength = 0
        for le in Length_start_vector:
            cumlength+=le
            Sum_Length_start_vector.append(cumlength)

        Length_final_vector     = []
        Diameter_final_vector   = []
        Class_final_vector      = []
        Stage_final_vector      = []
        Stiffness_final_vetor   = []
        Area_final_vector       = []
        Volume_final_vector     = []
        Mass_final_vector       = []

        Length_final_vector     .append(0)
        Diameter_final_vector   .append(0)
        Class_final_vector      .append('Head')
        Stage_final_vector      .append('Payload')
        Stiffness_final_vetor   .append(0)
        Area_final_vector       .append(0)
        Volume_final_vector     .append(0)

        numeric = []
        raised_length = []

        li = 0
        num = 0

        while (li<sum(Length_start_vector)):
            if (li >= constants.lenstep/2): Length_final_vector.append(constants.lenstep)
            raised_length.append(round(li,1))
            numeric.append(num)

            for i in range(len(Sum_Length_start_vector) - 1):
                if li < Sum_Length_start_vector[i+1] and li > Sum_Length_start_vector[i]:
                    Class_final_vector   .append(Class_start_vector[i])
                    Stage_final_vector   .append(Stage_start_vector[i])
                    delta_diameter = Diameter_start_vector[i + 1] - Diameter_start_vector[i]
                    delta_length = Sum_Length_start_vector[i + 1] - Sum_Length_start_vector[i]

                    offset = li - Sum_Length_start_vector[i]

                    if delta_length > 0:
                        current_diameter = Diameter_start_vector[i] + (delta_diameter / delta_length) * offset
                    else:
                        current_diameter = Diameter_start_vector[i]

                    Diameter_final_vector.append(current_diameter)
                    Stiffness_final_vetor.append(constants.calculate_stiffness(Diameter_start_vector[i]))
                    Area_final_vector.append(constants.cross_sectional_area(Diameter_start_vector[i]))
                    Volume_final_vector.append(Area_final_vector[-1]*constants.lenstep)


            li+=constants.lenstep
            num+=1

        def mass_per_class(mass_, classname_):
            return mass_ / sum(Volume_final_vector[i] for i in range(len(Class_final_vector)) if Class_final_vector[i] == classname_)

        m_construction_per = m_construction / sum(Volume_final_vector)
        m_payload_per = mass_per_class(m_payload, 'Head')
        m_oxidizer_per = mass_per_class(m_oxidyzer, 'Oxidizer')
        m_fuel_per = mass_per_class(m_fuel, 'Fuel')
        m_engine_per = mass_per_class(m_engine, 'Tail')

        for j in range(len(Volume_final_vector)):
            m = m_construction_per * Volume_final_vector[j]
            if Class_final_vector[j] == 'Head':
                m += m_payload_per * Volume_final_vector[j]
            elif Class_final_vector[j] == 'Oxidizer':
                m += m_oxidizer_per * Volume_final_vector[j]
            elif Class_final_vector[j] == 'Fuel':
                m += m_fuel_per * Volume_final_vector[j]
            elif Class_final_vector[j] == 'Tail':
                m += m_engine_per * Volume_final_vector[j]
            Mass_final_vector.append(m)

        #zero start

        print("Destr mass:", sum(Mass_final_vector))

        new_data = {
            'numeric':   numeric,
            'step':      Length_final_vector,
            'length':    raised_length,
            'diameter':  Diameter_final_vector,
            'area':      Area_final_vector,
            'mass':      Mass_final_vector,
            'stiffness': Stiffness_final_vetor,
            'class':     Class_final_vector,
            'stage':     Stage_final_vector
        }

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
        df_result.to_csv('output/rocket_data.csv', index=False, encoding='utf-8')

        return new_data

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

    # def changed_mass(self, time_):
    #     mass_t = self.masses.copy()
        
    #     # Сохраняем начальную сумму для проверки
    #     initial_sum = sum(mass_t)
        
    #     if time_ < 1E-7:
    #         return mass_t

    #     # Iteration on blocks
    #     for j in range(block_number):
    #         n_ox = int(sector_range_ox[0].length/constants.lenstep)
    #         n_fu = int(sector_range_fu[0].length/constants.lenstep)

    #         # Working time on block
    #         total_time = work_time[j]

    #         time_per_sector_ox = total_time / n_ox if n_ox > 0 else 0
    #         time_per_sector_fu = total_time / n_fu if n_fu > 0 else 0
            
    #         total_burn_time_ox = 0
    #         total_burn_time_fu = 0
            
    #         # Для каждого сектора считаем, сколько времени он горел
    #         for idx_sector in range(n_ox):
    #             sector_start_time = idx_sector * time_per_sector_ox
    #             sector_end_time = (idx_sector + 1) * time_per_sector_ox
    #             start_idx = int(sector_range_ox[0].start) + idx_sector
    #             end_idx = start_idx + 1
                
    #             # Определяем время горения этого сектора
    #             if current_time >= sector_end_time:
    #                 burn_time = time_per_sector_ox  # Сектор полностью сгорел
    #             elif current_time > sector_start_time:
    #                 burn_time = current_time - sector_start_time  # Сектор горит частично
    #             else:
    #                 burn_time = 0  # Сектор еще не начал гореть
                
    #             total_burn_time_ox += burn_time
                
    #             # Вычитаем массу пропорционально времени горения
    #             for k in range(start_idx, end_idx):
    #                 mass_t[k] = masses[k] - delta_mass_ox[0] * burn_time
    #                 mass_t[k] = max(mass_t[k], 0)
            
    #         # Аналогично для топлива
    #         for idx_sector in range(n_fu):
    #             sector_start_time = idx_sector * time_per_sector_fu
    #             sector_end_time = (idx_sector + 1) * time_per_sector_fu
    #             start_idx = int(sector_range_fu[0].start) + idx_sector
    #             end_idx = start_idx + 1
                
    #             # Определяем время горения этого сектора
    #             if current_time >= sector_end_time:
    #                 burn_time = time_per_sector_fu
    #             elif current_time > sector_start_time:
    #                 burn_time = current_time - sector_start_time
    #             else:
    #                 burn_time = 0
                
    #             total_burn_time_fu += burn_time
                
    #             # Вычитаем массу пропорционально времени горения
    #             for k in range(start_idx, end_idx):
    #                 mass_t[k] = masses[k] - delta_mass_fu[0] * burn_time
    #                 mass_t[k] = max(mass_t[k], 0)
            
    #         print(f"Time: {current_time}")
    #         print(f"Total burn time ox: {total_burn_time_ox}")
    #         print(f"Total burn time fu: {total_burn_time_fu}")
    #         print(f"Expected total burn time: {current_time}")
            
    #         # Выходим после первого блока
    #         break

    #     # Проверка сохранения массы
    #     final_sum = sum(mass_t)
    #     expected_difference = (delta_mass_ox[0] + delta_mass_fu[0]) * current_time
        
    #     # Вычисляем фактическую разницу
    #     actual_difference = initial_sum - final_sum
        
    #     # Проверяем с учетом погрешности вычислений
    #     if abs(actual_difference - expected_difference) > 1e-10:
    #         print(f"Warning: Mass conservation violation!")
    #         print(f"Initial sum: {initial_sum}")
    #         print(f"Final sum: {final_sum}")
    #         print(f"Expected difference: {expected_difference}")
    #         print(f"Actual difference: {actual_difference}")
    #         print(f"Error: {abs(actual_difference - expected_difference)}")
    #         print(f"current_time: {current_time}")
    #         print(f"delta_mass_ox: {delta_mass_ox[0]}")
    #         print(f"delta_mass_fu: {delta_mass_fu[0]}")
    #         quit()

    # return mass_t

rp = rocket_parser("falcon")
print(sum(rp.changed_mass(0)))


