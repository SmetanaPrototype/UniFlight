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
        self._distributed_handler(csv_filename)

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

        self.Length_parts.append(df[df['Stage'] == 'Payload']['L'].sum())
        self.Length_parts.append(df[df['Stage'] == 'First']['L'].sum())
        self.Length_parts.append(df[df['Stage'] == 'Second']['L'].sum())

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

        numeric = []
        raised_length = []

        li = 0
        num = 0

        while (li<sum(Length_start_vector)):
            Length_final_vector.append(constants.lenstep)
            raised_length.append(li)
            numeric.append(num)

            for i in range(len(Sum_Length_start_vector) - 1):
                if li < Sum_Length_start_vector[i+1] and li > Sum_Length_start_vector[i]:
                    Class_final_vector   .append(Class_start_vector[i])
                    Stage_final_vector   .append(Stage_start_vector[i])
                    Diameter_final_vector.append(Diameter_start_vector[i])
                    Stiffness_final_vetor.append(constants.calculate_stiffness(Diameter_start_vector[i]))
                    Area_final_vector.append(constants.cross_sectional_area(Diameter_start_vector[i]))
                    Volume_final_vector.append(Area_final_vector[-1]*constants.lenstep)


            li+=constants.lenstep
            num+=1

        m_construction_per = m_construction / sum(Volume_final_vector)
        m_payload_per = m_payload / sum(Volume_final_vector[i] for i in range(len(Stage_final_vector)) if Stage_final_vector[i] == 'Payload')

        for j in range(len(Volume_final_vector)):
            m = m_construction_per * Volume_final_vector[j]
            if Stage_final_vector[j] == 'Payload':
                m += m_payload_per * Volume_final_vector[j]
            Mass_final_vector.append(m)

        print(len(Class_final_vector))
        #zero start
        Length_final_vector[0]     = 0
        Diameter_final_vector[0]   = 0
        Class_final_vector[0]      = ['Head']
        Stage_final_vector[0]      = ['Payload']
        Stiffness_final_vetor[0]   = 0
        Area_final_vector[0]       = 0

        # new_data = {
        #     'step': self.Length_secs,
        #     'length': L_sec_vector_sum_new,
        #     'diameter': D_sec_vector_new,
        #     'area': A_vector_new,
        #     'mass': m_vector_new,
        #     'stiffness': s_vector_new,
        #     'class': Class_sec_vector_new,
        #     'stage': Stage_sec_vector_new
        # }

        # df_result = pd.DataFrame(new_data)
        # df_result.to_csv('output/rocket_data.csv', index=False, encoding='utf-8')

rp = rocket_parser("falcon")


