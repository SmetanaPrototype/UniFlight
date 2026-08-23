# print(rp.complex.flight_data.statics)
# plt.plot(rp.complex.flight_data.times, rp.complex.flight_data.centers)
# plt.show()

    # def get_step_length(self):
    #     return self.steps


    # def get_full_time(self):
    #     return self.full_time

    # def get_part_diameters(self):
    #     res = []
    #     for i in range(self.block_number + 1):
    #         res.append(self.diameters[-1])
    #     return res

    # def get_part_length(self):
    #     return self.stages_lengths

    # def get_thrust_from_time(self, time):
    #     return basis.get_y(time,self.time_vector,self.thrust_vector)

    # def get_mass_from_time(self, time):
    #     return basis.get_y(time,self.time_vector,self.mass_vector)

    # def get_inertia_from_time(self, time):
    #     return basis.get_y(time,self.time_vector,self.inertia_vector)

    # def get_center_from_time(self, time):
    #     return basis.get_y(time,self.time_vector,self.center_vector)

    # def get_propellant_from_time(self, time):
    #     return basis.get_y(time,self.time_vector,self.thrust_vector)

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.cm as cm
# from matplotlib.colors import Normalize

# # Создаем объект ракеты
# rp = rocket_parser()

# # Выбираем моменты времени для анализа

# ti = 0
# time_points = []
# while ti < rp.work_time[0]:
#     time_points.append(ti)
#     ti +=5
# # time_points = [0, 10, 20, 30, 40, 50, 90, 100, rp.full_time - 20, rp.full_time - 10, rp.full_time]

# # Создаем цветовую карту (от темного к светлому)
# colors = cm.plasma(np.linspace(0, 0.9, len(time_points)))

# # Создаем фигуру с двумя подграфиками
# fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

# # Получаем координаты по длине ракеты
# x_coords = rp.asc_length

# color_pairs = [
#     ([0.68, 0.85, 0.9], [0, 0, 1]),
#     ([1, 0.68, 0.68], [1, 0, 0]),
#     ([0.8, 0.8, 0.8], [0, 0, 0])
# ]

# for i, t in enumerate(time_points):
#     if t <= rp.full_time + 30:
#         mass_distribution = rp.changed_mass(t)

#         ax1.plot(x_coords, mass_distribution,
#                 color = basis.interpolate_color(color_pairs[0][0], color_pairs[0][1], i, len(time_points)),
#                 linewidth=2,
#                 alpha=0.8,
#                 label=f't = {t:.1f} с')

# ax1.set_xlabel('Длина ракеты, м')
# ax1.set_ylabel('Масса в сечении, кг')
# ax1.set_title('Распределение массы по длине ракеты в разные моменты времени')
# ax1.grid(True, alpha=0.3)

# plt.show()

# print(rp.main)
# print(rp.boost)
# print(rp.work_time)

        # Classification handling
        # class_counts = df["Class"].value_counts()
        # class_counts_dict = class_counts.to_dict()
        # tail_count = class_counts_dict.get("Tail", 0)
        # head_count = class_counts_dict.get("Head", 0)

        # self.stages_lengths.append(df[df['Stage'] == 'Payload']['L'].sum())
        # self.stages_lengths.append(df[df['Stage'] == 'First']['L'].sum())
        # self.stages_lengths.append(df[df['Stage'] == 'Second']['L'].sum())

    # # def effective_mass(self, time_, mode_index):

    # #     f1lev = self.fuel_coordinates[0].length
    # #     f2lev = self.fuel_coordinates[1].length
    # #     o1lev = self.oxidyzer_coordinates[0].length
    # #     o2lev = self.oxidyzer_coordinates[1].length

    # #     if   mode_index == 0: E = 1.841
    # #     elif mode_index == 1: E = 3.054
    # #     elif mode_index == 2: E = 4.201

    # #     first_stage_end = self.work_time[0]

    # #     mass_total = self.changed_mass(time_)
    # #     mass_effective = mass_total.copy()
    # #     for i in range(len(mass_effective)):
    # #         s = 0
    # #         h = 0
    # #         if self.classes[i] in ["Fuel"] and self.stages[i] in ["First"]:
    # #             h = f1lev * (first_stage_end - time_)/first_stage_end
    # #             s = self.structural_values[0]
    # #         elif self.classes[i] in ["Oxidizer"] and self.stages[i] in ["First"]:
    # #             h = o1lev * (first_stage_end - time_)/first_stage_end
    # #             s = self.structural_values[0]
    # #         elif self.classes[i] in ["Fuel"] and self.stages[i] in ["Second"]:
    # #             h = f2lev
    # #             s = self.structural_values[1]
    # #         elif self.classes[i] in ["Oxidizer"] and self.stages[i] in ["Second"]:
    # #             h = o2lev
    # #             s = self.structural_values[1]

    # #         if s!=0:
    # #             construction_ratio = 1/s
    # #             construction_mass = mass_effective[i] * construction_ratio
    # #             fluid_mass = mass_effective[i] - construction_mass
    # #             mass_effective[i] = construction_mass + fluid_mass*self.max_diameter/(E*h)*np.tanh(E*2*h/self.max_diameter)/(E*E-1)

    # #     return mass_effective

    # def effective_mass(self, time_, mode_index):
    #     mass_total = self.changed_mass(time_)
    #     mass_effective = mass_total.copy()

    #     # first_stage_end = self.work_time[0]
    #     # for i in range(len(mass_effective)):
    #     #     if self.classes[i] in ["Fuel", "Oxidizer"]:
    #     #         construction_ratio = 1/self.structural_values[0]
    #     #         construction_mass = mass_total[i] * construction_ratio
    #     #         fluid_mass = mass_total[i] - construction_mass

    #     #         if mode_index == 0: E = 1
    #     #         if mode_index == 1: E = 1
    #     #         if mode_index == 2: E = 1

    #     #         if fluid_mass > 0:  
    #     #             mass_effective[i] = construction_mass + E * fluid_mass

    #     return mass_effective

            # checkpoint = []
        # checkpoint.append(group.full_mass - group.propellant_mass[0])
        # checkpoint.append(
        #     self.full_mass
        #     - self.propellant_mass[0]
        #     - self.structural_mass[0]
        #     - self.propellant_mass[1]
        # )

    #             # boost condition
    #             if current_stage == 0 and self.rocket_type != "tandem":
    #                 thrust+=sum(self.boost.thrust)
    #             cent = self.static_moment / current_mass