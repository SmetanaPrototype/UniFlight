import rocket_parser as rp
import basis
import atmosphere as atmo
import aerodynamics as aero
import attack
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Глобальные списки для записи данных
Cbs_list = []
Cyw_list = []
Cww_list = []
Cyy_list = []
Cwy_list = []
Cwb_list = []
attack_list = []
vel_list = []
traj_list = []
alt_list = []
time_list = []
wind_list = []
Q_list = []
thrust_list = []
aog_list = []
mass_list = []
acceleration_list = []
CswQ_list = []
CsyQ_list = []
CssQ_list = []
mach_list = []

Csw_list = [[] for _ in range(basis.mode_num)]
Csy_list = [[] for _ in range(basis.mode_num)]
Csb_list = [[] for _ in range(basis.mode_num)]

parser = rp.rocket_parser()
rocketname = basis.current_rocket

#Загрузка данных жесткости
f_stiffness = [0] * basis.mode_num
f_stiffness_diff = [0] * basis.mode_num

oscillations_file = "output/"+rocketname+"_oscillations.csv"
f_stiffness[0] = basis.read_array_from_csv(oscillations_file, "form_1")
f_stiffness[1] = basis.read_array_from_csv(oscillations_file, "form_2")
f_stiffness[2] = basis.read_array_from_csv(oscillations_file, "form_3")
f_stiffness_diff[0] = basis.read_array_from_csv(oscillations_file, "difform_1")
f_stiffness_diff[1] = basis.read_array_from_csv(oscillations_file, "difform_2")
f_stiffness_diff[2] = basis.read_array_from_csv(oscillations_file, "difform_3")
coord_stiffness = basis.read_array_from_csv(oscillations_file, "length")

freq_file = "output/"+rocketname+"_frequency.csv"
freq_time = basis.read_array_from_csv(freq_file, "time")
freq_mass = basis.read_array_from_csv(freq_file, "freq_mass_1")

class ballistics:
    def __init__(self, N, Y, vel, alt, parser, get_attack_func, is_landing):
        self.N = N
        self.Y = Y
        self.vel = vel
        self.alt = alt
        self.parser = parser
        self.get_attack_func = get_attack_func

        self.G = aero.UnionStream()
        self.G.set_elnumber(parser.get_block_number() + 1)
        self.G.set_diameter(parser.get_part_diameters())
        self.G.set_length(parser.get_part_length())

        self.thrust = 0
        self.mass = 0
        self.inertia = 0
        self.attack = 0
        self.dencity = 0
        self.dypressure = 0
        self.first_point = 0

        self.atm = atmo.atmosphere(self.alt)
        self.last_time = None

    def update_params(self, time):
        print(round(time / parser.full_time * 100), "%", end="\r")
        if self.last_time != time:
            self.thrust = self.parser.get_thrust_from_time(time)
            # if time > parser.work_time[0]:
            #     self.thrust = 0
            # if time > 400 and time < 440:
            #     self.thrust = parser.thrust[0] * 0.2
            self.mass = self.parser.get_mass_from_time(time)
            self.inertia = self.parser.get_inertia_from_time(time)
            self.center = self.parser.get_center_from_time(time)

            if self.center is None:
                self.center = 0.0
            if self.thrust is None:
                self.thrust = 0.0
            if self.mass is None:
                self.mass = 0.1
            if self.inertia is None:
                self.inertia = 0.1

            self.attack = np.radians(self.get_attack_func(self.vel, time))
            self.G.calculate_CXY(self.vel, self.alt, self.attack)
            # if time > parser.work_time[0]:
            #     self.G.CY*=1.5
            # self.attack += 10000*aeroelasticity2.aero_attack(self.G.CY, self.dypressure, time)
            # self.G.calculate_CXY(self.vel, self.alt, self.attack)
            self.atm = atmo.atmosphere(self.alt)
            self.dencity = self.atm.get_density()
            self.wind = self.atm.get_wind()

            if self.alt < 100:
                sv = self.atm.get_SV()
                mach_list.append(self.vel / sv)

            focus_pos = (
                self.G.focus_position
                if hasattr(self.G, "focus_position")
                and self.G.focus_position is not None
                else 0.0
            )
            self.first_point = abs(focus_pos - self.center)
            self.second_point = abs(self.parser.rocket_length - self.center)

            if self.alt > 90000:
                self.G.CX = 0
                self.G.CY = 0
                self.dencity = 0

            if self.alt < 0:
                self.alt = 0

            self.last_time = time
            self.dypressure = self.dencity * self.vel**2 / 2

            # Запись данных
            attack_list.append(self.attack * 180 / np.pi)
            vel_list.append(self.vel)
            traj_list.append(self.Y * 180 / np.pi)
            alt_list.append(self.alt)
            time_list.append(time)
            wind_list.append(self.wind)
            Q_list.append(self.dypressure)
            thrust_list.append(self.thrust)
            aog_list.append(self.atm.get_AOG())
            mass_list.append(self.mass)

            Cbs_list.append(-self.thrust * self.parser.thrust_ratio / self.mass)
            Cyw_list.append(-(self.thrust + self.G.CY * self.dypressure * self.parser.maximum_area) / self.mass)
            Cww_list.append((-self.G.CY * self.dypressure * self.parser.maximum_area * self.first_point) / self.inertia)
            Cyy_list.append((self.G.CY * self.dypressure * self.parser.maximum_area) / (self.mass * self.vel))
            Cwy_list.append((self.G.CY * self.dypressure * self.parser.maximum_area * self.first_point) / self.inertia / self.vel)
            Cwb_list.append(self.thrust * self.parser.thrust_ratio * self.second_point / self.inertia)

            for k in range(basis.mode_num):
                Csw_list[k].append(self.thrust * self.parser.thrust_ratio / self.inertia *
                                  ((self.second_point - self.parser.rocket_length) * f_stiffness_diff[k][-1] + f_stiffness[k][-1]))
                Csy_list[k].append(self.thrust * self.parser.thrust_ratio * f_stiffness_diff[k][-1] / self.mass)
                Csb_list[k].append(self.thrust * self.parser.thrust_ratio / self.inertia * f_stiffness[k][-1])

            stream_ratio = -self.parser.maximum_area * self.dypressure
            cy_integral = [0, 0, 0]
            delta_stiffness = 1
            mass_s = basis.get_y(time, freq_time, freq_mass)

            for f, scoord in enumerate(coord_stiffness):
                dcy = self.G.get_cya_from_coord(scoord)
                cy_integral[0] += dcy * f_stiffness_diff[0][f] * (scoord - self.second_point) / delta_stiffness
                cy_integral[1] += dcy * f_stiffness_diff[0][f] / delta_stiffness
                cy_integral[2] += dcy * f_stiffness[0][f] * f_stiffness_diff[0][f] / delta_stiffness

            CswQ_list.append(stream_ratio / self.inertia * cy_integral[0])
            CsyQ_list.append(stream_ratio / self.mass * cy_integral[1])
            CssQ_list.append(stream_ratio / mass_s * cy_integral[2])

    def delta_velocity(self, time):
        self.update_params(time)
        F_P = self.thrust * np.cos(self.attack)
        F_X = self.G.CX * self.dypressure * self.parser.maximum_area
        res = (F_P - F_X) / self.mass - self.atm.get_AOG() * np.sin(self.Y)
        acceleration_list.append(res)
        return res

    def delta_trajangle(self, time):
        self.update_params(time)
        F_P = self.thrust * np.sin(self.attack)
        F_Y = self.G.CY * self.dypressure * self.parser.maximum_area
        F_G = (
            self.atm.get_AOG()
            * np.cos(self.Y)
            * (
                1
                - self.vel**2
                / (self.atm.get_AOG() * (basis.earth_radius + self.alt))
            )
        )
        return (F_P + F_Y) / (self.mass * self.vel) - F_G / self.vel

    def delta_polar(self, time):
        self.update_params(time)
        return (self.vel / (basis.earth_radius + self.alt)) * np.cos(self.Y)

    def delta_altitude(self, time):
        self.update_params(time)
        return self.vel * np.sin(self.Y)

    def delta_longitude(self, time):
        self.update_params(time)
        return self.vel * np.cos(self.Y)


def output(parser):
    """Сохранение результатов в файл"""
    rocketname = parser.name
    basis.write_arrays_to_csv(
        "output/" + rocketname + "_dynamic_coefs.csv",
        time=time_list,
        wind=wind_list,
        velocity=vel_list,
        Cbs=Cbs_list,
        Cyw=Cyw_list,
        Cww=Cww_list,
        Cyy=Cyy_list,
        Cwy=Cwy_list,
        Cwb=Cwb_list,
        Csw1=Csw_list[0],
        Csw2=Csw_list[1],
        Csw3=Csw_list[2],
        Csy1=Csy_list[0],
        Csy2=Csy_list[1],
        Csy3=Csy_list[2],
        Csb1=Csb_list[0],
        Csb2=Csb_list[1],
        Csb3=Csb_list[2],
        CswQ=CswQ_list,
        CsyQ=CsyQ_list,
        CssQ=CssQ_list,
    )


def to_reverse(parser):
    """Сохранение результатов в файл с интервалом ~1 секунда"""
    rocketname = parser.name

    # Прореживание массивов до ~1 секунды
    time_reduced = []
    mass_reduced = []
    acceleration_reduced = []
    thrust_reduced = []
    q_reduced = []
    g_reduced = []
    tetta_reduced = []
    vel_reduced = []

    # Интервал округления (секунды)
    interval = 1.0

    last_saved_time = -interval  # чтобы первая точка сохранилась

    for i in range(len(time_list)):
        current_time = time_list[i]

        # Сохраняем, если прошло достаточно времени или это последняя точка
        if current_time - last_saved_time >= interval - 1e-9 or i == len(time_list) - 1:
            time_reduced.append(current_time)
            mass_reduced.append(mass_list[i])
            acceleration_reduced.append(acceleration_list[i])
            thrust_reduced.append(thrust_list[i])
            q_reduced.append(Q_list[i])
            g_reduced.append(aog_list[i])
            tetta_reduced.append(traj_list[i])
            vel_reduced.append(vel_list[i])
            last_saved_time = current_time

    basis.write_arrays_to_csv(
        "output/" + rocketname + "_ball_data.csv",
        time=time_reduced,
        vel=vel_reduced,
        mass=mass_reduced,
        acceleration=acceleration_reduced,
        thrust=thrust_reduced,
        q=q_reduced,
        g=g_reduced,
        tetta=tetta_reduced,
    )

    print(f"Сжатие данных: {len(time_list)} -> {len(time_reduced)} точек")

def system(t, vars, parser, get_attack_func):
    """Система дифференциальных уравнений"""
    n, y, v, h, l = vars
    b = ballistics(n, y, v, h, parser, get_attack_func, False)
    return [
        b.delta_polar(t),
        b.delta_trajangle(t),
        b.delta_velocity(t),
        b.delta_altitude(t),
        b.delta_longitude(t),
    ]

def main():
    print("ЗАПУСК БАЛЛИСТИЧЕСКОГО РАСЧЕТА")
    print("=" * 60)
    print(f"Ракета: {rocketname}")
    print(f"Коэффициенты управления: {parser.attack_coefs}")
    print("=" * 60)

    # Начальные условия
    ft = parser.get_full_time()
    h = basis.timestep
    t_span = (0, min(ft - 1, 800))
    y0 = [0, np.pi / 2, 10.0, 100.0, 0.1]

    def fall_event(t, y, parser, get_attack_func):
        return y[3]

    fall_event.terminal = True
    fall_event.direction = -1

    # Запуск симуляции
    sol = solve_ivp(
        system,
        t_span,
        y0,
        method="RK45",
        max_step=h,
        args=(parser, parser.attack_func),
        events=fall_event,
        rtol=1e-6,
        atol=1e-8,
    )

    if sol.success:
        final_velocity = sol.y[2][-1]
        final_altitude = sol.y[3][-1]
        final_longitude = sol.y[4][-1]
        final_angle = sol.y[1][-1] * 180 / np.pi
        final_attack = attack_list[-1] if attack_list else 0

        print("\n=== РЕЗУЛЬТАТЫ РАСЧЕТА ===")
        print(f"Конечная скорость: {final_velocity:.2f} м/с")
        print(f"Конечная высота: {final_altitude/1000:.2f} км")
        print(f"Конечная дальность: {final_longitude/1000:.2f} км")
        print(f"Конечный угол траектории: {final_angle:.2f}°")
        print(f"Конечный угол атаки: {final_attack:.2f}°")
        print(f"Максимальный угол атаки: {max(attack_list):.2f}°")

        output(parser)
        to_reverse(parser)
    else:
        print("❌ Ошибка при расчете траектории")

    return sol


# Запуск расчета
sol = main()

# Построение графиков
if time_list:
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(time_list, attack_list, label="Угол атаки α(t)", color="red")
    plt.xlabel("Время, с")
    plt.ylabel("Угол атаки, градусы")
    plt.title("Угол атаки по времени")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time_list, alt_list, label="Высота h(t)", color="blue")
    plt.xlabel("Время, с")
    plt.ylabel("Высота, м")
    plt.title("Высота по времени")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time_list, vel_list, label="Скорость v(t)", color="green")
    plt.xlabel("Время, с")
    plt.ylabel("Скорость, м/с")
    plt.title("Скорость по времени")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(time_list, traj_list, label="Угол траектории θ(t)", color="purple")
    plt.xlabel("Время, с")
    plt.ylabel("Угол траектории, градусы")
    plt.title("Угол траектории по времени")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
