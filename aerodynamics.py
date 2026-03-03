import math
from matplotlib.cm import Reds
import matplotlib.pyplot as plt
import pandas as pd
import atmosphere
import path
import rocket_parser as rp
import numpy as np
import constants

class Element:
    def __init__(self):
        self.upper_diameter = 0.0
        self.lower_diameter = 0.0
        self.elem_length = 0.0
        self.virtual_length = 0.0
        self.ratio = 0.0
        self.round_area = 0.0
        self.upper_area = 0.0
        self.lower_area = 0.0
        self.base_line = 0.0

        self.C_fric = 0.0
        self.C_pres = 0.0
        self.C_ind = 0.0
        self.CX = 0.0
        self.CY = 0.0
        self.CYY = 0.0
        self.focus_position = 0.0


class Geometry:
    PI = math.pi

    def __init__(self):
        self.elem = []
        self.full_length = 0.0
        self.full_round_area = 0.0
        self.full_ratio = 0.0
        self.midel_diameter = 0.0
        self.midel_area = 0.0
        self.cif = 0.0
        self.num = 0.0

    def set_elnumber(self, n):
        self.elem = [Element() for _ in range(n)]

    def set_length(self, lengths):
        for i, l in enumerate(lengths):
            self.elem[i].elem_length = l
        self.pre_calculations()

    def set_diameter(self, diameters):
        for i, d in enumerate(diameters):
            if i == 0:
                self.elem[i].upper_diameter = 0.0
                self.elem[i].lower_diameter = d
            else:
                self.elem[i].upper_diameter = diameters[i - 1]
                self.elem[i].lower_diameter = d

    def pre_calculations(self):
        self.full_length = 0.0
        self.full_round_area = 0.0

        current_position = 0.0
        for i, e in enumerate(self.elem):
            if i == 0:
                e.base_line = (
                    2
                    * self.PI
                    * math.sqrt(e.elem_length**2 + (e.upper_diameter / 2) ** 2)
                )
                e.round_area = (
                    2 * self.PI * e.elem_length * e.upper_diameter / 2
                    + self.PI * (e.elem_length**2)
                )
                e.focus_position = current_position + e.elem_length * 0.67
            else:
                e.base_line = math.sqrt(
                    e.elem_length**2 + ((e.lower_diameter - e.upper_diameter) ** 2) / 4
                )
                e.round_area = (
                    self.PI * (e.upper_diameter + e.lower_diameter) * e.base_line / 2
                )
                e.focus_position = current_position + e.elem_length / 2

            e.upper_area = self.PI * (e.upper_diameter / 2) ** 2
            e.lower_area = self.PI * (e.lower_diameter / 2) ** 2

            try:
                if e.upper_diameter != 0:
                    ratio = e.lower_diameter / e.upper_diameter
                    denominator = ratio - 1

                    if abs(denominator) > 1e-10:
                        e.virtual_length = e.elem_length + e.elem_length / denominator
                    else:
                        e.virtual_length = e.elem_length
                else:
                    e.virtual_length = e.elem_length
            except Exception as e:
                e.virtual_length = e.elem_length

            self.full_length += e.elem_length
            self.full_round_area += e.round_area
            current_position += e.elem_length

            if e.upper_diameter < e.lower_diameter and abs(e.upper_diameter) > 0.1:
                e.ratio = e.virtual_length / e.lower_diameter
            else:
                if e.lower_diameter != 0:
                    e.ratio = e.elem_length / e.lower_diameter
                else:
                    e.ratio = 0

        if self.elem[-1].upper_diameter != 0:
            self.full_ratio = self.full_length / self.elem[-1].upper_diameter
        else:
            self.full_ratio = 0

        self.midel_diameter = self.elem[-1].upper_diameter
        self.midel_area = self.elem[-1].lower_area


class Friction(Geometry):
    def __init__(self):
        super().__init__()
        self.h_s = 8e-6
        self.area_ratio = 0.0
        self.Re = 0.0
        self.n = 0.0
        self.x_t = 0.0
        self.C_fric = 0.0

    def stream_calc(self, Re, Mach):
        if Re <= 485000:
            if abs(Re) < 0.0001:
                self.cif = 0
            else:
                self.cif = 2.656 / math.sqrt(Re)
            self.num = pow(1 + 0.1 * pow(Mach, 0.1), -0.125)
        elif Re < 10000000:
            try:
                log_arg = (self.h_s / self.full_length * Re) - 1
                denominator = 2.2 + 0.08 * pow(Mach, 2) / (1 + 0.312 * pow(Mach, 2))
                self.n = 5 + (1.3 + 0.6 * Mach * (1 - 0.25 * pow(Mach, 2))) * math.sqrt(
                    1 - pow(math.log10(log_arg) / denominator, 2)
                )
            except (ValueError, ZeroDivisionError):
                self.n = 5

            self.x_t = min(
                pow(10, self.n) / Re, self.elem[0].elem_length / self.full_length
            )
            if self.x_t >= 1:
                self.cif = (
                    0.91
                    / pow(math.log10(Re), 2.58)
                    * pow(
                        1 - self.x_t + 40 * pow(self.x_t, 0.625) / pow(Re, 0.375), 0.8
                    )
                )
            else:
                if abs(Re) < 0.0001:
                    self.cif = 0
                else:
                    self.cif = 2.656 / math.sqrt(Re)
            self.num = pow(1 + 0.1 * pow(Mach, 0.1), -2 / 3)
        else:
            self.cif = 0.91 / pow(math.log10(Re), 2.58)
            self.num = pow(1 + 0.1 * pow(Mach, 0.1), -2 / 3)

    def fricalc(self, Mach, SS, nu):
        if nu == None:
            nu = 1.789e-05
        if self.midel_area == 0:
            self.area_ratio = 0
        else:
            self.area_ratio = self.full_round_area / self.midel_area
        self.Re = SS * Mach * self.full_length / nu if nu != 0 else 0
        self.stream_calc(self.Re, Mach)

        for e in self.elem:
            e.C_fric = e.round_area * self.cif * self.num / 2

        self.C_fric = self.area_ratio * self.cif * self.num / 2
        return self.C_fric


class Pressure(Geometry):
    def read_pressure_file(self, filename, rows, cols):
        df = pd.read_csv(filename, delimiter=None, engine="python", header=None)
        if rows > 0:
            df = df.head(rows)

        if df.shape[1] < cols:
            raise ValueError(f"File {filename} has fewer than {cols} columns")

        df = df.iloc[:, :cols]
        data = df.values.tolist()
        transposed = list(map(list, zip(*data)))
        return transposed

    def interpolate_Mach(self, Mach, data):
        Mach_v = data[0]
        values = data[1]
        for i in range(1, len(Mach_v)):
            if Mach >= Mach_v[i - 1] and Mach < Mach_v[i]:
                return values[i - 1] + (Mach - Mach_v[i - 1]) * (
                    values[i] - values[i - 1]
                ) / (Mach_v[i] - Mach_v[i - 1])
        return values[-1]

    def select_ratio_data_pressure(self, ratio, data):
        # data: [Mach_v, H_0, H_025, H_05, H_1, H_2, H_25, H_3, H_4, H_5]
        if ratio < 0.25:
            return [data[0], data[1]]
        elif ratio < 0.5:
            return [data[0], data[2]]
        elif ratio < 1:
            return [data[0], data[3]]
        elif ratio < 2:
            return [data[0], data[4]]
        elif ratio < 2.5:
            return [data[0], data[5]]
        elif ratio < 3:
            return [data[0], data[7]]
        elif ratio < 4:
            return [data[0], data[8]]
        else:
            return [data[0], data[9]]

    def select_ratio_data_triangle(self, ratio, data):
        if ratio >= 1.5 and ratio < 2:
            return [data[0], data[1]]
        elif ratio >= 2 and ratio < 2.5:
            return [data[0], data[2]]
        elif ratio >= 2.5 and ratio < 3:
            return [data[0], data[3]]
        elif ratio >= 3 and ratio < 4:
            return [data[0], data[4]]
        elif ratio >= 4:
            return [data[0], data[5]]
        else:
            return [data[0], data[1]]

    def bottom_pres(self, Mach):
        if Mach < 1:
            if self.cif * self.num * self.full_ratio == 0:
                return 0
            return 0.0155 / math.sqrt(self.cif * self.num * self.full_ratio)
        else:
            data = self.read_pressure_file(path.root_path + "HeadPressure.csv", 10, 10)
            # 7-й столбец (индекс 6) - как в исходнике
            return self.interpolate_Mach(Mach, [data[0], data[6]])

    def head_Cpres(self, Mach):
        data = self.read_pressure_file(path.root_path + "HeadPressure.csv", 10, 10)
        ratio = self.elem[0].ratio
        H_current = self.select_ratio_data_pressure(ratio, data)
        return self.interpolate_Mach(Mach, H_current)

    def triangle_Cpres(self, Mach, ratio):
        data = self.read_pressure_file(path.root_path + "TrianglePressure.csv", 10, 7)
        H_current = self.select_ratio_data_triangle(ratio, data)
        return self.interpolate_Mach(Mach, H_current)

    def prescalc(self, Mach):
        res = 0.0
        for i in range(1, len(self.elem)):
            self.elem[i].C_pres = self.triangle_Cpres(Mach, self.elem[i].ratio) * (
                1 - self.elem[i].upper_area / self.elem[i].lower_area
            )
            res += self.elem[i].C_pres
        return res + self.head_Cpres(Mach) + self.bottom_pres(Mach)


class Inductance(Geometry):
    def read_pressure_file(self, filename, rows, cols):
        data = []
        with open(filename, "r") as f:
            for _ in range(rows):
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) < cols:
                    raise ValueError(
                        f"File {filename} line {_+1} has fewer than {cols} columns"
                    )
                data.append([float(x) for x in parts[:cols]])
        transposed = list(map(list, zip(*data)))
        return transposed

    def E_pressure(self, angle, Mach):
        N = 10
        data = self.read_pressure_file(path.root_path + "EPressure.csv", N, 3)
        Mach_v = data[0]
        H_head = data[1]
        H_cone = data[2]

        if Mach < 1:
            Mach_val = (
                -math.sqrt(1 - constants.sqr(Mach)) / self.elem[0].ratio
                if self.elem[0].ratio != 0
                else 0
            )
        else:
            Mach_val = (
                math.sqrt(constants.sqr(Mach) - 1) / self.elem[0].ratio
                if self.elem[0].ratio != 0
                else 0
            )

        E_head = 0
        E_cone = 0
        for i in range(1, N):
            if Mach_val >= Mach_v[i - 1] and Mach_val < Mach_v[i]:
                E_head = H_head[i - 1] + (Mach_val - Mach_v[i - 1]) * (
                    H_head[i] - H_head[i - 1]
                ) / (Mach_v[i] - Mach_v[i - 1])
                E_cone = H_cone[i - 1] + (Mach_val - Mach_v[i - 1]) * (
                    H_cone[i] - H_cone[i - 1]
                ) / (Mach_v[i] - Mach_v[i - 1])

        self.elem[0].C_ind = (self.elem[0].CY + constants.rad(2 * E_head)) * constants.sqr(angle)
        for j in range(1, len(self.elem)):
            if self.elem[j].upper_diameter < self.elem[j].lower_diameter:
                ratio = self.elem[-1].upper_area if self.elem[-1].upper_area != 0 else 1
                self.elem[j].C_ind = (
                    self.elem[j].CY * self.elem[j].upper_area / ratio
                    + constants.rad(2 * E_cone * self.elem[j].upper_area / ratio)
                ) * constants.sqr(angle)

        E = sum(e.C_ind for e in self.elem)
        return E


class LiftForce(Inductance):
    def sqr(self, x):
        return x * x

    def rad(self, x):
        return x / 57.3

    def head_lift(self, Mach):
        N = 9
        data = self.read_pressure_file(path.root_path + "HeadNormal.csv", N, 6)
        Mah_v = data[0]
        H_0 = data[1]
        H_05 = data[2]
        H_1 = data[3]
        H_2 = data[4]
        H_4 = data[5]

        L_cyl = 0.0
        for j in range(1, len(self.elem)):
            if self.elem[j].upper_diameter < self.elem[j].lower_diameter:
                if self.elem[j].lower_diameter != 0:
                    L_cyl /= self.elem[j].lower_diameter
                break
            else:
                L_cyl += self.elem[j].elem_length

        ratio = L_cyl / self.elem[0].ratio if self.elem[0].ratio != 0 else 0

        H_current = []
        for i in range(N):
            if 0 <= ratio < 0.5:
                H_current.append(H_0[i])
            elif 0.5 <= ratio < 1:
                H_current.append(H_05[i])
            elif 1 <= ratio < 2:
                H_current.append(H_1[i])
            elif 2 <= ratio < 4:
                H_current.append(H_2[i])
            elif ratio >= 4:
                H_current.append(H_4[i])
            else:
                H_current.append(H_0[i])

        if Mach < 1:
            Mach_val = (
                -math.sqrt(1 - constants.sqr(Mach)) / self.elem[0].ratio
                if self.elem[0].ratio != 0
                else 0
            )
        else:
            Mach_val = (
                math.sqrt(constants.sqr(Mach) - 1) / self.elem[0].ratio
                if self.elem[0].ratio != 0
                else 0
            )

        C_head = 0.035
        for i in range(1, N):
            if Mach_val >= Mah_v[i - 1] and Mach_val < Mah_v[i]:
                C_head = H_current[i - 1] + (Mach_val - Mah_v[i - 1]) * (
                    H_current[i] - H_current[i - 1]
                ) / (Mah_v[i] - Mah_v[i - 1])

        self.elem[0].CY = C_head
        return C_head

    def free_triangle_lift(self, index):
        arg = (
            self.elem[index].lower_diameter
            / 2
            / (self.elem[index].virtual_length - self.elem[index].elem_length)
            if (self.elem[index].virtual_length - self.elem[index].elem_length) != 0
            else 0
        )
        Q = math.atan(arg)
        return (2 / 57.3) * constants.sqr(math.cos(Q))

    def triangle_lift(self, Mach, ratio, index):
        N = 9
        data = self.read_pressure_file(path.root_path + "TriangleNormal.csv", N, 6)
        Mah_v = data[0]
        H_0 = data[1]
        H_1 = data[2]
        H_2 = data[3]
        H_3 = data[4]
        H_4 = data[5]

        L_cyl = 0.0
        for j in range(index + 1, len(self.elem)):
            if self.elem[j].upper_diameter < self.elem[j].lower_diameter:
                if self.elem[j].lower_diameter != 0:
                    L_cyl /= self.elem[j].lower_diameter
                break
            else:
                L_cyl += self.elem[j].elem_length

        ratio_new = L_cyl / ratio if ratio != 0 else 0

        H_current = []
        for i in range(N):
            if 0 < ratio_new < 1:
                H_current.append(H_0[i])
            elif 1 <= ratio_new < 2:
                H_current.append(H_1[i])
            elif 2 <= ratio_new < 3:
                H_current.append(H_2[i])
            elif 3 <= ratio_new < 4:
                H_current.append(H_3[i])
            elif ratio_new >= 4:
                H_current.append(H_4[i])
            else:
                H_current.append(H_0[i])

        if Mach < 1:
            Mach_val = -math.sqrt(1 - constants.sqr(Mach)) / ratio if ratio != 0 else 0
        else:
            Mach_val = math.sqrt(constants.sqr(Mach) - 1) / ratio if ratio != 0 else 0

        C_head = 0.0
        for i in range(1, N):
            if Mach_val >= Mah_v[i - 1] and Mach_val < Mah_v[i]:
                C_head = H_current[i - 1] + (Mach_val - Mah_v[i - 1]) * (
                    H_current[i] - H_current[i - 1]
                ) / (Mah_v[i] - Mah_v[i - 1])

        return C_head

    def calculate_CY(self, Mach):
        return self.head_lift(Mach) + self.un_triangle_lift(Mach)

    def un_triangle_lift(self, Mach):
        res = 0.0
        for i in range(1, len(self.elem)):
            if self.elem[i].upper_area < self.elem[i].lower_area:
                big_rat = self.elem[i].ratio
                S_rat = (
                    self.elem[i].upper_area / self.elem[i].lower_area
                    if self.elem[i].lower_area != 0
                    else 0
                )
                self.elem[i].CY = (
                    self.triangle_lift(Mach, big_rat, i)
                    - self.free_triangle_lift(i) * S_rat
                )
            else:
                self.elem[i].CY = constants.rad(2)
            res += (
                self.elem[i].CY * self.elem[i].upper_area / self.elem[-1].upper_area
                if self.elem[-1].upper_area != 0
                else 0
            )
        return res


class DragForce(Friction, Pressure):
    def calculate_CX(self, Mach, SS, nu):
        return self.fricalc(Mach, SS, nu) + self.prescalc(Mach)


class UnionStream(DragForce, LiftForce):
    def __init__(self):
        super().__init__()
        self.E = 0.0
        self.CX = 0.0
        self.CY = 0.0
        self.focus_position = 0.0
        self.focus_relative = 0.0
        self.liftprofile = []
        self.lengthprofile = []

    def calculate_aerodynamic_focus(self, velocity, altitude, attack_angle):

        A = atmosphere.atmosphere(altitude)
        if A.get_SV() is None:
            return 0.0, 0.0

        Mach = velocity / A.get_SV()

        self.calculate_CY(Mach)
        self.E_pressure(attack_angle, Mach)

        total_moment = 0.0
        total_normal_force = 0.0

        for i, elem in enumerate(self.elem):
            if i == 0:
                nose_cy = self.head_lift(Mach)
                nose_normal_force = nose_cy * attack_angle
                total_normal_force += nose_normal_force
                total_moment += nose_normal_force * elem.focus_position

            else:
                if elem.upper_diameter < elem.lower_diameter:
                    big_rat = elem.ratio
                    S_rat = (
                        elem.upper_area / elem.lower_area if elem.lower_area != 0 else 0
                    )
                    elem_cy = (
                        self.triangle_lift(Mach, big_rat, i)
                        - self.free_triangle_lift(i) * S_rat
                    )
                    elem_normal_force = (
                        elem_cy
                        * attack_angle
                        * elem.upper_area
                        / self.elem[-1].upper_area
                        if self.elem[-1].upper_area != 0
                        else 0
                    )
                else:
                    elem_normal_force = elem.C_ind * attack_angle

                total_normal_force += elem_normal_force
                total_moment += elem_normal_force * elem.focus_position

        if total_normal_force != 0:
            self.focus_position = total_moment / total_normal_force
            self.focus_relative = (
                self.focus_position / self.full_length if self.full_length != 0 else 0
            )
        else:
            self.focus_position = self.full_length / 2
            self.focus_relative = 0.5

        return self.focus_position, self.focus_relative

    def calculate_CXY(self, velocity, altitude, attack_angle):
        A = atmosphere.atmosphere(altitude)
        if A.get_SV() != None:
            Mach = velocity / A.get_SV()
            self.CX = self.calculate_CX(Mach, A.get_SV(), A.get_dyn())
            self.CY = self.calculate_CY(Mach)
            self.E = self.E_pressure(attack_angle, Mach)
            self.CX += self.CY + self.E
            self.CY -= constants.rad(self.CY + self.E)

            # from normal to lift for parts
            # TODO
            self.CYY = self.CY
            self.CY = self.CYY * attack_angle

            self.calculate_aerodynamic_focus(velocity, altitude, attack_angle)
            self.calculate_aeroprofile()

    def calculate_aeroprofile(self):
        li = 0
        cy = 0

        self.liftprofile.clear()
        self.lengthprofile.clear()

        length_edge = [
            self.elem[0].elem_length,
            self.elem[0].elem_length + self.elem[1].elem_length,
            self.elem[0].elem_length
            + self.elem[1].elem_length
            + self.elem[2].elem_length,
        ]

        while li < length_edge[2]:
            if li < length_edge[0]:
                cy = self.elem[0].CY
            elif li < length_edge[1]:
                cy = self.elem[1].CY
            elif li < length_edge[2]:
                cy = self.elem[2].CY

            self.liftprofile.append(cy)
            self.lengthprofile.append(li)
            li += 0.1

    def get_cya_from_coord(self, x):
        return constants.get_coefficient_simple(x, self.lengthprofile, self.liftprofile)

def main():
    parser = rp.rocket_parser("falcon")
    G = UnionStream()
    G.set_elnumber(parser.get_block_number() + 1)
    G.set_diameter(parser.get_part_diameters())
    G.set_length(parser.get_part_length())

    arrayVelocity = np.linspace(10, 3000, 100)
    altitudes = [10]
    red_colors = Reds(np.linspace(0.2, 1.0, len(arrayVelocity)))

    plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)

    focus_data = []
    profile_lines = []
    cx_lines = []
    cy_lines = []
    legend_labels = []

    for alt in altitudes:
        CX_list = []
        CY_list = []
        focus_positions = []
        focus_relative_list = []

        label = f"H={alt}km"

        for v, vel in enumerate(arrayVelocity):
            G.calculate_CXY(vel, alt * 1000, 0)
            CX_list.append(G.CX)
            CY_list.append(G.CYY)
            focus_positions.append(G.focus_position)
            focus_relative_list.append(G.focus_relative)

            focus_data.append(
                {
                    "velocity": vel,
                    "altitude": alt,
                    "focus_position": G.focus_position,
                    "focus_relative": G.focus_relative,
                    "CX": G.CX,
                    "CY": G.CY,
                }
            )

            (line1,) = ax1.plot(
                G.lengthprofile, G.liftprofile, label=label, color=red_colors[v]
            )

        (line2,) = ax2.plot(arrayVelocity, CX_list, label=label)
        (line3,) = ax3.plot(arrayVelocity, CY_list, label=label)

        profile_lines.append(line1)
        cx_lines.append(line2)
        cy_lines.append(line3)
        legend_labels.append(label)

    ax1.set_xlabel("Координаты, м")
    ax1.set_ylabel("CYY")
    ax1.grid(True)
    ax1.set_title("Профиль CYY по длине ракеты")

    ax2.set_xlabel("Скорость, м/с")
    ax2.set_ylabel("CX")
    ax2.grid(True)
    ax2.set_title("Зависимость CX от скорости")

    ax3.set_xlabel("Скорость, м/с")
    ax3.set_ylabel("CYY")
    ax3.grid(True)
    ax3.set_title("Зависимость CYY от скорости")

    plt.suptitle("Аэродинамические коэффициенты")
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    fig = plt.gcf()
    fig.legend(
        profile_lines,
        legend_labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.show()

    if focus_data:
        print(focus_data[-1])


if __name__ == "__main__":
    main()
