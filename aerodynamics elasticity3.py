import math
import copy
import multiprocessing
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import Reds
import warnings
import os

# Импорт пользовательских модулей
import atmosphere
import path
import rocket_parser as rp
import constants

warnings.filterwarnings("ignore")

# ============================================================================
# ЗАГРУЗЧИК РЕАЛЬНЫХ ФОРМ КОЛЕБАНИЙ ИЗ ПЕРВОГО РАСЧЕТА
# ============================================================================


class RealModeShapeLoader:
    """Загрузчик реальных форм колебаний из результатов первого расчета"""

    def __init__(self, rocket_name="amur"):
        self.rocket_name = rocket_name
        self.time_vector = []
        self.freq_vectors = [[], [], []]  # freq_1, freq_2, freq_3
        self.freqmass_vectors = [[], [], []]  # freq_mass_1, freq_mass_2, freq_mass_3
        self.mode_shapes = []  # формы для разных моментов времени
        self.mode_shapes_diff = []  # производные форм
        self.current_time_index = 0
        self.x_coords = []  # координаты по длине
        self.has_frequency_data = False
        self.has_mode_data = False

    def load_frequency_data(self, filename=None):
        freq_file = "output/"+constants.current_rocket+"_frequency.csv"
        self.time_vector = constants.read_array_from_csv(freq_file, "time")
        self.freq_vectors[0]   = constants.read_array_from_csv(freq_file, "freq_1")
        self.freq_vectors[1]   = constants.read_array_from_csv(freq_file, "freq_2")
        self.freq_vectors[2]   = constants.read_array_from_csv(freq_file, "freq_3")
        self.freqmass_vectors[0]   = constants.read_array_from_csv(freq_file, "freq_mass_1")
        self.freqmass_vectors[1]   = constants.read_array_from_csv(freq_file, "freq_mass_2")
        self.freqmass_vectors[2]   = constants.read_array_from_csv(freq_file, "freq_mass_3")
        self.has_frequency_data = True
        return True

    def load_mode_shapes(self, filename=None):

        f_stiffness = [0] * constants.mode_num
        f_stiffness_diff = [0] * constants.mode_num
        oscillations_file = "output/"+constants.current_rocket+"_oscillations.csv"
        f_stiffness[0]      = constants.read_array_from_csv(oscillations_file, "form_1")
        f_stiffness[1]      = constants.read_array_from_csv(oscillations_file, "form_2")
        f_stiffness[2]      = constants.read_array_from_csv(oscillations_file, "form_3")
        f_stiffness_diff[0] = constants.read_array_from_csv(oscillations_file, "difform_1")
        f_stiffness_diff[1] = constants.read_array_from_csv(oscillations_file, "difform_2")
        f_stiffness_diff[2] = constants.read_array_from_csv(oscillations_file, "difform_3")

        self.mode_shapes = [f_stiffness[s] for s in range(constants.mode_num)]
        self.mode_shapes_diff = [f_stiffness_diff[s] for s in range(constants.mode_num)]

        self.has_mode_data = True
        return True


    def set_time(self, time_seconds):
        """Установка текущего времени полета для выбора соответствующей формы"""
        if not self.time_vector:
            self.current_time_index = 0
            return 0

        # Находим ближайший индекс
        self.current_time_index = np.argmin(
            np.abs(np.array(self.time_vector) - time_seconds)
        )
        return self.current_time_index

    def get_mode_at_time(self, mode_index, time_seconds=None):
        """
        Получение формы колебаний для указанной моды в заданный момент времени
        """
        if time_seconds is not None:
            self.set_time(time_seconds)

        # Получаем частоту и обобщенную массу
        frequency = 0
        gen_mass = 0

        if self.has_frequency_data:
            if mode_index < len(self.freq_vectors) and self.current_time_index < len(
                self.freq_vectors[mode_index]
            ):
                frequency = self.freq_vectors[mode_index][self.current_time_index]

            if mode_index < len(
                self.freqmass_vectors
            ) and self.current_time_index < len(self.freqmass_vectors[mode_index]):
                gen_mass = self.freqmass_vectors[mode_index][self.current_time_index]

        # Создаем объект ModeShape
        mode = ModeShape(
            name=f"{mode_index+1}-я изгибная мода",
            frequency=frequency,
            generalized_mass=gen_mass,
            damping=0.02,
            nodes=self.x_coords if self.x_coords else [],
            values=[],
            derivatives=[],
        )

        # Загружаем значения формы и производной
        time_idx = min(self.current_time_index, len(self.mode_shapes) - 1)

        if self.has_mode_data and self.mode_shapes and time_idx < len(self.mode_shapes):
            shapes_dict = self.mode_shapes[time_idx]
            if mode_index in shapes_dict:
                mode.values = shapes_dict[mode_index]

        if (
            self.has_mode_data
            and self.mode_shapes_diff
            and time_idx < len(self.mode_shapes_diff)
        ):
            diff_dict = self.mode_shapes_diff[time_idx]
            if mode_index in diff_dict:
                mode.derivatives = diff_dict[mode_index]

        return mode

    def get_all_modes_at_time(self, time_seconds=None):
        """Получение всех трех мод для заданного момента времени"""
        modes = []
        for i in range(3):
            mode = self.get_mode_at_time(i, time_seconds)
            modes.append(mode)
        return modes

    def has_data(self):
        """Проверка наличия загруженных данных"""
        return self.has_frequency_data and self.has_mode_data


# ============================================================================
# КЛАССЫ ДАННЫХ
# ============================================================================


@dataclass
class Element:
    """Элемент ракеты"""

    upper_diameter: float = 0.0
    lower_diameter: float = 0.0
    elem_length: float = 0.0
    virtual_length: float = 0.0
    ratio: float = 0.0
    round_area: float = 0.0
    upper_area: float = 0.0
    lower_area: float = 0.0
    base_line: float = 0.0
    focus_position: float = 0.0
    x_start: float = 0.0
    x_end: float = 0.0
    C_fric: float = 0.0
    C_pres: float = 0.0
    C_ind: float = 0.0
    CX: float = 0.0
    CY: float = 0.0
    CYY: float = 0.0


@dataclass
class Geometry:
    """Геометрия ракеты"""

    PI = math.pi

    def __init__(self):
        self.elem: List[Element] = []
        self.full_length: float = 0.0
        self.full_round_area: float = 0.0
        self.full_ratio: float = 0.0
        self.midel_diameter: float = 0.0
        self.midel_area: float = 0.0
        self.cif: float = 0.0
        self.num: float = 0.0
        self.x_coords: List[float] = []

    def set_elnumber(self, n: int):
        self.elem = [Element() for _ in range(n)]

    def set_length(self, lengths: List[float]):
        for i, l in enumerate(lengths):
            if i < len(self.elem):
                self.elem[i].elem_length = l
        self.pre_calculations()

    def set_diameter(self, diameters: List[float]):
        for i, d in enumerate(diameters):
            if i >= len(self.elem):
                break
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
            e.x_start = current_position
            e.x_end = current_position + e.elem_length

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
                d_diff = e.lower_diameter - e.upper_diameter
                e.base_line = math.sqrt(e.elem_length**2 + (d_diff**2) / 4)
                e.round_area = (
                    self.PI * (e.upper_diameter + e.lower_diameter) * e.base_line / 2
                )
                e.focus_position = current_position + e.elem_length / 2

            e.upper_area = self.PI * (e.upper_diameter / 2) ** 2
            e.lower_area = self.PI * (e.lower_diameter / 2) ** 2

            # Виртуальная длина и удлинение
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
            except:
                e.virtual_length = e.elem_length

            if e.upper_diameter < e.lower_diameter and abs(e.upper_diameter) > 0.1:
                e.ratio = e.virtual_length / e.lower_diameter
            else:
                e.ratio = (
                    e.elem_length / e.lower_diameter if e.lower_diameter != 0 else 0
                )

            self.full_length += e.elem_length
            self.full_round_area += e.round_area
            current_position += e.elem_length

        if self.elem:
            if self.elem[-1].upper_diameter != 0:
                self.full_ratio = self.full_length / self.elem[-1].upper_diameter
            self.midel_diameter = self.elem[-1].upper_diameter
            self.midel_area = self.elem[-1].lower_area

        self.x_coords = np.arange(0, self.full_length, 0.1).tolist()
        if self.x_coords[-1] < self.full_length:
            self.x_coords.append(self.full_length)


# ============================================================================
# КЛАСС ФОРМ КОЛЕБАНИЙ
# ============================================================================


@dataclass
class ModeShape:
    """Форма колебаний с готовыми значениями из первого расчета"""

    name: str
    frequency: float
    generalized_mass: float  # Обобщенная масса из freq_mass
    damping: float = 0.02
    nodes: List[float] = None
    values: List[float] = None
    derivatives: List[float] = None

    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []
        if self.values is None:
            self.values = []
        if self.derivatives is None:
            self.derivatives = []
        self.omega = 2 * math.pi * self.frequency

    def interpolate(self, x: float) -> float:
        """Интерполяция формы в заданной точке"""
        if not self.nodes or not self.values:
            return 0.0
        return np.interp(x, self.nodes, self.values)

    def interpolate_derivative(self, x: float) -> float:
        """Интерполяция производной формы в заданной точке"""
        if not self.nodes or not self.derivatives:
            # Если производной нет, вычисляем численно
            return (self.interpolate(x + 0.01) - self.interpolate(x - 0.01)) / 0.02
        return np.interp(x, self.nodes, self.derivatives)


# ============================================================================
# КЛАССЫ АЭРОДИНАМИКИ
# ============================================================================


class Friction(Geometry):
    """Расчет сил трения"""

    def __init__(self):
        super().__init__()
        self.h_s = 8e-6
        self.area_ratio = 0.0
        self.Re = 0.0
        self.n = 0.0
        self.x_t = 0.0
        self.C_fric = 0.0

    def stream_calc(self, Re: float, Mach: float):
        if Re <= 485000:
            self.cif = 2.656 / math.sqrt(Re) if abs(Re) >= 0.0001 else 0
            self.num = pow(1 + 0.1 * pow(Mach, 0.1), -0.125)
        elif Re < 10000000:
            try:
                log_arg = (self.h_s / self.full_length * Re) - 1
                denominator = 2.2 + 0.08 * pow(Mach, 2) / (1 + 0.312 * pow(Mach, 2))
                self.n = 5 + (1.3 + 0.6 * Mach * (1 - 0.25 * pow(Mach, 2))) * math.sqrt(
                    1 - pow(math.log10(log_arg) / denominator, 2)
                )
            except:
                self.n = 5
            self.x_t = min(pow(10, self.n) / Re, 1.0)
            if self.x_t >= 1:
                self.cif = (
                    0.91
                    / pow(math.log10(Re), 2.58)
                    * pow(
                        1 - self.x_t + 40 * pow(self.x_t, 0.625) / pow(Re, 0.375), 0.8
                    )
                )
            else:
                self.cif = 2.656 / math.sqrt(Re) if abs(Re) >= 0.0001 else 0
            self.num = pow(1 + 0.1 * pow(Mach, 0.1), -2 / 3)
        else:
            self.cif = 0.91 / pow(math.log10(Re), 2.58)
            self.num = pow(1 + 0.1 * pow(Mach, 0.1), -2 / 3)

    def fricalc(self, Mach: float, SS: float, nu: float) -> float:
        if nu is None:
            nu = 1.789e-05
        self.area_ratio = (
            self.full_round_area / self.midel_area if self.midel_area != 0 else 0
        )
        self.Re = SS * Mach * self.full_length / nu if nu != 0 else 0
        self.stream_calc(self.Re, Mach)
        for e in self.elem:
            e.C_fric = e.round_area * self.cif * self.num / 2
        self.C_fric = self.area_ratio * self.cif * self.num / 2
        return self.C_fric


class Pressure(Geometry):
    """Расчет давления"""

    def read_pressure_file(
        self, filename: str, rows: int, cols: int
    ) -> List[List[float]]:
        try:
            df = pd.read_csv(filename, delimiter=None, engine="python", header=None)
            if rows > 0:
                df = df.head(rows)
            if df.shape[1] < cols:
                raise ValueError(f"File {filename} has fewer than {cols} columns")
            df = df.iloc[:, :cols]
            data = df.values.tolist()
            return list(map(list, zip(*data)))
        except:
            return []

    def interpolate_Mach(self, Mach: float, data: List[List[float]]) -> float:
        if not data or len(data) < 2:
            return 0.0
        Mach_v, values = data[0], data[1]
        for i in range(1, len(Mach_v)):
            if Mach >= Mach_v[i - 1] and Mach < Mach_v[i]:
                return values[i - 1] + (Mach - Mach_v[i - 1]) * (
                    values[i] - values[i - 1]
                ) / (Mach_v[i] - Mach_v[i - 1])
        return values[-1] if values else 0.0

    def select_ratio_data_pressure(
        self, ratio: float, data: List[List[float]]
    ) -> List[List[float]]:
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

    def select_ratio_data_triangle(
        self, ratio: float, data: List[List[float]]
    ) -> List[List[float]]:
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

    def bottom_pres(self, Mach: float) -> float:
        if Mach < 1:
            if self.cif * self.num * self.full_ratio == 0:
                return 0
            return 0.0155 / math.sqrt(self.cif * self.num * self.full_ratio)
        else:
            data = self.read_pressure_file(path.root_path + "HeadPressure.csv", 10, 10)
            return self.interpolate_Mach(Mach, [data[0], data[6]]) if data else 0

    def head_Cpres(self, Mach: float) -> float:
        data = self.read_pressure_file(path.root_path + "HeadPressure.csv", 10, 10)
        if not data or not self.elem:
            return 0
        ratio = self.elem[0].ratio
        H_current = self.select_ratio_data_pressure(ratio, data)
        return self.interpolate_Mach(Mach, H_current)

    def triangle_Cpres(self, Mach: float, ratio: float) -> float:
        data = self.read_pressure_file(path.root_path + "TrianglePressure.csv", 10, 7)
        if not data:
            return 0
        H_current = self.select_ratio_data_triangle(ratio, data)
        return self.interpolate_Mach(Mach, H_current)

    def prescalc(self, Mach: float) -> float:
        res = 0.0
        for i in range(1, len(self.elem)):
            if self.elem[i].lower_area != 0:
                self.elem[i].C_pres = self.triangle_Cpres(Mach, self.elem[i].ratio) * (
                    1 - self.elem[i].upper_area / self.elem[i].lower_area
                )
                res += self.elem[i].C_pres
        return res + self.head_Cpres(Mach) + self.bottom_pres(Mach)


class Inductance(Geometry):
    """Расчет индуктивного сопротивления"""

    def read_pressure_file(
        self, filename: str, rows: int, cols: int
    ) -> List[List[float]]:
        data = []
        try:
            with open(filename, "r") as f:
                for _ in range(rows):
                    line = f.readline()
                    if not line:
                        break
                    parts = line.strip().split()
                    if len(parts) < cols:
                        continue
                    data.append([float(x) for x in parts[:cols]])
            return list(map(list, zip(*data))) if data else []
        except:
            return []

    def E_pressure(self, angle: float, Mach: float) -> float:
        N = 10
        data = self.read_pressure_file(path.root_path + "EPressure.csv", N, 3)
        if not data or len(data) < 3 or not self.elem:
            return 0.0

        Mach_v, H_head, H_cone = data[0], data[1], data[2]

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

        E_head = E_cone = 0
        for i in range(1, len(Mach_v)):
            if Mach_val >= Mach_v[i - 1] and Mach_val < Mach_v[i]:
                E_head = H_head[i - 1] + (Mach_val - Mach_v[i - 1]) * (
                    H_head[i] - H_head[i - 1]
                ) / (Mach_v[i] - Mach_v[i - 1])
                E_cone = H_cone[i - 1] + (Mach_val - Mach_v[i - 1]) * (
                    H_cone[i] - H_cone[i - 1]
                ) / (Mach_v[i] - Mach_v[i - 1])

        self.elem[0].C_ind = (
            self.elem[0].CY + constants.rad(2 * E_head)
        ) * constants.sqr(angle)

        for j in range(1, len(self.elem)):
            if (
                self.elem[j].upper_diameter < self.elem[j].lower_diameter
                and self.elem[-1].upper_area != 0
            ):
                ratio = self.elem[-1].upper_area
                self.elem[j].C_ind = (
                    self.elem[j].CY * self.elem[j].upper_area / ratio
                    + constants.rad(2 * E_cone * self.elem[j].upper_area / ratio)
                ) * constants.sqr(angle)

        return sum(e.C_ind for e in self.elem)


class LiftForce(Inductance):
    """Расчет подъемной силы"""

    def sqr(self, x):
        return x * x

    def rad(self, x):
        return x / 57.3

    def head_lift(self, Mach: float) -> float:
        N = 9
        data = self.read_pressure_file(path.root_path + "HeadNormal.csv", N, 6)
        if not data or len(data) < 6 or not self.elem:
            return 0.035

        Mah_v, H_0, H_05, H_1, H_2, H_4 = (
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
        )

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

    def free_triangle_lift(self, index: int) -> float:
        if index >= len(self.elem):
            return 0.0
        denominator = self.elem[index].virtual_length - self.elem[index].elem_length
        if abs(denominator) > 1e-6:
            arg = self.elem[index].lower_diameter / 2 / denominator
            Q = math.atan(arg)
            return (2 / 57.3) * constants.sqr(math.cos(Q))
        return 0.0

    def triangle_lift(self, Mach: float, ratio: float, index: int) -> float:
        N = 9
        data = self.read_pressure_file(path.root_path + "TriangleNormal.csv", N, 6)
        if not data or len(data) < 6 or index >= len(self.elem):
            return 0.0

        Mah_v, H_0, H_1, H_2, H_3, H_4 = (
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
        )

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

    def calculate_CY(self, Mach: float) -> float:
        return self.head_lift(Mach) + self.un_triangle_lift(Mach)

    def un_triangle_lift(self, Mach: float) -> float:
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
            if self.elem[-1].upper_area != 0:
                res += (
                    self.elem[i].CY * self.elem[i].upper_area / self.elem[-1].upper_area
                )
        return res


class DragForce(Friction, Pressure):
    """Расчет силы сопротивления"""

    def calculate_CX(self, Mach: float, SS: float, nu: float) -> float:
        return self.fricalc(Mach, SS, nu) + self.prescalc(Mach)


# ============================================================================
# РАСШИРЕННЫЙ КЛАСС АЭРОУПРУГОСТИ С РЕАЛЬНЫМИ ФОРМАМИ
# ============================================================================


@dataclass
class AeroElasticParams:
    mass_per_length: float = 1000.0
    bending_stiffness: float = 5e6
    torsional_stiffness: float = 8e6


class AdvancedAeroElasticity:
    """
    Расчет аэроупругости с использованием готовых форм колебаний и обобщенных масс
    """

    def __init__(self, geometry, mode_loader=None):
        self.geom = geometry
        self.mode_loader = mode_loader
        self.modes = []
        self.params = AeroElasticParams()
        self.current_time = 0.0

        if mode_loader and mode_loader.has_data():
            self.set_flight_time(0.0)
            print(f"   Используются реальные формы колебаний")
        else:
            print(f"   Внимание: используются приближенные формы колебаний")

    def set_flight_time(self, time_seconds):
        """Установка времени полета для выбора соответствующих форм"""
        self.current_time = time_seconds
        if self.mode_loader and self.mode_loader.has_data():
            self.modes = self.mode_loader.get_all_modes_at_time(time_seconds)

    def _get_local_diameter(self, x: float) -> float:
        for elem in self.geom.elem:
            if elem.x_start <= x <= elem.x_end:
                t = (x - elem.x_start) / elem.elem_length if elem.elem_length > 0 else 0
                return elem.upper_diameter * (1 - t) + elem.lower_diameter * t
        return self.geom.midel_diameter

    def _get_local_area(self, x: float) -> float:
        d = self._get_local_diameter(x)
        return math.pi * (d / 2) ** 2

    def calculate_mode_amplitudes(
        self,
        velocity: float,
        altitude: float,
        CYY_dist: List[float],
        angle_of_attack: float,
    ) -> List[float]:
        """
        Расчет амплитуд для каждой моды
        q_i = F_i / (M_i * ω_i²)
        """
        atmos = atmosphere.atmosphere(altitude)
        if atmos.get_SV() is None:
            return [0.0] * len(self.modes)

        rho = atmos.get_density()
        q = 0.5 * rho * velocity**2  # скоростной напор

        amplitudes = []
        for mode in self.modes:
            if not mode.nodes or not mode.values:
                amplitudes.append(0.0)
                continue

            # Обобщенная сила
            F_mode = 0.0
            for i, x in enumerate(self.geom.x_coords[:-1]):
                x1 = x
                x2 = self.geom.x_coords[i + 1]
                dx = x2 - x1
                x_center = (x1 + x2) / 2

                # Площадь поперечного сечения в точке
                S_local = self._get_local_area(x_center)

                # Распределенная аэродинамическая нагрузка (Н)
                cyy_local = CYY_dist[i] if i < len(CYY_dist) else 0
                p_local = q * cyy_local * math.radians(angle_of_attack) * S_local

                # Форма колебаний
                phi = mode.interpolate(x_center)

                # Интегрирование по длине
                F_mode += p_local * phi

            # Используем готовую обобщенную массу
            M_mode = mode.generalized_mass

            if M_mode > 0 and mode.omega > 0:
                # Амплитуда
                amplitude = F_mode / (M_mode * mode.omega**2)
                amplitudes.append(abs(amplitude))
            else:
                amplitudes.append(0.0)

        return amplitudes

    def calculate_local_gain(self, x: float, amplitudes: List[float]) -> float:
        """Локальный коэффициент усиления от деформации"""
        if not self.modes or not amplitudes:
            return 1.0

        deformation = 0.0
        dalpha = 0.0

        for mode, amp in zip(self.modes, amplitudes):
            deformation += amp * mode.interpolate(x)
            dalpha += amp * mode.interpolate_derivative(x)

        # Коэффициент усиления зависит от деформации и угла
        gain = 1.0 + abs(deformation) * 10.0 + abs(dalpha) * 20.0

        # Ограничиваем максимальное усиление
        return min(gain, 3.0)

    def calculate_focus_shift(
        self,
        velocity: float,
        altitude: float,
        CYY_dist: List[float],
        original_focus: float,
        angle_of_attack: float,
    ) -> float:
        """Смещение фокуса под влиянием аэроупругости"""
        if not self.geom.x_coords or not CYY_dist:
            return original_focus

        atmos = atmosphere.atmosphere(altitude)
        if atmos.get_SV() is None:
            return original_focus

        rho = atmos.get_density()
        q = 0.5 * rho * velocity**2

        amplitudes = self.calculate_mode_amplitudes(
            velocity, altitude, CYY_dist, angle_of_attack
        )

        moment_sum = 0.0
        force_sum = 0.0

        for i, x in enumerate(self.geom.x_coords[:-1]):
            x1 = x
            x2 = self.geom.x_coords[i + 1]
            dx = x2 - x1
            x_center = (x1 + x2) / 2

            S_local = self._get_local_area(x_center)
            cyy_local = CYY_dist[i] if i < len(CYY_dist) else 0
            p_base = q * S_local * cyy_local * math.radians(angle_of_attack)

            gain = self.calculate_local_gain(x_center, amplitudes)
            p_total = p_base * gain

            force_sum += p_total * dx
            moment_sum += p_total * x_center * dx

        return moment_sum / force_sum if force_sum > 0 else original_focus

    def calculate_distributed_corrections(
        self,
        velocity: float,
        altitude: float,
        Mach: float,
        CX: float,
        CYY: float,
        CYY_dist: List[float],
        angle_of_attack: float,
    ) -> Dict:
        """Расчет распределенных аэроупругих поправок"""
        result = {
            "CX_corr": 0.0,
            "CY_corr": 0.0,
            "gain_global": 1.0,
            "gain_dist": [],
            "additional_angle_dist": [],
            "amplitudes": [],
            "mode_names": [],
            "mode_frequencies": [],
            "mode_generalized_masses": [],
            "focus_shift": 0.0,
            "focus_new": 0.0,
        }

        if not self.geom.x_coords or not CYY_dist or not self.modes:
            return result

        amplitudes = self.calculate_mode_amplitudes(
            velocity, altitude, CYY_dist, angle_of_attack
        )
        result["amplitudes"] = amplitudes
        result["mode_names"] = [mode.name for mode in self.modes]
        result["mode_frequencies"] = [mode.frequency for mode in self.modes]
        result["mode_generalized_masses"] = [
            mode.generalized_mass for mode in self.modes
        ]

        local_gains = []
        additional_angles = []
        for x in self.geom.x_coords:
            gain = self.calculate_local_gain(x, amplitudes)
            local_gains.append(gain)

            dalpha = 0.0
            for mode, amp in zip(self.modes, amplitudes):
                dalpha += amp * mode.interpolate_derivative(x)
            additional_angles.append(math.degrees(dalpha))

        result["gain_dist"] = local_gains
        result["additional_angle_dist"] = additional_angles
        result["gain_global"] = np.mean(local_gains)

        original_focus = self._calculate_focus_position(CYY_dist)
        new_focus = self.calculate_focus_shift(
            velocity, altitude, CYY_dist, original_focus, angle_of_attack
        )
        result["focus_shift"] = new_focus - original_focus
        result["focus_new"] = new_focus

        L = self.geom.full_length
        for i, x in enumerate(self.geom.x_coords[:-1]):
            dx = self.geom.x_coords[i + 1] - x
            gain_local = local_gains[i]
            cyy_local = CYY_dist[i] if i < len(CYY_dist) else CYY

            result["CX_corr"] += CX * (gain_local - 1) * 0.05 * dx / L
            result["CY_corr"] += cyy_local * (gain_local - 1) * 0.1 * dx / L

        return result

    def _calculate_focus_position(self, CYY_dist: List[float]) -> float:
        """Положение фокуса без учета аэроупругости"""
        if not self.geom.x_coords or not CYY_dist:
            return self.geom.full_length / 2

        moment_sum = 0.0
        force_sum = 0.0

        for i, x in enumerate(self.geom.x_coords[:-1]):
            dx = self.geom.x_coords[i + 1] - x
            x_center = (x + self.geom.x_coords[i + 1]) / 2
            cyy_local = CYY_dist[i] if i < len(CYY_dist) else 0

            force_sum += cyy_local * dx
            moment_sum += cyy_local * x_center * dx

        return moment_sum / force_sum if force_sum > 0 else self.geom.full_length / 2

    def print_mode_info(self):
        """Вывод информации о модах"""
        print("\n" + "=" * 50)
        print("ИНФОРМАЦИЯ О ФОРМАХ КОЛЕБАНИЙ")
        print("=" * 50)
        for i, mode in enumerate(self.modes):
            print(f"Мода {i+1}: {mode.name}")
            print(f"  Частота: {mode.frequency:.3f} Гц")
            print(f"  Обобщенная масса: {mode.generalized_mass:.1f} кг")
            print(f"  Циклическая частота: {mode.omega:.3f} рад/с")
            if mode.nodes and len(mode.nodes) > 10 and mode.values:
                print(
                    f"  Амплитуда: min={min(mode.values):.3f}, max={max(mode.values):.3f}"
                )
                nodes_pos = []
                for j in range(len(mode.values) - 1):
                    if mode.values[j] * mode.values[j + 1] <= 0:
                        nodes_pos.append(f"{mode.nodes[j]:.2f}")
                if nodes_pos:
                    print(f"  Узлы: {', '.join(nodes_pos[:5])} м")
        print("=" * 50)


# ============================================================================
# ОСНОВНОЙ КЛАСС АЭРОДИНАМИЧЕСКОГО РАСЧЕТА
# ============================================================================


class UnionStream(DragForce, LiftForce):
    """Основной класс для аэродинамических расчетов"""

    def __init__(self):
        super().__init__()
        self.E = 0.0
        self.CX = 0.0
        self.CY = 0.0
        self.focus_position = 0.0
        self.focus_relative = 0.0
        self.liftprofile = []
        self.lengthprofile = []
        self.CYY = 0.0
        self.CYY_distribution = []

    def calculate_CXY(self, velocity: float, altitude: float, attack_angle: float):
        A = atmosphere.atmosphere(altitude)
        if A.get_SV() is not None:
            Mach = velocity / A.get_SV()
            self.CX = self.calculate_CX(Mach, A.get_SV(), A.get_dyn())
            self.CYY = self.calculate_CY(Mach)
            self.E = self.E_pressure(attack_angle, Mach)
            self.CX += self.CYY + self.E
            self.CY = self.CYY * (attack_angle / 57.3)
            self._generate_CYY_distribution()

    def _generate_CYY_distribution(self):
        self.CYY_distribution = []
        for x in self.x_coords:
            for elem in self.elem:
                if elem.x_start <= x <= elem.x_end:
                    self.CYY_distribution.append(elem.CY)
                    break
            else:
                self.CYY_distribution.append(self.CYY)


# ============================================================================
# КЛАСС РЕЗУЛЬТАТА
# ============================================================================


@dataclass
class AeroResult:
    velocity: float
    altitude: float
    Mach: float
    CX: float
    CYY: float
    CY: float
    CX_elastic: Optional[float] = None
    CYY_elastic: Optional[float] = None
    CY_elastic: Optional[float] = None
    gain: Optional[float] = None
    additional_angle_mean: Optional[float] = None
    additional_angle_max: Optional[float] = None
    mode_amplitudes: Optional[List[float]] = None
    mode_names: Optional[List[str]] = None
    mode_frequencies: Optional[List[float]] = None
    mode_generalized_masses: Optional[List[float]] = None
    focus_rigid: Optional[float] = None
    focus_elastic: Optional[float] = None
    focus_shift: Optional[float] = None


# ============================================================================
# КЛАСС ПАРАЛЛЕЛЬНОГО РАСЧЕТА
# ============================================================================


class ParallelAerodynamics:
    """Параллельный аэродинамический расчет с реальными формами"""

    def __init__(self, base_calculator, mode_loader=None):
        self.base_calc = base_calculator
        self.num_workers = mp.cpu_count()
        self.mode_loader = mode_loader
        self.aero_elastic = None

    def set_flight_time(self, time_seconds):
        """Установка времени полета для выбора форм"""
        if self.aero_elastic:
            self.aero_elastic.set_flight_time(time_seconds)

    def calculate_range(
        self,
        velocities: np.ndarray,
        altitudes: List[float],
        attack_angle: float = 2.0,
        use_elastic: bool = False,
        flight_time: float = 0.0,
    ) -> List[AeroResult]:
        """Параллельный расчет для диапазона скоростей и высот"""

        if use_elastic and self.mode_loader:
            print(
                f"   Используются реальные формы колебаний для t = {flight_time:.1f} с"
            )

        args_list = []
        for alt in altitudes:
            for vel in velocities:
                args_list.append(
                    (vel, alt * 1000, attack_angle, use_elastic, flight_time)
                )

        print(f"   Запуск на {self.num_workers} процессах, точек: {len(args_list)}")

        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._calculate_point, args) for args in args_list
            ]

            for i, future in enumerate(futures):
                try:
                    results.append(future.result())
                    if (i + 1) % 50 == 0:
                        print(f"   Обработано {i + 1}/{len(args_list)} точек")
                except Exception as e:
                    print(f"   Ошибка в расчете точки {i}: {e}")

        return results

    def _calculate_point(self, args: Tuple) -> AeroResult:
        velocity, altitude, attack_angle, use_elastic, flight_time = args

        calc = copy.deepcopy(self.base_calc)
        calc.calculate_CXY(velocity, altitude, attack_angle)

        A = atmosphere.atmosphere(altitude)
        Mach = velocity / A.get_SV() if A.get_SV() else 0

        # Положение фокуса без аэроупругости
        focus_rigid = 0.0
        moment_sum = 0.0
        force_sum = 0.0
        for i, x in enumerate(calc.x_coords[:-1]):
            dx = calc.x_coords[i + 1] - x
            x_center = (x + calc.x_coords[i + 1]) / 2
            cyy_local = (
                calc.CYY_distribution[i] if i < len(calc.CYY_distribution) else calc.CYY
            )
            force_sum += cyy_local * dx
            moment_sum += cyy_local * x_center * dx
        focus_rigid = (
            moment_sum / force_sum / calc.full_length if force_sum > 0 else 0.5
        )

        result = AeroResult(
            velocity=velocity,
            altitude=altitude,
            Mach=Mach,
            CX=calc.CX,
            CYY=calc.CYY,
            CY=calc.CY,
            focus_rigid=focus_rigid,
        )
        if not self.mode_loader.has_data():
            raise SystemExit

        if use_elastic:
            aero_elastic = AdvancedAeroElasticity(calc, self.mode_loader)
            if self.mode_loader:
                aero_elastic.set_flight_time(flight_time)

            corrections = aero_elastic.calculate_distributed_corrections(
                velocity,
                altitude,
                Mach,
                calc.CX,
                calc.CYY,
                calc.CYY_distribution,
                attack_angle,
            )

            result.CX_elastic = calc.CX + corrections["CX_corr"]
            result.CYY_elastic = calc.CYY + corrections["CY_corr"]
            result.CY_elastic = calc.CY + corrections["CY_corr"]
            result.gain = corrections["gain_global"]
            result.focus_elastic = corrections["focus_new"] / calc.full_length
            result.focus_shift = corrections["focus_shift"] / calc.full_length

            if corrections["additional_angle_dist"]:
                result.additional_angle_mean = np.mean(
                    np.abs(corrections["additional_angle_dist"])
                )
                result.additional_angle_max = np.max(
                    np.abs(corrections["additional_angle_dist"])
                )

            result.mode_amplitudes = corrections["amplitudes"]
            result.mode_names = corrections["mode_names"]
            result.mode_frequencies = corrections["mode_frequencies"]
            result.mode_generalized_masses = corrections["mode_generalized_masses"]

        return result


# ============================================================================
# КЛАСС ВИЗУАЛИЗАЦИИ
# ============================================================================


class SimpleVisualizer:
    def __init__(self):
        self.fig_size = (25, 15)
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def plot_results(
        self,
        results_rigid: List[AeroResult],
        results_elastic: List[AeroResult],
        altitudes: List[float],
    ):

        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)
        axes = axes.flatten()
        axes[5].set_visible(False)

        for idx, alt in enumerate(altitudes):
            rigid_alt = [r for r in results_rigid if abs(r.altitude / 1000 - alt) < 0.1]
            elastic_alt = [
                r for r in results_elastic if abs(r.altitude / 1000 - alt) < 0.1
            ]

            if not rigid_alt or not elastic_alt:
                continue

            rigid_alt.sort(key=lambda x: x.velocity)
            elastic_alt.sort(key=lambda x: x.velocity)

            velocities = [r.velocity for r in rigid_alt]
            color = self.colors[idx % len(self.colors)]
            label = f"H = {alt} км"

            # График 1: CYY
            axes[0].plot(
                velocities,
                [r.CYY for r in rigid_alt],
                "--",
                color=color,
                linewidth=2,
                alpha=0.7,
            )
            axes[0].plot(
                velocities,
                [r.CYY_elastic for r in elastic_alt],
                "-",
                color=color,
                linewidth=2,
                label=label,
            )

            # График 2: CX
            axes[1].plot(
                velocities,
                [r.CX for r in rigid_alt],
                "--",
                color=color,
                linewidth=2,
                alpha=0.7,
            )
            axes[1].plot(
                velocities,
                [r.CX_elastic for r in elastic_alt],
                "-",
                color=color,
                linewidth=2,
            )

            # График 3: Влияние аэроупругости
            influence = [
                (r_el.CYY_elastic - r_rig.CYY) / r_rig.CYY * 100
                for r_el, r_rig in zip(elastic_alt, rigid_alt)
            ]
            axes[2].plot(
                velocities, influence, "-", color=color, linewidth=2, label=label
            )

            # График 4: Дополнительный угол атаки
            angles = [
                r.additional_angle_mean for r in elastic_alt if r.additional_angle_mean
            ]
            if angles:
                axes[3].plot(
                    velocities[: len(angles)],
                    angles,
                    "-",
                    color=color,
                    linewidth=2,
                    label=label,
                )

            # График 5: Положение фокуса
            axes[4].plot(
                velocities,
                [r.focus_rigid for r in rigid_alt],
                "--",
                color=color,
                linewidth=2,
                alpha=0.7,
            )
            focus_elastic = [r.focus_elastic for r in elastic_alt if r.focus_elastic]
            if focus_elastic:
                axes[4].plot(
                    velocities[: len(focus_elastic)],
                    focus_elastic,
                    "-",
                    color=color,
                    linewidth=2,
                    label=label,
                )

        # Настройка графиков
        titles = [
            "Коэффициент подъемной силы CYY",
            "Коэффициент лобового сопротивления CX",
            "Влияние аэроупругости на CYY, %",
            "Дополнительный угол атаки, град",
            "Положение фокуса X_focus/L",
        ]

        ylabels = ["CYY", "CX", "Изменение, %", "Угол, град", "X_focus / L"]

        for i, ax in enumerate(axes[:5]):
            ax.set_xlabel("Скорость, м/с", fontsize=12)
            ax.set_ylabel(ylabels[i], fontsize=12)
            ax.set_title(titles[i], fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            ax.set_xlim([0, 2100])
            if i == 4:
                ax.axhline(
                    y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
                )
                ax.set_ylim([0.3, 0.7])

        plt.tight_layout()
        plt.show()

    def print_summary(
        self, results_rigid: List[AeroResult], results_elastic: List[AeroResult]
    ):
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ВЛИЯНИЯ АЭРОУПРУГОСТИ")
        print("=" * 60)

        altitudes = sorted(set(r.altitude / 1000 for r in results_rigid))

        for alt in altitudes:
            rigid_alt = [r for r in results_rigid if abs(r.altitude / 1000 - alt) < 0.1]
            elastic_alt = [
                r for r in results_elastic if abs(r.altitude / 1000 - alt) < 0.1
            ]

            if not rigid_alt or not elastic_alt:
                continue

            rigid_alt.sort(key=lambda x: x.velocity)
            elastic_alt.sort(key=lambda x: x.velocity)

            cyy_rigid = np.mean([r.CYY for r in rigid_alt])
            cyy_elastic = np.mean([r.CYY_elastic for r in elastic_alt])
            cyy_diff = (cyy_elastic - cyy_rigid) / cyy_rigid * 100

            gains = [r.gain for r in elastic_alt if r.gain]
            max_gain = max(gains) if gains else 1.0

            angles = [
                r.additional_angle_mean for r in elastic_alt if r.additional_angle_mean
            ]
            mean_angle = np.mean(angles) if angles else 0.0

            focus_shifts = [r.focus_shift for r in elastic_alt if r.focus_shift]
            mean_shift = np.mean(focus_shifts) if focus_shifts else 0.0

            print(f"\nВысота: {alt} км")
            print(
                f"  CYY: без упр. = {cyy_rigid:.4f}, с упр. = {cyy_elastic:.4f} ({cyy_diff:+.1f}%)"
            )
            print(f"  Макс. усиление: {max_gain:.3f}")
            print(f"  Доп. угол атаки (средн.): {mean_angle:.3f}°")
            print(f"  Смещение фокуса: {mean_shift:.4f} L ({mean_shift*100:+.1f}%)")

            # Информация о модах для первой точки
            if elastic_alt and elastic_alt[0].mode_amplitudes:
                print(f"  Амплитуды мод (V={elastic_alt[0].velocity:.0f} м/с):")
                for name, amp, freq, mass in zip(
                    elastic_alt[0].mode_names,
                    elastic_alt[0].mode_amplitudes,
                    elastic_alt[0].mode_frequencies,
                    elastic_alt[0].mode_generalized_masses,
                ):
                    print(f"    {name}: f={freq:.2f} Гц, M={mass:.0f} кг, A={amp:.6f}")


# ============================================================================
# ФУНКЦИЯ СОХРАНЕНИЯ РЕЗУЛЬТАТОВ
# ============================================================================


def save_results(results_rigid: List[AeroResult], results_elastic: List[AeroResult]):
    data = []
    for r_rigid, r_elastic in zip(results_rigid, results_elastic):
        row = {
            "velocity": r_rigid.velocity,
            "altitude_km": r_rigid.altitude / 1000,
            "Mach": r_rigid.Mach,
            "CX_rigid": r_rigid.CX,
            "CYY_rigid": r_rigid.CYY,
            "CY_rigid": r_rigid.CY,
            "focus_rigid": r_rigid.focus_rigid,
            "CX_elastic": r_elastic.CX_elastic,
            "CYY_elastic": r_elastic.CYY_elastic,
            "CY_elastic": r_elastic.CY_elastic,
            "focus_elastic": r_elastic.focus_elastic,
            "focus_shift": r_elastic.focus_shift,
            "gain": r_elastic.gain,
            "additional_angle_mean": r_elastic.additional_angle_mean,
            "additional_angle_max": r_elastic.additional_angle_max,
        }

        if r_elastic.mode_amplitudes:
            for i, (name, amp, freq, mass) in enumerate(
                zip(
                    r_elastic.mode_names,
                    r_elastic.mode_amplitudes,
                    r_elastic.mode_frequencies,
                    r_elastic.mode_generalized_masses,
                )
            ):
                row[f"mode{i+1}_name"] = name
                row[f"mode{i+1}_freq"] = freq
                row[f"mode{i+1}_mass"] = mass
                row[f"mode{i+1}_amplitude"] = amp

        data.append(row)

    df = pd.DataFrame(data)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/aerodynamics_with_real_modes.csv", index=False)
    print(f"\nРезультаты сохранены в output/aerodynamics_with_real_modes.csv")


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================


def main():
    print("=" * 60)
    print("АЭРОДИНАМИЧЕСКИЙ РАСЧЕТ С УЧЕТОМ АЭРОУПРУГОСТИ")
    print("=" * 60)

    # Загрузка геометрии ракеты
    print("\n1. Загрузка геометрии ракеты...")
    try:
        parser = rp.rocket_parser()  # Указываем имя ракеты
        calculator = UnionStream()
        calculator.set_elnumber(parser.get_block_number() + 1)
        calculator.set_diameter(parser.get_part_diameters())
        calculator.set_length(parser.get_part_length())
        print(f"   Элементов: {len(calculator.elem)}")
        print(f"   Полная длина: {calculator.full_length:.2f} м")
    except Exception as e:
        print(f"   Ошибка загрузки: {e}")
        import traceback

        traceback.print_exc()
        return

    # Загрузка форм колебаний
    print("\n2. Загрузка результатов первого расчета...")
    mode_loader = RealModeShapeLoader("amur")
    if not mode_loader.load_frequency_data() or not mode_loader.load_mode_shapes():
        print("   Используются приближенные формы колебаний")
        mode_loader = None

    parallel = ParallelAerodynamics(calculator, mode_loader)

    # Расширенный список точек
    points = [
        # Старт и начальный участок (плотные слои атмосферы)
        {"V": 500, "t": 25, "H": 5, "name": "Старт (500 м/с, 25 с, 5 км)"},
        # Трансзвуковая область (максимальные аэродинамические нагрузки)
        {"V": 680, "t": 62, "H": 24, "name": "Околозвук (680 м/с, M≈2.0, 24 км)"},
        {"V": 720, "t": 65, "H": 26, "name": "Max Q область (720 м/с, 65 с, 26 км)"},
        {"V": 750, "t": 67, "H": 27, "name": "Пик нагрузок (750 м/с, 67 с, 27 км)"},
        # Работа первой ступени (основные нагрузки)
        {
            "V": 800,
            "t": 70,
            "H": 28,
            "name": "Середина работы 1 ступени (800 м/с, 70 с, 28 км)",
        },
        {"V": 900, "t": 78, "H": 33, "name": "Сверхзвук (900 м/с, 78 с, 33 км)"},
        {
            "V": 1000,
            "t": 85,
            "H": 38,
            "name": "Окончание плотных слоев (1000 м/с, 85 с, 38 км)",
        },
    ]

    attack_angle = 4.0

    print(f"\nПараметры расчета:")
    print(f"   Угол атаки: {attack_angle}°")
    print(f"\nРасчетных точек: {len(points)}")

    # Списки для сбора всех приращений
    all_cx_changes = []
    all_cyy_changes = []
    all_xf_changes = []
    all_alpha_add = []

    # Заголовок таблицы
    print("\n" + "=" * 140)
    print(
        f"{'Точка':<35} {'Cx (старый)':>12} {'Cx (новый)':>12} {'Cx изм,%':>10} "
        f"{'Cyy (старый)':>12} {'Cyy (новый)':>12} {'Cyy изм,%':>10} "
        f"{'Xf (старый)':>10} {'Xf (новый)':>10} {'Xf изм,%':>8} {'Δα, °':>6}"
    )
    print("=" * 140)

    for point in points:
        V, t, H = point["V"], point["t"], point["H"]

        # Создаем списки для одной точки
        velocities = [V]
        altitudes = [H]

        # Жесткий расчет
        results_rigid = parallel.calculate_range(
            velocities, altitudes, attack_angle, False
        )

        # Упругий расчет
        results_elastic = parallel.calculate_range(
            velocities, altitudes, attack_angle, True, t
        )

        if not results_rigid or not results_elastic:
            print(
                f"{point['name'][:33]:<33} {'Ошибка':>12} {'Ошибка':>12} {'Ошибка':>10} "
                f"{'Ошибка':>12} {'Ошибка':>12} {'Ошибка':>10} "
                f"{'Ошибка':>10} {'Ошибка':>10} {'Ошибка':>8} {'Ошибка':>6}"
            )
            continue

        result_rigid = results_rigid[0]
        result_elastic = results_elastic[0]

        # Cx
        cx_rigid = result_rigid.CX
        cx_elastic = (
            result_elastic.CX_elastic if result_elastic.CX_elastic else cx_rigid
        )
        cx_change = (cx_elastic - cx_rigid) / cx_rigid * 100 if cx_rigid != 0 else 0

        # Cyy
        cyy_rigid = result_rigid.CYY
        cyy_elastic = (
            result_elastic.CYY_elastic if result_elastic.CYY_elastic else cyy_rigid
        )
        cyy_change = (
            (cyy_elastic - cyy_rigid) / cyy_rigid * 100 if cyy_rigid != 0 else 0
        )

        # Xf (положение фокуса в долях длины)
        xf_rigid = result_rigid.focus_rigid if result_rigid.focus_rigid else 0
        xf_elastic = (
            result_elastic.focus_elastic if result_elastic.focus_elastic else xf_rigid
        )
        xf_change = (xf_elastic - xf_rigid) * 100  # в процентах от длины

        # Дополнительный угол атаки
        alpha_add = (
            result_elastic.additional_angle_mean
            if result_elastic.additional_angle_mean
            else 0
        )

        # Сохраняем для статистики (абсолютные значения)
        all_cx_changes.append(abs(cx_change))
        all_cyy_changes.append(abs(cyy_change))
        all_xf_changes.append(abs(xf_change))
        all_alpha_add.append(abs(alpha_add))

        # Вывод строки таблицы
        print(
            f"{point['name'][:33]:<33} "
            f"{cx_rigid:>12.6f} {cx_elastic:>12.6f} {cx_change:>+9.2f}% "
            f"{cyy_rigid:>12.6f} {cyy_elastic:>12.6f} {cyy_change:>+9.2f}% "
            f"{xf_rigid:>10.4f} {xf_elastic:>10.4f} {xf_change:>+7.2f}% {alpha_add:>+5.2f}"
        )

    print("=" * 140)

    # Вывод максимальных приращений
    print("\n" + "=" * 80)
    print("МАКСИМАЛЬНЫЕ ПРИРАЩЕНИЯ ПО ВСЕМ ТОЧКАМ")
    print("=" * 80)

    if all_cx_changes:
        max_cx_idx = np.argmax(all_cx_changes)
        print(
            f"Максимальное изменение Cx: {all_cx_changes[max_cx_idx]:.2f}% "
            f"(в точке: {points[max_cx_idx]['name']})"
        )
    else:
        print("Максимальное изменение Cx: нет данных")

    if all_cyy_changes:
        max_cyy_idx = np.argmax(all_cyy_changes)
        print(
            f"Максимальное изменение Cyy: {all_cyy_changes[max_cyy_idx]:.2f}% "
            f"(в точке: {points[max_cyy_idx]['name']})"
        )
    else:
        print("Максимальное изменение Cyy: нет данных")

    if all_xf_changes:
        max_xf_idx = np.argmax(all_xf_changes)
        print(
            f"Максимальное смещение фокуса: {all_xf_changes[max_xf_idx]:.2f}% "
            f"(в точке: {points[max_xf_idx]['name']})"
        )
    else:
        print("Максимальное смещение фокуса: нет данных")

    if all_alpha_add:
        max_alpha_idx = np.argmax(all_alpha_add)
        print(
            f"Максимальный доп. угол атаки: {all_alpha_add[max_alpha_idx]:.2f}° "
            f"(в точке: {points[max_alpha_idx]['name']})"
        )
    else:
        print("Максимальный доп. угол атаки: нет данных")

    print("\n" + "=" * 80)
    print("РАСЧЕТ ЗАВЕРШЕН")
    print("=" * 80)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
