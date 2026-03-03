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

# Импорт пользовательских модулей
import atmosphere
import path
import rocket_parser as rp
import constants

warnings.filterwarnings("ignore")

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
    x_start: float = 0.0  # Начальная координата элемента
    x_end: float = 0.0  # Конечная координата элемента

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
        self.x_coords: List[float] = []  # Координаты для дискретизации

    def set_elnumber(self, n: int):
        """Установка количества элементов"""
        self.elem = [Element() for _ in range(n)]

    def set_length(self, lengths: List[float]):
        """Установка длин элементов"""
        for i, l in enumerate(lengths):
            if i < len(self.elem):
                self.elem[i].elem_length = l
        self.pre_calculations()

    def set_diameter(self, diameters: List[float]):
        """Установка диаметров элементов"""
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
        """Предварительные вычисления"""
        self.full_length = 0.0
        self.full_round_area = 0.0

        current_position = 0.0
        for i, e in enumerate(self.elem):
            e.x_start = current_position
            e.x_end = current_position + e.elem_length

            if i == 0:
                # Носовой обтекатель
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
                # Цилиндрические и конические секции
                d_diff = e.lower_diameter - e.upper_diameter
                e.base_line = math.sqrt(e.elem_length**2 + (d_diff**2) / 4)
                e.round_area = (
                    self.PI * (e.upper_diameter + e.lower_diameter) * e.base_line / 2
                )
                e.focus_position = current_position + e.elem_length / 2

            e.upper_area = self.PI * (e.upper_diameter / 2) ** 2
            e.lower_area = self.PI * (e.lower_diameter / 2) ** 2

            # Виртуальная длина
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

            # Удлинение
            if e.upper_diameter < e.lower_diameter and abs(e.upper_diameter) > 0.1:
                e.ratio = e.virtual_length / e.lower_diameter
            else:
                if e.lower_diameter != 0:
                    e.ratio = e.elem_length / e.lower_diameter
                else:
                    e.ratio = 0

            self.full_length += e.elem_length
            self.full_round_area += e.round_area
            current_position += e.elem_length

        # Финальные параметры
        if self.elem:
            if self.elem[-1].upper_diameter != 0:
                self.full_ratio = self.full_length / self.elem[-1].upper_diameter
            else:
                self.full_ratio = 0
            self.midel_diameter = self.elem[-1].upper_diameter
            self.midel_area = self.elem[-1].lower_area

        # Генерация координат для дискретизации (шаг 0.1 м)
        self.x_coords = np.arange(0, self.full_length, 0.1).tolist()
        if self.x_coords[-1] < self.full_length:
            self.x_coords.append(self.full_length)


# ============================================================================
# КЛАСС ФОРМ КОЛЕБАНИЙ
# ============================================================================


@dataclass
class ModeShape:
    """Форма колебаний"""

    name: str  # Название моды
    frequency: float  # Собственная частота, Гц
    damping: float = 0.02  # Коэффициент демпфирования для этой моды
    nodes: List[float] = None  # Координаты узлов
    values: List[float] = None  # Значения формы

    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []
        if self.values is None:
            self.values = []
        self.omega = 2 * math.pi * self.frequency

    def interpolate(self, x: float) -> float:
        """Интерполяция формы в заданной точке"""
        if not self.nodes or not self.values:
            return 0.0
        return np.interp(x, self.nodes, self.values)

    def derivative(self, x: float, dx: float = 0.01) -> float:
        """Производная формы в заданной точке (численно)"""
        return (self.interpolate(x + dx) - self.interpolate(x - dx)) / (2 * dx)


class ModeShapeGenerator:
    """Генератор форм колебаний для свободно-свободной балки"""

    @staticmethod
    def generate_first_bending(L: float, num_points: int = 100) -> ModeShape:
        """
        Первая изгибная мода (симметричная)
        φ₁(x) = sin(πx/L)
        Максимум на концах, узел в центре
        Частота: примерно 15 Гц (типовое значение)
        """
        x_nodes = np.linspace(0, L, num_points)
        # Нормировка: максимальное значение = 1
        values = [math.sin(math.pi * x / L) for x in x_nodes]
        # Нормируем
        max_val = max(abs(v) for v in values)
        values = [v / max_val for v in values]

        return ModeShape(
            name="1st Bending",
            frequency=15.0,
            damping=0.02,
            nodes=x_nodes.tolist(),
            values=values,
        )

    @staticmethod
    def generate_second_bending(L: float, num_points: int = 100) -> ModeShape:
        """
        Вторая изгибная мода (антисимметричная)
        φ₂(x) = sin(2πx/L)
        Два узла внутри, максимумы и минимумы
        Частота: примерно 40 Гц (типовое значение)
        """
        x_nodes = np.linspace(0, L, num_points)
        values = [math.sin(2 * math.pi * x / L) for x in x_nodes]
        # Нормируем
        max_val = max(abs(v) for v in values)
        values = [v / max_val for v in values]

        return ModeShape(
            name="2nd Bending",
            frequency=40.0,
            damping=0.03,
            nodes=x_nodes.tolist(),
            values=values,
        )

    @staticmethod
    def generate_first_torsion(L: float, num_points: int = 100) -> ModeShape:
        """
        Первая крутильная мода
        φ₃(x) = 2x/L - 1 (линейная)
        Ноль в центре, -1 на одном конце, +1 на другом
        Частота: примерно 25 Гц (типовое значение)
        """
        x_nodes = np.linspace(0, L, num_points)
        values = [2 * x / L - 1 for x in x_nodes]

        return ModeShape(
            name="1st Torsion",
            frequency=25.0,
            damping=0.01,
            nodes=x_nodes.tolist(),
            values=values,
        )


# ============================================================================
# КЛАСС ТРЕНИЯ
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
        """Расчет параметров пограничного слоя"""
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
                if abs(Re) < 0.0001:
                    self.cif = 0
                else:
                    self.cif = 2.656 / math.sqrt(Re)
            self.num = pow(1 + 0.1 * pow(Mach, 0.1), -2 / 3)
        else:
            self.cif = 0.91 / pow(math.log10(Re), 2.58)
            self.num = pow(1 + 0.1 * pow(Mach, 0.1), -2 / 3)

    def fricalc(self, Mach: float, SS: float, nu: float) -> float:
        """Расчет коэффициента трения"""
        if nu is None:
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


# ============================================================================
# КЛАСС ДАВЛЕНИЯ
# ============================================================================


class Pressure(Geometry):
    """Расчет давления"""

    def read_pressure_file(
        self, filename: str, rows: int, cols: int
    ) -> List[List[float]]:
        """Чтение файла с данными давления"""
        try:
            df = pd.read_csv(filename, delimiter=None, engine="python", header=None)
            if rows > 0:
                df = df.head(rows)

            if df.shape[1] < cols:
                raise ValueError(f"File {filename} has fewer than {cols} columns")

            df = df.iloc[:, :cols]
            data = df.values.tolist()
            transposed = list(map(list, zip(*data)))
            return transposed
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return []

    def interpolate_Mach(self, Mach: float, data: List[List[float]]) -> float:
        """Линейная интерполяция по числу Маха"""
        if not data or len(data) < 2:
            return 0.0
        Mach_v = data[0]
        values = data[1]
        for i in range(1, len(Mach_v)):
            if Mach >= Mach_v[i - 1] and Mach < Mach_v[i]:
                return values[i - 1] + (Mach - Mach_v[i - 1]) * (
                    values[i] - values[i - 1]
                ) / (Mach_v[i] - Mach_v[i - 1])
        return values[-1] if values else 0.0

    def select_ratio_data_pressure(
        self, ratio: float, data: List[List[float]]
    ) -> List[List[float]]:
        """Выбор данных по удлинению для давления"""
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
        """Выбор данных по удлинению для конических частей"""
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
        """Донное давление"""
        if Mach < 1:
            if self.cif * self.num * self.full_ratio == 0:
                return 0
            return 0.0155 / math.sqrt(self.cif * self.num * self.full_ratio)
        else:
            data = self.read_pressure_file(path.root_path + "HeadPressure.csv", 10, 10)
            if not data:
                return 0
            return self.interpolate_Mach(Mach, [data[0], data[6]])

    def head_Cpres(self, Mach: float) -> float:
        """Давление на носовой части"""
        data = self.read_pressure_file(path.root_path + "HeadPressure.csv", 10, 10)
        if not data or not self.elem:
            return 0
        ratio = self.elem[0].ratio
        H_current = self.select_ratio_data_pressure(ratio, data)
        return self.interpolate_Mach(Mach, H_current)

    def triangle_Cpres(self, Mach: float, ratio: float) -> float:
        """Давление на конической части"""
        data = self.read_pressure_file(path.root_path + "TrianglePressure.csv", 10, 7)
        if not data:
            return 0
        H_current = self.select_ratio_data_triangle(ratio, data)
        return self.interpolate_Mach(Mach, H_current)

    def prescalc(self, Mach: float) -> float:
        """Полный расчет давления"""
        res = 0.0
        for i in range(1, len(self.elem)):
            if self.elem[i].lower_area != 0:
                self.elem[i].C_pres = self.triangle_Cpres(Mach, self.elem[i].ratio) * (
                    1 - self.elem[i].upper_area / self.elem[i].lower_area
                )
                res += self.elem[i].C_pres
        return res + self.head_Cpres(Mach) + self.bottom_pres(Mach)


# ============================================================================
# КЛАСС ИНДУКТИВНОСТИ
# ============================================================================


class Inductance(Geometry):
    """Расчет индуктивного сопротивления"""

    def read_pressure_file(
        self, filename: str, rows: int, cols: int
    ) -> List[List[float]]:
        """Чтение файла с данными"""
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
            transposed = list(map(list, zip(*data))) if data else []
            return transposed
        except:
            return []

    def E_pressure(self, angle: float, Mach: float) -> float:
        """Расчет индуктивного давления"""
        N = 10
        data = self.read_pressure_file(path.root_path + "EPressure.csv", N, 3)
        if not data or len(data) < 3:
            return 0.0

        Mach_v = data[0]
        H_head = data[1]
        H_cone = data[2]

        if not self.elem:
            return 0.0

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

        # Интерполяция
        E_head = 0
        E_cone = 0
        for i in range(1, len(Mach_v)):
            if Mach_val >= Mach_v[i - 1] and Mach_val < Mach_v[i]:
                E_head = H_head[i - 1] + (Mach_val - Mach_v[i - 1]) * (
                    H_head[i] - H_head[i - 1]
                ) / (Mach_v[i] - Mach_v[i - 1])
                E_cone = H_cone[i - 1] + (Mach_val - Mach_v[i - 1]) * (
                    H_cone[i] - H_cone[i - 1]
                ) / (Mach_v[i] - Mach_v[i - 1])

        # Расчет для носовой части
        self.elem[0].C_ind = (
            self.elem[0].CY + constants.rad(2 * E_head)
        ) * constants.sqr(angle)

        # Расчет для остальных частей
        for j in range(1, len(self.elem)):
            if self.elem[j].upper_diameter < self.elem[j].lower_diameter:
                if self.elem[-1].upper_area != 0:
                    ratio = self.elem[-1].upper_area
                    self.elem[j].C_ind = (
                        self.elem[j].CY * self.elem[j].upper_area / ratio
                        + constants.rad(2 * E_cone * self.elem[j].upper_area / ratio)
                    ) * constants.sqr(angle)

        E = sum(e.C_ind for e in self.elem)
        return E


# ============================================================================
# КЛАСС ПОДЪЕМНОЙ СИЛЫ
# ============================================================================


class LiftForce(Inductance):
    """Расчет подъемной силы"""

    def sqr(self, x: float) -> float:
        return x * x

    def rad(self, x: float) -> float:
        return x / 57.3

    def head_lift(self, Mach: float) -> float:
        """Подъемная сила носовой части"""
        N = 9
        data = self.read_pressure_file(path.root_path + "HeadNormal.csv", N, 6)
        if not data or len(data) < 6:
            return 0.035

        Mah_v = data[0]
        H_0 = data[1]
        H_05 = data[2]
        H_1 = data[3]
        H_2 = data[4]
        H_4 = data[5]

        if not self.elem:
            return 0.035

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
        """Подъемная сила свободного конуса"""
        if index >= len(self.elem):
            return 0.0
        denominator = self.elem[index].virtual_length - self.elem[index].elem_length
        if abs(denominator) > 1e-6:
            arg = self.elem[index].lower_diameter / 2 / denominator
            Q = math.atan(arg)
            return (2 / 57.3) * constants.sqr(math.cos(Q))
        return 0.0

    def triangle_lift(self, Mach: float, ratio: float, index: int) -> float:
        """Подъемная сила конической части"""
        N = 9
        data = self.read_pressure_file(path.root_path + "TriangleNormal.csv", N, 6)
        if not data or len(data) < 6:
            return 0.0

        Mah_v = data[0]
        H_0 = data[1]
        H_1 = data[2]
        H_2 = data[3]
        H_3 = data[4]
        H_4 = data[5]

        if index >= len(self.elem):
            return 0.0

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
        """Расчет полного коэффициента CY"""
        return self.head_lift(Mach) + self.un_triangle_lift(Mach)

    def un_triangle_lift(self, Mach: float) -> float:
        """Расчет подъемной силы для конических частей"""
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


# ============================================================================
# КЛАСС СИЛЫ СОПРОТИВЛЕНИЯ
# ============================================================================


class DragForce(Friction, Pressure):
    """Расчет силы сопротивления"""

    def calculate_CX(self, Mach: float, SS: float, nu: float) -> float:
        """Расчет коэффициента сопротивления"""
        return self.fricalc(Mach, SS, nu) + self.prescalc(Mach)


# ============================================================================
# РАСШИРЕННЫЙ КЛАСС АЭРОУПРУГОСТИ С ФОРМАМИ КОЛЕБАНИЙ
# ============================================================================


@dataclass
class AeroElasticParams:
    """Параметры аэроупругости"""

    mass_per_length: float = 200.0  # Погонная масса, кг/м
    bending_stiffness: float = 5e6  # Изгибная жесткость, Н·м²
    torsional_stiffness: float = 8e6  # Крутильная жесткость, Н·м²/рад


class AdvancedAeroElasticity:
    """
    Расчет аэроупругости с учетом первых трех форм колебаний
    """

    def __init__(self, geometry):
        self.geom = geometry
        self.modes = self._generate_mode_shapes()
        self.params = AeroElasticParams()

    def _generate_mode_shapes(self) -> List[ModeShape]:
        """Генерация первых трех форм колебаний"""
        L = self.geom.full_length
        if L <= 0:
            return []

        modes = []

        # 1-я изгибная мода (симметричная)
        mode1 = ModeShapeGenerator.generate_first_bending(L)
        modes.append(mode1)

        # 2-я изгибная мода (антисимметричная)
        mode2 = ModeShapeGenerator.generate_second_bending(L)
        modes.append(mode2)

        # 1-я крутильная мода
        mode3 = ModeShapeGenerator.generate_first_torsion(L)
        modes.append(mode3)

        return modes

    def _get_local_diameter(self, x: float) -> float:
        """Получение локального диаметра"""
        for elem in self.geom.elem:
            if elem.x_start <= x <= elem.x_end:
                t = (x - elem.x_start) / elem.elem_length
                return elem.upper_diameter * (1 - t) + elem.lower_diameter * t
        return self.geom.midel_diameter

    def _get_local_chord(self, x: float) -> float:
        """Получение локальной хорды (для угла атаки)"""
        return self._get_local_diameter(x)

    def _get_local_area(self, x: float) -> float:
        """Получение локальной площади поперечного сечения"""
        d = self._get_local_diameter(x)
        return math.pi * (d / 2) ** 2

    def _calculate_generalized_mass(self, mode: ModeShape) -> float:
        """
        Расчет обобщенной массы для моды
        M_i = ∫ m(x) φ_i²(x) dx
        """
        if not self.geom.x_coords:
            return 0.0

        M = 0.0
        for i in range(len(self.geom.x_coords) - 1):
            x1 = self.geom.x_coords[i]
            x2 = self.geom.x_coords[i + 1]
            dx = x2 - x1

            # Среднее значение формы на интервале
            phi1 = mode.interpolate(x1)
            phi2 = mode.interpolate(x2)
            phi_avg = (phi1 + phi2) / 2

            # Погонная масса (может быть уточнена)
            m_line = self.params.mass_per_length

            M += m_line * phi_avg**2 * dx

        return M

    def _calculate_generalized_force(
        self, mode: ModeShape, q: float, CYY_dist: List[float]
    ) -> float:
        """
        Расчет обобщенной аэродинамической силы для моды
        F_i = ∫ p(x) φ_i(x) dx, где p(x) = q * S_local(x) * CYY_local(x)
        """
        if not self.geom.x_coords or not CYY_dist:
            return 0.0

        F = 0.0
        for i, x in enumerate(self.geom.x_coords[:-1]):
            x1 = x
            x2 = self.geom.x_coords[i + 1]
            dx = x2 - x1

            # Локальная подъемная сила
            S_local = self._get_local_area(x1)
            p_local = q * S_local * CYY_dist[i]

            # Значение формы
            phi = mode.interpolate(x1)

            F += p_local * phi * dx

        return F

    def calculate_mode_amplitudes(
        self, velocity: float, altitude: float, CYY_dist: List[float]
    ) -> List[float]:
        """
        Расчет амплитуд для каждой моды
        q_i = F_i / (M_i * ω_i²)
        """
        atmos = atmosphere.atmosphere(altitude)
        if atmos.get_SV() is None:
            return [0.0] * len(self.modes)

        rho = atmos.get_density()
        q = 0.5 * rho * velocity**2

        amplitudes = []
        for mode in self.modes:
            M_mode = self._calculate_generalized_mass(mode)
            F_mode = self._calculate_generalized_force(mode, q, CYY_dist)

            if M_mode > 0 and mode.omega > 0:
                # Статическая амплитуда: q_i = F_i / (M_i * ω_i²)
                amplitude = F_mode / (M_mode * mode.omega**2)
            else:
                amplitude = 0.0

            amplitudes.append(amplitude)

        return amplitudes

    def calculate_local_gain(self, x: float, amplitudes: List[float]) -> float:
        """
        Расчет локального коэффициента усиления в точке x
        Учитывает вклад всех мод
        """
        if not self.modes or not amplitudes:
            return 1.0

        # Суммарная деформация в точке
        deformation = 0.0
        for mode, amp in zip(self.modes, amplitudes):
            deformation += amp * mode.interpolate(x)

        # Производная деформации (изменение угла)
        dalpha = 0.0
        for mode, amp in zip(self.modes, amplitudes):
            dalpha += amp * mode.derivative(x)

        # Локальный коэффициент усиления
        # Учитываем как деформацию, так и изменение угла
        gain = 1.0 + abs(deformation) * 5.0 + abs(dalpha) * 10.0

        return min(gain, 2.0)  # Ограничиваем

    def calculate_focus_shift(
        self,
        velocity: float,
        altitude: float,
        CYY_dist: List[float],
        original_focus: float,
    ) -> float:
        """
        Расчет смещения фокуса (центра давления) под влиянием аэроупругости

        Положение фокуса: X_f = ∫ x * p(x) dx / ∫ p(x) dx
        где p(x) - распределение подъемной силы
        """
        if not self.geom.x_coords or not CYY_dist:
            return original_focus

        atmos = atmosphere.atmosphere(altitude)
        if atmos.get_SV() is None:
            return original_focus

        rho = atmos.get_density()
        q = 0.5 * rho * velocity**2

        # Получаем амплитуды мод
        amplitudes = self.calculate_mode_amplitudes(velocity, altitude, CYY_dist)

        # Расчет распределения подъемной силы с учетом деформаций
        moment_sum = 0.0
        force_sum = 0.0

        for i, x in enumerate(self.geom.x_coords[:-1]):
            x1 = x
            x2 = self.geom.x_coords[i + 1]
            dx = x2 - x1
            x_center = (x1 + x2) / 2

            # Базовая локальная подъемная сила
            S_local = self._get_local_area(x_center)
            p_base = q * S_local * CYY_dist[i]

            # Коэффициент усиления из-за деформации
            gain = self.calculate_local_gain(x_center, amplitudes)

            # Итоговая подъемная сила
            p_total = p_base * gain

            force_sum += p_total * dx
            moment_sum += p_total * x_center * dx

        # Новое положение фокуса
        if force_sum > 0:
            new_focus = moment_sum / force_sum
        else:
            new_focus = original_focus

        return new_focus

    def calculate_distributed_corrections(
        self,
        velocity: float,
        altitude: float,
        Mach: float,
        CX: float,
        CYY: float,
        CYY_dist: List[float],
    ) -> Dict[str, any]:
        """
        Расчет распределенных аэроупругих поправок
        """
        # Базовый результат
        result = {
            "CX_corr": 0.0,
            "CY_corr": 0.0,
            "gain_global": 1.0,
            "gain_dist": [],
            "additional_angle_dist": [],  # Дополнительный угол атаки по длине
            "amplitudes": [],
            "mode_names": [],
            "focus_shift": 0.0,  # Смещение фокуса
            "focus_new": 0.0,  # Новое положение фокуса
        }

        if not self.geom.x_coords or not CYY_dist:
            return result

        # Расчет амплитуд мод
        amplitudes = self.calculate_mode_amplitudes(velocity, altitude, CYY_dist)
        result["amplitudes"] = amplitudes
        result["mode_names"] = [mode.name for mode in self.modes]

        # Расчет локальных усилений и дополнительных углов атаки
        local_gains = []
        additional_angles = []
        for x in self.geom.x_coords:
            gain = self.calculate_local_gain(x, amplitudes)
            local_gains.append(gain)

            # Дополнительный угол атаки от деформации (в градусах)
            dalpha = 0.0
            for mode, amp in zip(self.modes, amplitudes):
                dalpha += amp * mode.derivative(x)
            additional_angle = math.degrees(dalpha)  # переводим в градусы
            additional_angles.append(additional_angle)

        result["gain_dist"] = local_gains
        result["additional_angle_dist"] = additional_angles
        result["gain_global"] = np.mean(local_gains)

        # Расчет исходного фокуса (положения центра давления)
        original_focus = self._calculate_focus_position(CYY_dist)

        # Расчет нового фокуса с учетом аэроупругости
        new_focus = self.calculate_focus_shift(
            velocity, altitude, CYY_dist, original_focus
        )
        result["focus_shift"] = new_focus - original_focus
        result["focus_new"] = new_focus

        # Интегрирование поправок по длине
        L = self.geom.full_length
        cx_corr_total = 0.0
        cy_corr_total = 0.0

        for i, x in enumerate(self.geom.x_coords[:-1]):
            x1 = x
            x2 = self.geom.x_coords[i + 1]
            dx = x2 - x1

            gain_local = local_gains[i]

            # Локальные коэффициенты (интерполируем)
            cyy_local = CYY_dist[i] if i < len(CYY_dist) else CYY

            # Локальные поправки
            cx_corr_local = CX * (gain_local - 1) * 0.1 * dx / L
            cy_corr_local = cyy_local * (gain_local - 1) * 0.3 * dx / L

            cx_corr_total += cx_corr_local
            cy_corr_total += cy_corr_local

        result["CX_corr"] = cx_corr_total
        result["CY_corr"] = cy_corr_total

        return result

    def _calculate_focus_position(self, CYY_dist: List[float]) -> float:
        """
        Расчет положения фокуса (центра давления)
        X_f = ∫ x * CYY(x) dx / ∫ CYY(x) dx
        """
        if not self.geom.x_coords or not CYY_dist:
            return self.geom.full_length / 2

        moment_sum = 0.0
        force_sum = 0.0

        for i, x in enumerate(self.geom.x_coords[:-1]):
            x1 = x
            x2 = self.geom.x_coords[i + 1]
            dx = x2 - x1
            x_center = (x1 + x2) / 2

            cyy_local = CYY_dist[i] if i < len(CYY_dist) else 0

            force_sum += cyy_local * dx
            moment_sum += cyy_local * x_center * dx

        if force_sum > 0:
            return moment_sum / force_sum
        else:
            return self.geom.full_length / 2

    def print_mode_info(self):
        """Вывод информации о модах"""
        print("\nИнформация о формах колебаний:")
        print("-" * 50)
        for i, mode in enumerate(self.modes):
            print(f"Мода {i+1}: {mode.name}")
            print(f"  Частота: {mode.frequency:.1f} Гц")
            print(f"  Демпфирование: {mode.damping:.3f}")

            # Находим узлы (где форма = 0)
            nodes = []
            for j in range(len(mode.nodes) - 1):
                if mode.values[j] * mode.values[j + 1] <= 0:
                    nodes.append(f"{mode.nodes[j]:.2f}")
            if nodes:
                print(f"  Узлы: {', '.join(nodes)} м")


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
        self.CYY_distribution = []  # Распределение CYY по длине

    def calculate_CXY(self, velocity: float, altitude: float, attack_angle: float):
        """Расчет коэффициентов CX и CY"""
        A = atmosphere.atmosphere(altitude)
        if A.get_SV() is not None:
            Mach = velocity / A.get_SV()
            self.CX = self.calculate_CX(Mach, A.get_SV(), A.get_dyn())
            self.CYY = self.calculate_CY(Mach)
            self.E = self.E_pressure(attack_angle, Mach)
            self.CX += self.CYY + self.E
            self.CY = self.CYY * (attack_angle / 57.3)  # перевод в радианы

            # Генерируем распределение CYY по длине (для аэроупругости)
            self._generate_CYY_distribution()

    def _generate_CYY_distribution(self):
        """Генерация распределения CYY по длине"""
        self.CYY_distribution = []
        for x in self.x_coords:
            # Определяем элемент
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
    """Результаты аэродинамического расчета"""

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
    additional_angle_mean: Optional[float] = None  # Средний дополнительный угол атаки
    additional_angle_max: Optional[float] = (
        None  # Максимальный дополнительный угол атаки
    )
    mode_amplitudes: Optional[List[float]] = None
    mode_names: Optional[List[str]] = None
    focus_rigid: Optional[float] = None  # Положение фокуса без учета аэроупругости
    focus_elastic: Optional[float] = None  # Положение фокуса с учетом аэроупругости
    focus_shift: Optional[float] = None  # Смещение фокуса


# ============================================================================
# КЛАСС ПАРАЛЛЕЛЬНОГО РАСЧЕТА
# ============================================================================


class ParallelAerodynamics:
    """Параллельный аэродинамический расчет"""

    def __init__(self, base_calculator):
        self.base_calc = base_calculator
        self.num_workers = mp.cpu_count()
        self.aero_elastic = AdvancedAeroElasticity(base_calculator)

    def calculate_range(
        self,
        velocities: np.ndarray,
        altitudes: List[float],
        attack_angle: float = 2.0,
        use_elastic: bool = False,
    ) -> List[AeroResult]:
        """
        Параллельный расчет для диапазона скоростей и высот
        """
        # Подготавливаем аргументы
        args_list = []
        for alt in altitudes:
            for vel in velocities:
                args_list.append((vel, alt * 1000, attack_angle, use_elastic))

        print(f"   Запуск на {self.num_workers} процессах, точек: {len(args_list)}")

        # Параллельный расчет
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
        """Расчет одной точки"""
        velocity, altitude, attack_angle, use_elastic = args

        # Создаем копию калькулятора
        calc = copy.deepcopy(self.base_calc)

        # Базовый расчет
        calc.calculate_CXY(velocity, altitude, attack_angle)

        # Получаем атмосферные параметры
        A = atmosphere.atmosphere(altitude)
        Mach = velocity / A.get_SV() if A.get_SV() else 0

        # Расчет положения фокуса без учета аэроупругости
        focus_rigid = self.aero_elastic._calculate_focus_position(calc.CYY_distribution)

        result = AeroResult(
            velocity=velocity,
            altitude=altitude,
            Mach=Mach,
            CX=calc.CX,
            CYY=calc.CYY,
            CY=calc.CY,
            focus_rigid=focus_rigid / calc.full_length,  # Относительная координата
        )

        # Аэроупругие поправки с формами колебаний
        if use_elastic:
            corrections = self.aero_elastic.calculate_distributed_corrections(
                velocity, altitude, Mach, calc.CX, calc.CYY, calc.CYY_distribution
            )

            result.CX_elastic = calc.CX + corrections["CX_corr"]
            result.CYY_elastic = calc.CYY + corrections["CY_corr"]
            result.CY_elastic = calc.CY + corrections["CY_corr"]
            result.gain = corrections["gain_global"]

            # Дополнительные углы атаки
            if corrections["additional_angle_dist"]:
                result.additional_angle_mean = np.mean(
                    np.abs(corrections["additional_angle_dist"])
                )
                result.additional_angle_max = np.max(
                    np.abs(corrections["additional_angle_dist"])
                )

            # Положение фокуса с учетом аэроупругости
            result.focus_elastic = corrections["focus_new"] / calc.full_length
            result.focus_shift = corrections["focus_shift"] / calc.full_length

            result.mode_amplitudes = corrections["amplitudes"]
            result.mode_names = corrections["mode_names"]

        return result


# ============================================================================
# КЛАСС ВИЗУАЛИЗАЦИИ (5 ГРАФИКОВ - ДОБАВЛЕН ГРАФИК ФОКУСА)
# ============================================================================


class SimpleVisualizer:
    """
    Визуализация с пятью основными графиками (добавлен график положения фокуса)
    """

    def __init__(self):
        self.fig_size = (25, 15)  # Увеличен размер для 5 графиков
        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
        ]  # Синий, оранжевый, зеленый, красный, фиолетовый

    def plot_results(
        self,
        results_rigid: List[AeroResult],
        results_elastic: List[AeroResult],
        altitudes: List[float],
    ):
        """
        Построение пяти графиков:
        1. Коэффициент подъемной силы CYY (сравнение)
        2. Коэффициент лобового сопротивления CX (сравнение)
        3. Процент влияния аэроупругости на CYY
        4. Дополнительный угол атаки от аэроупругости
        5. Положение фокуса (центра давления)
        """
        fig, axes = plt.subplots(2, 3, figsize=self.fig_size)  # 2x3 сетка
        axes = axes.flatten()

        # Скрываем шестой (пустой) подграфик
        axes[5].set_visible(False)

        for idx, alt in enumerate(altitudes):
            # Фильтруем данные для текущей высоты
            rigid_alt = [r for r in results_rigid if abs(r.altitude / 1000 - alt) < 0.1]
            elastic_alt = [
                r for r in results_elastic if abs(r.altitude / 1000 - alt) < 0.1
            ]

            if not rigid_alt or not elastic_alt:
                continue

            # Сортируем по скорости
            rigid_alt.sort(key=lambda x: x.velocity)
            elastic_alt.sort(key=lambda x: x.velocity)

            velocities = [r.velocity for r in rigid_alt]

            color = self.colors[idx % len(self.colors)]
            label = f"H = {alt} км"

            # ГРАФИК 1: Коэффициент подъемной силы CYY
            ax = axes[0]
            # Без учета аэроупругости (пунктир)
            ax.plot(
                velocities,
                [r.CYY for r in rigid_alt],
                "--",
                color=color,
                linewidth=2,
                alpha=0.7,
                label=f"{label} (без упр.)",
            )
            # С учетом аэроупругости (сплошная)
            ax.plot(
                velocities,
                [r.CYY_elastic for r in elastic_alt],
                "-",
                color=color,
                linewidth=2,
                label=f"{label} (с упр.)",
            )

            # ГРАФИК 2: Коэффициент лобового сопротивления CX
            ax = axes[1]
            ax.plot(
                velocities,
                [r.CX for r in rigid_alt],
                "--",
                color=color,
                linewidth=2,
                alpha=0.7,
            )
            ax.plot(
                velocities,
                [r.CX_elastic for r in elastic_alt],
                "-",
                color=color,
                linewidth=2,
            )

            # ГРАФИК 3: Процент влияния аэроупругости на CYY
            ax = axes[2]
            influence = [
                ((r_el.CYY_elastic - r_rig.CYY) / r_rig.CYY * 100)
                for r_el, r_rig in zip(elastic_alt, rigid_alt)
            ]
            ax.plot(velocities, influence, "-", color=color, linewidth=2, label=label)

            # ГРАФИК 4: Дополнительный угол атаки от аэроупругости
            ax = axes[3]
            additional_angles_mean = [
                r.additional_angle_mean for r in elastic_alt if r.additional_angle_mean
            ]
            additional_angles_max = [
                r.additional_angle_max for r in elastic_alt if r.additional_angle_max
            ]

            if additional_angles_mean:
                ax.plot(
                    velocities[: len(additional_angles_mean)],
                    additional_angles_mean,
                    "-",
                    color=color,
                    linewidth=2,
                    label=f"{label} (средний)",
                )
                ax.fill_between(
                    velocities[: len(additional_angles_mean)],
                    0,
                    additional_angles_max,
                    color=color,
                    alpha=0.2,
                    label=f"{label} (макс)",
                )

            # ГРАФИК 5: Положение фокуса (X_focus/L)
            ax = axes[4]
            # Без учета аэроупругости (пунктир)
            ax.plot(
                velocities,
                [r.focus_rigid for r in rigid_alt],
                "--",
                color=color,
                linewidth=2,
                alpha=0.7,
                label=f"{label} (без упр.)",
            )
            # С учетом аэроупругости (сплошная)
            focus_elastic_values = [
                r.focus_elastic for r in elastic_alt if r.focus_elastic is not None
            ]
            if focus_elastic_values:
                ax.plot(
                    velocities[: len(focus_elastic_values)],
                    focus_elastic_values,
                    "-",
                    color=color,
                    linewidth=2,
                    label=f"{label} (с упр.)",
                )

        # Настройка графиков
        axes[0].set_xlabel("Скорость, м/с", fontsize=12)
        axes[0].set_ylabel("CYY", fontsize=12)
        axes[0].set_title("Коэффициент подъемной силы CYY", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=9)
        axes[0].set_xlim([0, 2100])

        axes[1].set_xlabel("Скорость, м/с", fontsize=12)
        axes[1].set_ylabel("CX", fontsize=12)
        axes[1].set_title("Коэффициент лобового сопротивления CX", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 2100])

        axes[2].set_xlabel("Скорость, м/с", fontsize=12)
        axes[2].set_ylabel("Влияние аэроупругости, %", fontsize=12)
        axes[2].set_title("Влияние аэроупругости на CYY", fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best", fontsize=9)
        axes[2].axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
        axes[2].set_xlim([0, 2100])

        axes[3].set_xlabel("Скорость, м/с", fontsize=12)
        axes[3].set_ylabel("Дополнительный угол атаки, град", fontsize=12)
        axes[3].set_title("Дополнительный угол атаки от аэроупругости", fontsize=12)
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="best", fontsize=9)
        axes[3].axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
        axes[3].set_xlim([0, 2100])

        axes[4].set_xlabel("Скорость, м/с", fontsize=12)
        axes[4].set_ylabel("X_focus / L", fontsize=12)
        axes[4].set_title("Положение фокуса (центра давления)", fontsize=12)
        axes[4].grid(True, alpha=0.3)
        axes[4].legend(loc="best", fontsize=9)
        axes[4].axhline(
            y=0.5,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
            label="Центр длины",
        )
        axes[4].set_xlim([0, 2100])
        axes[4].set_ylim([0.3, 0.7])

        plt.tight_layout()
        plt.show()

    def print_summary(
        self, results_rigid: List[AeroResult], results_elastic: List[AeroResult]
    ):
        """Вывод краткой статистики"""
        print("\n" + "=" * 60)
        print("КРАТКАЯ СТАТИСТИКА ВЛИЯНИЯ АЭРОУПРУГОСТИ")
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

            # Средние значения
            cyy_rigid = np.mean([r.CYY for r in rigid_alt])
            cyy_elastic = np.mean([r.CYY_elastic for r in elastic_alt])
            cyy_diff = (cyy_elastic - cyy_rigid) / cyy_rigid * 100

            cx_rigid = np.mean([r.CX for r in rigid_alt])
            cx_elastic = np.mean([r.CX_elastic for r in elastic_alt])
            cx_diff = (cx_elastic - cx_rigid) / cx_rigid * 100

            # Максимальное усиление
            gains = [r.gain for r in elastic_alt if r.gain]
            max_gain = max(gains) if gains else 1.0

            # Дополнительные углы атаки
            add_angles_mean = [
                r.additional_angle_mean for r in elastic_alt if r.additional_angle_mean
            ]
            add_angles_max = [
                r.additional_angle_max for r in elastic_alt if r.additional_angle_max
            ]
            max_add_angle = max(add_angles_max) if add_angles_max else 0.0
            mean_add_angle = np.mean(add_angles_mean) if add_angles_mean else 0.0

            # Положение фокуса
            focus_rigid_mean = np.mean([r.focus_rigid for r in rigid_alt])
            focus_elastic_mean = np.mean(
                [r.focus_elastic for r in elastic_alt if r.focus_elastic]
            )
            focus_shift_mean = focus_elastic_mean - focus_rigid_mean

            print(f"\nВысота: {alt} км")
            print(f"  CYY: без упр. = {cyy_rigid:.4f}, с упр. = {cyy_elastic:.4f}")
            print(f"      изменение = {cyy_diff:+.1f}%")
            print(f"  CX:  без упр. = {cx_rigid:.4f}, с упр. = {cx_elastic:.4f}")
            print(f"      изменение = {cx_diff:+.1f}%")
            print(f"  Макс. усиление: {max_gain:.3f}")
            print(
                f"  Доп. угол атаки: средний = {mean_add_angle:.3f}°, макс = {max_add_angle:.3f}°"
            )
            print(
                f"  Положение фокуса: без упр. = {focus_rigid_mean:.3f} L, с упр. = {focus_elastic_mean:.3f} L"
            )
            print(
                f"      смещение = {focus_shift_mean:+.4f} L ({focus_shift_mean*100:+.1f}% от длины)"
            )

            # Информация о модах для первой точки
            if elastic_alt and elastic_alt[0].mode_amplitudes:
                print(f"  Амплитуды мод (при V={elastic_alt[0].velocity:.0f} м/с):")
                for name, amp in zip(
                    elastic_alt[0].mode_names, elastic_alt[0].mode_amplitudes
                ):
                    print(f"    {name}: {amp:.6f}")


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================


def main():
    """Основная функция"""

    print("=" * 60)
    print("АЭРОДИНАМИЧЕСКИЙ РАСЧЕТ С УЧЕТОМ АЭРОУПРУГОСТИ")
    print("(с учетом первых трех форм колебаний)")
    print("=" * 60)

    # Загрузка геометрии ракеты
    print("\n1. Загрузка геометрии ракеты...")
    try:
        parser = rp.rocket_parser()

        # Создание базового калькулятора
        calculator = UnionStream()
        calculator.set_elnumber(parser.get_block_number() + 1)
        calculator.set_diameter(parser.get_part_diameters())
        calculator.set_length(parser.get_part_length())

        print(f"   Элементов: {len(calculator.elem)}")
        print(f"   Полная длина: {calculator.full_length:.2f} м")
        print(f"   Миделево сечение: {calculator.midel_diameter:.3f} м")

    except Exception as e:
        print(f"   Ошибка загрузки: {e}")
        print("   Используется тестовая геометрия")

        # Создаем тестовую геометрию
        calculator = UnionStream()
        calculator.set_elnumber(3)
        calculator.set_diameter([1.0, 1.0, 0.8])
        calculator.set_length([2.0, 5.0, 3.0])
        print(f"   Тестовая геометрия: L={calculator.full_length:.1f} м")

    # Параметры расчета
    velocities = np.linspace(50, 2000, 30)  # 30 точек по скорости
    altitudes = [0, 20, 40, 70]  # км
    attack_angle = 2.0  # градусы

    print(f"\n2. Параметры расчета:")
    print(f"   Скорости: {velocities[0]:.0f} - {velocities[-1]:.0f} м/с")
    print(f"   Высоты: {altitudes} км")
    print(f"   Угол атаки: {attack_angle}°")
    print(f"   Всего точек: {len(velocities) * len(altitudes)}")

    # Информация о формах колебаний
    aero_elastic = AdvancedAeroElasticity(calculator)
    aero_elastic.print_mode_info()

    # Параллельный расчет
    parallel = ParallelAerodynamics(calculator)

    # Расчет без аэроупругости
    print(f"\n3. Расчет без учета аэроупругости...")
    results_rigid = parallel.calculate_range(
        velocities, altitudes, attack_angle, use_elastic=False
    )

    # Расчет с аэроупругостью
    print(f"4. Расчет с учетом аэроупругости (3 формы колебаний)...")
    results_elastic = parallel.calculate_range(
        velocities, altitudes, attack_angle, use_elastic=True
    )

    # Визуализация (5 графиков - добавлен график положения фокуса)
    print(f"\n5. Построение графиков...")
    visualizer = SimpleVisualizer()
    visualizer.plot_results(results_rigid, results_elastic, altitudes)

    # Статистика
    visualizer.print_summary(results_rigid, results_elastic)

    # Сохранение результатов
    print(f"\n6. Сохранение результатов...")
    save_results(results_rigid, results_elastic)

    print(f"\n{'='*60}")
    print("РАСЧЕТ ЗАВЕРШЕН")
    print("=" * 60)


def save_results(results_rigid: List[AeroResult], results_elastic: List[AeroResult]):
    """Сохранение результатов в CSV"""

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

        # Добавляем амплитуды мод
        if r_elastic.mode_amplitudes:
            for i, (name, amp) in enumerate(
                zip(r_elastic.mode_names, r_elastic.mode_amplitudes)
            ):
                row[f"mode{i+1}_{name}_amplitude"] = amp

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("aerodynamics_results_with_focus.csv", index=False)
    print(
        f"   Результаты сохранены в aerodynamics_results_with_focus.csv ({len(df)} записей)"
    )


if __name__ == "__main__":
    # Для корректной работы multiprocessing в Windows
    multiprocessing.freeze_support()
    main()
