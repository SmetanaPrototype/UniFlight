import enum
import numpy as np
import os
import csv
import pandas as pd
import math

lamb = (4.73, 7.853, 10.996, 14.137, 17.279)

current_rocket = 'soyuz21b'

class Density(enum.Enum):
    LOX = 1100.0      # Жидкий кислород
    RP1 = 810.0      # Очищенный керосин
    UDMH = 790.0      # Несимметричный диметилгидразин
    N2O4 = 1450.0     # Азотный тетраоксид
    HTPB = 1800.0     # Твердое топливо (гидроксил-терминированный полибутадиен)
    AP = 1950.0       # Перхлорат аммония (окислитель в твердом топливе)
    RG1 = 440.0      # Российский ракетный керосин
    UH25 = 880.0      # Смесь UDMH и гидразина
    CH4 = 422.0       # Сжиженный метан (жидкий природный газ)
    LH2 = 70.8        # Жидкий водород (Liquid Hydrogen)
    Naphthyl = 840.0  # Нафтил (модифицированный керосин для Союз-5, плотность между RP-1 и RG-1)

class FuelRatio(enum.Enum):
    LOX_RP1 = 2.56
    LOX_RG1 = 2.6
    LOX_Naphthyl = 2.5
    LOX_LH2 = 5.0
    N2O4_UDMH = 2.6
    N2O4_UH25 = 2.5
    LOX_CH4 = 3.5
    AP_HTPB = 6.0

engine_control_coefficients = {
    1: 1.0,
    2: 1.0,
    3: 0.33,
    4: 0.5,
    5: 0.4,
    6: 0.6,
}

mode_num = 3

timestep = 2
lenstep  = 0.01

tail_coefficient = 1/3

young_modulus = 71000000000
earth_radius  = 6371000

accuracy = 1e-9

@staticmethod
def get_y(x, x_array, y_array):
    if len(x_array) == 0 or len(y_array) == 0:
        return 0
    return np.interp(x, x_array, y_array)

@staticmethod
def calculate_stiffness(diameter):
    return young_modulus * np.pi * cross_sectional_area(diameter)

@staticmethod
def cross_sectional_area(diameter):
    return np.pi * (diameter**2) / 4

@staticmethod
def calculate_static(mass_, shoulder):
    return 0.5 * mass_ * shoulder

@staticmethod
def calculate_inertia(mass_, shoulder, length, diameter):
    return mass_ * (0.25 * shoulder**2 + 0.333 * length**2 + (diameter/2)**2)

@staticmethod
def write_arrays_to_csv(filename, **arrays):
   """Запись массивов в CSV файл"""
   if not arrays:
       raise ValueError("Array is required.")

   os.makedirs(os.path.dirname(filename), exist_ok=True)

   headers = list(arrays.keys())
   max_length = min(len(arr) for arr in arrays.values())

   with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
       writer = csv.writer(csvfile)
       writer.writerow(headers)
       for i in range(max_length):
           row = [arrays[name][i] for name in headers]
           writer.writerow(row)
   print(f"Data was moved to '{filename}'.")

@staticmethod
def read_array_from_csv(filename, arrayname):
    try:
        df = pd.read_csv(filename)
        if arrayname in df.columns:
            column_c = df[arrayname].tolist()
            return column_c
        else:
            print(f"Столбец '{arrayname}' не найден в файле {filename}")
            return None
    except FileNotFoundError:
        print(f"Файл {filename} не найден")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None

@staticmethod
def interpolate_color(start_color, end_color, i, total):
   return [
       start_color[j] + (end_color[j] - start_color[j]) * i / (total - 1)
       for j in range(mode_num)
   ]
@staticmethod
def aerostat_file(filename):
    df = pd.read_csv(filename)
    return df.T.values.tolist()

@staticmethod
def calculate_multi(*args):
    return [math.prod(items) for items in zip(*args)]

@staticmethod
def calculate_sum(*args):
    if not args:
        return []

    vectors = [list(v) for v in args]
    base_idx = max(range(len(vectors)), key=lambda i: len(vectors[i]))
    result = vectors[base_idx].copy()

    for i, vec in enumerate(vectors):
        if i == base_idx:
            continue
        for j, value in enumerate(vec):
            result[j] += value

    return result

@staticmethod
def get_classes_list():
    return ["Tail", "Fuel", "Oxidizer", "Construction", "Head"]

@staticmethod
def get_stages_list(block_num):
    if block_num not in [2,3,4,5]:
        raise ValueError("Current block number is not supported")

    if block_num == 2:
        return ["First", "Second", "Payload"]
    elif block_num == 3:
        return ["First", "Second", "Third", "Payload"]
    elif block_num == 4:
        return ["First", "Second", "Third", "Fourth", "Payload"]
    elif block_num == 5:
        return ["First", "Second", "Third", "Fourth", "Fifth", "Payload"]

@staticmethod
def normalize_list(dataset, target):
    """Приводит сумму dataset.masses к target пропорционально.
    Если текущая сумма 0 — ничего не делает."""
    current = sum(dataset)
    if current <= 0.0:
        return
    scale = target / current
    for i in range(len(dataset)):
        dataset[i] *= scale