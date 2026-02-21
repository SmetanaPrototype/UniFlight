import enum
import numpy as np
import math
import os
import csv
import pandas as pd

acceleration_of_gravity = 9.81
earth_radius = 6371000
lamb = (4.73, 7.853, 10.996, 14.137, 17.279)


class density(enum.Enum):
    LOX = 1100.0  # Жидкий кислород
    RP_1 = 810.0  # Очищенный керосин
    UDMH = 790.0  # Несимметричный диметилгидразин
    N2O4 = 1450.0  # Азотный тетраоксид
    HTPB = 1800.0  # Твердое топливо (гидроксил-терминированный полибутадиен)
    AP = 1950.0  # Перхлорат аммония (окислитель в твердом топливе)
    RG_1 = 440.0  # Российский ракетный керосин
    UH25 = 880.0  # Смесь UDMH и гидразина
    CH4 = 450.0  # Сжиженный метан

mode_num = 3

timestep = 1
lenstep  = 1

young_modulus = 71000000000

def get_coefficient_simple(x, x_array, y_array):
    if len(x_array) == 0 or len(y_array) == 0:
        return 0
    return np.interp(x, x_array, y_array)

def calculate_stiffness(diameter):
    return young_modulus * math.pi * cross_sectional_area(diameter)

def cross_sectional_area(diameter):
    return math.pi * (diameter**2) / 4

def calculate_static(mass_, shoulder):
    return 0.5 * mass_ * shoulder

def calculate_inertia(mass_, shoulder, shoulder_diff, diameter):
    return 0.25 * mass_ * (math.pow(shoulder, 2) + 0.333 * math.pow(shoulder_diff, 2) + math.pow((diameter/2), 2))

def sqr(x):
    return x * x

def rad(x):
    return x / 57.3

def absmax(iterable):
    return max(iterable, key=abs)

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