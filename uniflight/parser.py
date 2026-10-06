import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sv_ttk
import tkinter as tk
from collections import defaultdict
from tkinter import ttk

import config as cfg
import physics as phys

import utils.geometry as u_geometry
import utils.math as u_math
import utils.plot as u_plot
# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_class_stage_mass(parser_obj, class_, stage_):
    res = 0
    eachtailmass = parser_obj.payload_mass * cfg.tail_coefficient / (parser_obj.block_number + parser_obj.boosters_number - 1)
    if class_ == "Head" and stage_ == "Payload":
        res = parser_obj.payload_mass
    
    for b in range(parser_obj.block_number):
        if class_ == "Tail" and stage_ == parser_obj.stages[b]:
            res = eachtailmass
        elif class_ == "Oxidizer" and stage_ == parser_obj.stages[b]:
            res = parser_obj.mass_ox[b]
        elif class_ == "Fuel" and stage_ == parser_obj.stages[b]:
            res = parser_obj.mass_fu[b]
        elif class_ == "Construction" and stage_ == parser_obj.stages[b]:
            res = parser_obj.structural_mass[b] - eachtailmass
        
    return res

def stage_active_dt(self, index, time, dt):
    """Сколько секунд ступени index горит внутри интервала [time, time+dt]."""
    if index == 0 or (self.is_packet and index == 1):
        t_start = 0.0
    else:
        t_start = self.step_time[index - 1]
    t_end = self.step_time[index]

    lo = max(time, t_start)
    hi = min(time + dt, t_end)
    return max(0.0, hi - lo)


def stage_is_active_at(self, index, time):
    """
    Мгновенная проверка: горит ли ступень index в момент time.

    Отличается от stage_active_dt тем, что не «размазывает» активность
    на интервал [time, time+dt]. В момент переключения ступеней ровно
    одна из них активна — это устраняет ложные срабатывания проверки
    _validate_thrust_isolation на границах step_time.

    Для последней ступени (Payload) всегда возвращает False,
    так как её нет в self.step_time.
    """
    if index < 0 or index >= len(self.step_time):
        return False

    if index == 0 or (self.is_packet and index == 1):
        t_start = 0.0
    else:
        t_start = self.step_time[index - 1]
    t_end = self.step_time[index]

    return t_start <= time < t_end

# ---------------------------------------------------------------------------
# Классы-контейнеры
# ---------------------------------------------------------------------------

class Coord_element:
    def __init__(self, start, length):
        self.start  = float(start)
        self.length = float(length)
        self.end    = self.start + self.length


class LengthData:
    def __init__(self):
        self.numbers     = []
        self.lengths     = []
        self.cumlengths  = []
        self.diameters   = []
        self.classes     = []
        self.stages      = []
        self.stiffnesses = []
        self.areas       = []
        self.volumes     = []
        self.masses      = []
        self.coords      = []


class TimeData:
    def __init__(self):
        self.thrusts      = []
        self.masses       = []
        self.distromasses = []
        self.times        = []
        self.statics      = []
        self.inertionsX   = []
        self.inertionsY   = []
        self.centers      = []


class Block:
    def __init__(self):
        self.time_data   = TimeData()
        self.length_data = LengthData()


class Rocket:
    def __init__(self, block_num):
        sectors_num = len(cfg.get_stages_list(block_num))
        self.blocks = [Block() for _ in range(sectors_num)]
        self.total = Block()

    def set_LengthData(self, LengthDatas):
        if len(LengthDatas) != len(self.blocks):
            raise ValueError(
                f"Число length-датасетов ({len(LengthDatas)}) "
                f"не совпадает с числом блоков ({len(self.blocks)})"
            )
        for block, length_ds in zip(self.blocks, LengthDatas):
            block.length_data = length_ds

    def set_TimeData(self, TimeDatas):
        if len(TimeDatas) != len(self.blocks):
            raise ValueError(
                f"Число time-датасетов ({len(TimeDatas)}) "
                f"не совпадает с числом блоков ({len(self.blocks)})"
            )
        for block, time_ds in zip(self.blocks, TimeDatas):
            block.time_data = time_ds

    def build_rocket(self, boosters_number):
        is_packet = boosters_number > 0
        # length data
        rocket_length = (
            sum(m.length_data.cumlengths[-1] for m in self.blocks)
            if not is_packet
            else sum(m.length_data.cumlengths[-1] for m in self.blocks[1:])
        )

        self.total.length_data.lengths = [
            i * cfg.lenstep for i in range(int(rocket_length / cfg.lenstep))
        ]

        united_blocks = (
            self.blocks if not is_packet
            else self.blocks + (boosters_number - 1) * [self.blocks[0]]
        )
        # time data
        self.total.time_data.times = list(self.blocks[0].time_data.times)
        self.total.time_data.masses = u_math.calculate_sum(
            *(m.time_data.masses for m in united_blocks))
        self.total.time_data.thrusts = u_math.calculate_sum(
            *(m.time_data.thrusts for m in united_blocks))

        # # --- статический момент всей ракеты ---
        n_times = len(self.total.time_data.times)
        n_blocks = len(self.blocks)

# ---------------------------------------------------------------------------
# Парсер
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, rocket):
        json_filename = "rockets/" + rocket + "/constant.json"

        with open(json_filename, "r") as r_file:
            r_data = json.load(r_file)

        # common data
        self.name              = r_data["name"]
        self.payload_mass      = r_data["payload_mass"]
        self.block_mass        = r_data["block_mass"]
        self.exhaust_velocity  = r_data["exhaust_velocity"]
        self.thrust            = r_data["thrust"]
        self.structural_values = r_data["structural_values"]
        self.fuel_type         = r_data["fuel_type"]
        self.oxidizer_type     = r_data["oxidizer_type"]
        self.boosters_number   = r_data["booster_number"]
        self.attack_coefs      = r_data["attack_coefs"]
        self.chumbers_number   = r_data["chumbers_number"]
        self.prop_reserve      = r_data["prop_reserve"]
        self.coast_phase       = r_data["coast_phase"]

        self.block_number = len(self.block_mass)

        is_equal = len(set(map(len,
            [self.oxidizer_type,
             self.fuel_type,
             self.structural_values,
             self.thrust,
             self.exhaust_velocity,
             self.block_mass,
             self.prop_reserve,
             self.coast_phase]))) == 1

        if not is_equal:
            raise ValueError("Не хватает данных для всех ступеней!")

        self.csv_filename = "rockets/" + rocket + "/distributed.csv"

        self.is_packet   = self.boosters_number > 1
        self.is_reusable = [p > 0.05 for p in self.prop_reserve]
        self.full_mass   = (self.payload_mass
                            + sum(self.block_mass[1:])
                            + self.boosters_number * self.block_mass[0])

        self.prop_reserve.append(0)

        if self.is_packet and self.coast_phase[0] > 0:
            raise ValueError("Центральный блок работает непрерывно!")

        print(f"Payload mass: {self.payload_mass}")
        print(f"Block mass:   {self.block_mass}")
        print(f"Full mass:    {self.full_mass}")
        print(f"Reusable:     {self.is_reusable}")

        self._calculate_stage_parameters()

        self.rocket = Rocket(self.block_number)
        self.rocket.set_LengthData(self._distributed_handler())
        self.rocket.set_TimeData(self._flight_handler())

        self.rocket.build_rocket(self.boosters_number)

    # ------------------------------------------------------------------ #
    # Параметры ступеней
    # ------------------------------------------------------------------ #
    def _calculate_stage_parameters(self):
        self.mass_ox           = []
        self.mass_fu           = []
        self.delta_mass        = []
        self.work_time         = []
        self.structural_mass   = []
        self.delta_mass_ox     = []
        self.delta_mass_fu     = []
        self.fuel_density      = []
        self.oxidizer_density  = []
        self.propellant_mass   = []
        self.burnable_mass     = []
        self.reserve_mass      = []
        mixture_ratio          = []

        for k in range(self.block_number):
            self.fuel_density.append(
                phys.read_propellant_density(self.fuel_type[k]))
            self.oxidizer_density.append(
                phys.read_propellant_density(self.oxidizer_type[k]))
            mixture_ratio.append(
                phys.read_mixture_ratio(self.fuel_type[k],
                                        self.oxidizer_type[k]))

            full_propellant = (
                self.block_mass[k] * self.structural_values[k]
                / (self.structural_values[k] + 1)
            )

            burnable = full_propellant * (1.0 - self.prop_reserve[k])
            reserve  = full_propellant * self.prop_reserve[k]

            self.propellant_mass.append(full_propellant)
            self.burnable_mass.append(burnable)
            self.reserve_mass.append(reserve)

            self.mass_ox.append(
                full_propellant * mixture_ratio[k] / (mixture_ratio[k] + 1))
            self.mass_fu.append(
                full_propellant / (mixture_ratio[k] + 1))

            self.delta_mass.append(self.thrust[k] / self.exhaust_velocity[k])

            self.work_time.append(burnable / self.delta_mass[k])

            self.structural_mass.append(self.block_mass[k] - full_propellant)

            self.delta_mass_ox.append(
                self.delta_mass[k] * mixture_ratio[k] / (mixture_ratio[k] + 1))
            self.delta_mass_fu.append(
                self.delta_mass[k] / (mixture_ratio[k] + 1))

        n = self.block_number
        self.step_time = [0.0] * n
        if self.is_packet:
            offset = self.work_time[1] if n > 1 else self.work_time[0]
            self.step_time[0] = self.work_time[0]
            if n > 1:
                self.step_time[1] = self.work_time[1]
            acc = offset
            for k in range(2, n):
                acc += self.work_time[k]
                self.step_time[k] = acc
        else:
            acc = 0.0
            for k in range(n):
                acc += self.work_time[k]
                self.step_time[k] = acc
        self.full_time = self.step_time[-1]

        print("Propellant:   ", self.propellant_mass)
        print("  burnable:   ", self.burnable_mass)
        print("  reserve:    ", self.reserve_mass)
        print("Structural:   ", self.structural_mass)
        print("Work_time:    ", self.work_time)
        print("Stage_time:   ", self.step_time)

    # ------------------------------------------------------------------ #
    # Распределённые по длине параметры
    # ------------------------------------------------------------------ #
    def _distributed_handler(self):
        start_ = LengthData()

        df = pd.read_csv(self.csv_filename)
        start_.lengths    = pd.to_numeric(df["L"], errors='coerce').tolist()
        start_.diameters  = pd.to_numeric(df["D"], errors='coerce').tolist()
        start_.classes    = df["Class"].tolist()
        start_.stages     = df["Stage"].tolist()
        start_.cumlengths = np.cumsum(start_.lengths).tolist()
        start_.numbers    = list(range(len(start_.lengths)))

        if len(start_.cumlengths) != len(set(start_.cumlengths)):
            raise ValueError("Найдены одинаковые кумулятивные длины")

        self.stages  = cfg.get_stages_list(self.block_number)
        self.classes = cfg.get_classes_list()

        self.stages_lengths = []
        class_stage_masses  = defaultdict(dict)
        for s in self.stages:
            self.stages_lengths.append(
                u_geometry.get_stage_length(s, start_))
            for c in self.classes:
                class_stage_masses[c][s] = get_class_stage_mass(
                    self, c, s)

        shared_mass = sum(sum(inner.values()) for inner in class_stage_masses.values())
        booster_mass = sum(class_stage_masses[c].get("First", 0) for c in self.classes)
        shared_mass += booster_mass * (self.boosters_number - 1)

        if abs(shared_mass - self.full_mass) > 0.01:
            raise ValueError(f"Неправильное распределение масс по ступеням: "
                             f"{shared_mass} vs {self.full_mass}")

        dis_partial_ = [LengthData() for _ in range(len(self.stages))]

        for b in range(len(dis_partial_)):
            li  = 0.0
            num = 0

            while li < self.stages_lengths[b] - cfg.accuracy:
                step_len = min(cfg.lenstep, self.stages_lengths[b] - li)

                cur_class    = u_geometry.get_local_class(
                    self, self.stages[b], li, start_)
                local_length = u_geometry.get_stageclass_length(
                    self.stages[b], cur_class, start_)

                if local_length > cfg.accuracy:
                    mass_value = (get_class_stage_mass(
                        self, cur_class, self.stages[b])
                        * step_len / local_length)
                else:
                    mass_value = 0.0

                dis_partial_[b].numbers.append(num)
                dis_partial_[b].lengths.append(step_len)
                dis_partial_[b].cumlengths.append(li)
                dis_partial_[b].stages.append(self.stages[b])
                dis_partial_[b].classes.append(cur_class)
                dis_partial_[b].masses.append(mass_value)
                dis_partial_[b].coords.append(Coord_element(li, step_len))

                diameter = u_geometry.get_stage_diameter(self.stages[b], start_)
                dis_partial_[b].diameters.append(diameter)
                dis_partial_[b].stiffnesses.append(phys.calculate_stiffness(diameter))
                area = phys.cross_sectional_area(diameter)
                dis_partial_[b].areas.append(area)
                dis_partial_[b].volumes.append(area * step_len)

                li  += step_len
                num += 1

        targets = [sum(class_stage_masses[c][s] for c in self.classes)
                   for s in self.stages]
        for b, ds in enumerate(dis_partial_):
            u_math.normalize_list(ds.masses, targets[b])

        return dis_partial_

    # ------------------------------------------------------------------ #
    # Проверка изоляции тяг
    # ------------------------------------------------------------------ #
    def _validate_thrust_isolation(self, time_partial_):
        """
        Проверяет, что на каждом шаге по времени работает двигатель
        не более чем одной ступени.

        Исключение — пакетная схема: боковые блоки (индекс 0) и
        центральный блок (индекс 1) горят параллельно по построению,
        это разрешено.

        Возбуждает RuntimeError при обнаружении недопустимого пересечения.
        """
        n_stages = len(time_partial_)
        times    = time_partial_[0].times
        tol      = cfg.accuracy

        parallel_ok = {0, 1} if self.is_packet else set()

        for i, t in enumerate(times):
            burning = [
                s for s in range(n_stages)
                if time_partial_[s].thrusts[i] > tol
            ]

            if len(burning) <= 1:
                continue

            if self.is_packet and set(burning).issubset(parallel_ok):
                continue

            thrusts = [time_partial_[s].thrusts[i] for s in burning]
            raise RuntimeError(
                f"Пересечение работы ступеней в t = {t:.3f} с: "
                f"одновременно горят ступени {burning} "
                f"(тяги = {thrusts} Н). "
                f"Схема: {'пакетная' if self.is_packet else 'тандемная'}. "
                f"Проверьте step_time={self.step_time}, "
                f"work_time={self.work_time}."
            )

    # ------------------------------------------------------------------ #
    # Временные параметры + распределение массы (модель уровня топлива)
    # ------------------------------------------------------------------ #
    def _flight_handler(self):
        n_stages      = len(self.stages)
        time_partial_ = [TimeData() for _ in range(n_stages)]

        prop_reserve_full = list(self.prop_reserve) + \
                            [0.0] * (n_stages - len(self.prop_reserve))

        init_masses = [list(self.rocket.blocks[b].length_data.masses)
                       for b in range(n_stages)]

        # Геометрия для моментов
        x_coords  = []
        r_half_sq = []
        for b in range(n_stages):
            ld = self.rocket.blocks[b].length_data
            x_coords.append(np.array([
                ld.cumlengths[i] + ld.lengths[i] / 2.0
                for i in range(len(ld.lengths))
            ]))
            r_half_sq.append((np.array(ld.diameters) / 2.0) ** 2)

        tmp_mass  = list(self.block_mass) + [self.payload_mass]
        burn_time = [0.0] * n_stages

        time = 0.0
        while time < self.full_time:
            for index, tp in enumerate(time_partial_):
                tp.times.append(time)

                is_payload = (index == n_stages - 1)

                # ------------------------------------------------------
                # 1. Мгновенная активность (для записи тяги)
                # ------------------------------------------------------
                is_active = False
                if not is_payload:
                    is_active = stage_is_active_at(
                        self, index, time)

                # ------------------------------------------------------
                # 2. Интеграл горения на [time, time+dt] (для массы)
                # ------------------------------------------------------
                if is_active and self.work_time[index] > 0.0:
                    active_dt = stage_active_dt(
                        self, index, time, cfg.timestep)
                    remaining = self.work_time[index] - burn_time[index]
                    active_dt = max(0.0, min(active_dt, remaining))
                    if active_dt > 0.0:
                        tmp_mass[index] -= self.delta_mass[index] * active_dt
                        burn_time[index] += active_dt
                    else:
                        # Топливо кончилось раньше расписания — ступень гаснет
                        is_active = False

                # ------------------------------------------------------
                # 3. Тяга: мгновенная
                # ------------------------------------------------------
                thrust_val = self.thrust[index] if is_active else 0.0
                tp.thrusts.append(thrust_val)
                tp.masses.append(tmp_mass[index])

                # ------------------------------------------------------
                # 4. Модель уровня топлива с учётом резерва
                # ------------------------------------------------------
                if not is_payload and self.work_time[index] > 0.0:
                    burn_ratio = min(1.0, burn_time[index] / self.work_time[index])
                else:
                    burn_ratio = 0.0

                ld     = self.rocket.blocks[index].length_data
                distro = phys.fuel_distribution_by_level(
                    ld, init_masses[index], burn_ratio,
                    reserve_fraction=prop_reserve_full[index],
                )
                u_math.normalize_list(distro, tmp_mass[index])
                tp.distromasses.append(distro)

                # ------------------------------------------------------
                # 5. Моменты
                # ------------------------------------------------------
                m = np.asarray(distro, dtype=float)
                M = float(m.sum())
                if M > cfg.accuracy:
                    x = x_coords[index]
                    R2 = r_half_sq[index]

                    S = float(np.sum(m * x))
                    c = S / M

                    # Продольный: Σ mᵢ · Rᵢ² / 2
                    I_x = float(np.sum(m * R2 / 2.0))

                    # Поперечный: Σ [ mᵢ · Rᵢ²/4 + mᵢ · (xᵢ − c)² ]
                    I_y = float(np.sum(m * (R2 / 4.0 + (x - c) ** 2)))
                else:
                    S, c, I_x, I_y = 0.0, 0.0, 0.0, 0.0

                tp.statics.append(S)
                tp.centers.append(c)
                tp.inertionsX.append(I_x)
                tp.inertionsY.append(I_y)

            time += cfg.timestep

        # ------------------------------------------------------------------
        # Диагностика
        # ------------------------------------------------------------------
        for index in range(n_stages):
            tp = time_partial_[index]
            for t_idx in range(len(tp.times)):
                s = sum(tp.distromasses[t_idx])
                m = tp.masses[t_idx]
                if abs(s - m) > cfg.accuracy * max(1.0, abs(m)):
                    print(f"[WARN] Ступень {index}, t={tp.times[t_idx]:.2f}: "
                          f"sum(distro)={s:.6f} vs mass={m:.6f}")
                    break

        # Конечная масса = сухая конструкция + reserve
        for k in range(n_stages - 1):
            expected = self.structural_mass[k] + self.reserve_mass[k]
            if abs(tmp_mass[k] - expected) > cfg.accuracy:
                print(f"[WARN] Ступень {k}: {tmp_mass[k]} vs {expected}")

        # Проверка изоляции тяг
        self._validate_thrust_isolation(time_partial_)

        return time_partial_


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_stage_parameters(rp):
    fig1, ax1 = plt.subplots(3, 2)

    n_blocks = len(rp.Rocket.blocks)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[0, 0].plot(rp.Rocket.blocks[i].time_data.times,
                       rp.Rocket.blocks[i].time_data.masses,
                       label=label)
    ax1[0, 0].set_xlabel("Время, с")
    ax1[0, 0].set_ylabel("Масса, кг")
    ax1[0, 0].legend()
    ax1[0, 0].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[1, 0].plot(rp.Rocket.blocks[i].time_data.times,
                       rp.Rocket.blocks[i].time_data.thrusts,
                       label=label)
    ax1[1, 0].set_xlabel("Время, с")
    ax1[1, 0].set_ylabel("Тяга, Н")
    ax1[1, 0].legend()
    ax1[1, 0].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[2, 0].plot(rp.Rocket.blocks[i].time_data.times,
                       rp.Rocket.blocks[i].time_data.statics,
                       label=label)
    ax1[2, 0].set_xlabel("Время, с")
    ax1[2, 0].set_ylabel("Статический момент, кг·м")
    ax1[2, 0].legend()
    ax1[2, 0].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[0, 1].plot(rp.Rocket.blocks[i].time_data.times,
                       rp.Rocket.blocks[i].time_data.inertionsX,
                       label=label)
    ax1[0, 1].set_xlabel("Время, с")
    ax1[0, 1].set_ylabel("Продольный момент, кг·м2")
    ax1[0, 1].legend()
    ax1[0, 1].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[1, 1].plot(rp.Rocket.blocks[i].time_data.times,
                       rp.Rocket.blocks[i].time_data.inertionsY,
                       label=label)
    ax1[1, 1].set_xlabel("Время, с")
    ax1[1, 1].set_ylabel("Поперечный момент, кг·м2")
    ax1[1, 1].legend()
    ax1[1, 1].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[2, 1].plot(rp.Rocket.blocks[i].time_data.times,
                       rp.Rocket.blocks[i].time_data.centers,
                       label=label)
    ax1[2, 1].set_xlabel("Время, с")
    ax1[2, 1].set_ylabel("Центр масс, м")
    ax1[2, 1].legend()
    ax1[2, 1].grid(True)

    fig1.tight_layout()
    plt.show()


def plot_rocket_parameters(rp):
    fig1, ax1 = plt.subplots(4, 1)

    ax1[0].plot(rp.Rocket.total.time_data.times,
                rp.Rocket.total.time_data.masses)
    ax1[0].set_xlabel("Время, с")
    ax1[0].set_ylabel("Масса, кг")
    ax1[0].grid(True)

    ax1[1].plot(rp.Rocket.total.time_data.times,
                rp.Rocket.total.time_data.thrusts)
    ax1[1].set_xlabel("Время, с")
    ax1[1].set_ylabel("Тяга, Н")
    ax1[1].grid(True)

    # ax1[2].plot(rp.Rocket.total.time_data.times,
    #             rp.Rocket.total.time_data.statics)
    # ax1[2].set_xlabel("Время, с")
    # ax1[2].set_ylabel("Стат момент, кг м")
    # ax1[2].grid(True)

    # ax1[3].plot(rp.Rocket.total.time_data.times,
    #             rp.Rocket.total.time_data.centers)
    # ax1[3].set_xlabel("Время, с")
    # ax1[3].set_ylabel("Центр масс, м")
    # ax1[3].grid(True)

    fig1.tight_layout()
    plt.show()


def plot_distromasses_parameters(rp):
    n_blocks = len(rp.Rocket.blocks)
    fig2, axes = plt.subplots(n_blocks, 1, squeeze=False)
    axes = axes.ravel()

    start_color = [0.55, 0.85, 1.00]
    end_color   = [0.05, 0.05, 0.30]

    for b in range(n_blocks):
        block = rp.Rocket.blocks[b]
        ld    = block.length_data
        td    = block.time_data
        is_payload = (b == n_blocks - 1)

        x = np.array(ld.cumlengths) + np.array(ld.lengths) / 2

        if is_payload:
            t_start, t_end = 0.0, rp.full_time
        else:
            t_end   = rp.step_time[b]
            t_start = 0.0 if (b == 0 or (rp.is_packet and b == 1)) else rp.step_time[b - 1]

        times_arr = np.array(td.times)
        mask    = (times_arr >= t_start - cfg.accuracy) & (times_arr <= t_end + cfg.accuracy)
        n_local = int(mask.sum())

        if n_local == 0:
            axes[b].set_title(f"Блок {b + 1}: нет данных")
            continue

        if is_payload:
            local_idx = np.where(mask)[0]
            snap_idx  = [local_idx[0]]
            n_snap    = 1
        else:
            n_snap    = n_local
            local_idx = np.where(mask)[0]
            snap_idx  = local_idx[np.linspace(0, n_local - 1, n_snap, dtype=int)]

        for i, k in enumerate(snap_idx):
            color = start_color if n_snap == 1 else \
                    u_plot.interpolate_color(start_color, end_color, i, n_snap)
            if is_payload:
                lbl = "ГО"
            elif i == 0 or i == n_snap - 1:
                lbl = f"t = {td.times[k]:.1f} с"
            else:
                lbl = "_nolegend_"
            axes[b].plot(x, td.distromasses[k],
                         color=color, linewidth=1.8, label=lbl)

        axes[b].set_xlabel("Длина блока, м")
        axes[b].set_ylabel("Распределённая масса, кг")
        axes[b].set_yscale("symlog", linthresh=1.0)
        axes[b].set_title(
            "ГО: распределение массы по длине" if is_payload
            else f"Ступень {b + 1}: распределение массы "
                 f"({t_start:.1f} – {t_end:.1f} с)"
        )
        axes[b].legend(loc="upper right", fontsize=9)
        axes[b].grid(True, which="both", alpha=0.3)

    fig2.tight_layout()
    plt.show()


def main():
    # rocket_sequence = []
    # for r in cfg.available_rockets:
    #     rocket_sequence.append(Parser(r))

    rp = Parser('soyuz21b') # pocket

    root = tk.Tk()
    sv_ttk.set_theme("dark")
    root.title("Данные парсера")
    root.geometry("400x300")

    button_frame = ttk.Frame(root)
    button_frame.pack(expand=True)

    btn1 = ttk.Button(button_frame, text="Распределенные массы блоков",
                      command=lambda: plot_distromasses_parameters(rp))
    btn1.pack(pady=5, ipadx=10, ipady=5)

    btn2 = ttk.Button(button_frame, text="МЦИХ блоков",
                      command=lambda: plot_stage_parameters(rp))
    btn2.pack(pady=5, ipadx=10, ipady=5)

    btn3 = ttk.Button(button_frame, text="МЦИХ ракеты",
                      command=lambda: plot_rocket_parameters(rp))
    btn3.pack(pady=5, ipadx=10, ipady=5)

    root.mainloop()


if __name__ == "__main__":
    main()