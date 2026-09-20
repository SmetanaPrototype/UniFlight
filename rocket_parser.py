import pandas as pd
import numpy as np
import basis
import json
import matplotlib.pyplot as plt
import rocket_parser_utils
from collections import defaultdict
import tkinter as tk
from tkinter import ttk
import sv_ttk


# ---------------------------------------------------------------------------
# Классы-контейнеры
# ---------------------------------------------------------------------------

class Coord_element:
    def __init__(self, start, length):
        self.start  = float(start)
        self.length = float(length)
        self.end    = self.start + self.length


class Length_dataset:
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

class Time_dataset:
    def __init__(self):
        self.thrusts      = []
        self.masses       = []
        self.distromasses = []
        self.times        = []
        self.statics      = []
        self.inertionsX   = []
        self.inertionsY   = []
        self.centers      = []

class Block_data:
    def __init__(self):
        self.time_data   = Time_dataset()
        self.length_data = Length_dataset()

class United_data:
    def __init__(self, block_num):
        sectors_num = len(basis.get_stages_list(block_num))
        self.blocks = [Block_data() for _ in range(sectors_num)]

    def set_length_dataset(self, length_datasets):
        if len(length_datasets) != len(self.blocks):
            raise ValueError(
                f"Число length-датасетов ({len(length_datasets)}) "
                f"не совпадает с числом блоков ({len(self.blocks)})"
            )
        for block, length_ds in zip(self.blocks, length_datasets):
            block.length_data = length_ds

    def set_time_dataset(self, time_datasets):
        if len(time_datasets) != len(self.blocks):
            raise ValueError(
                f"Число time-датасетов ({len(time_datasets)}) "
                f"не совпадает с числом блоков ({len(self.blocks)})"
            )
        for block, time_ds in zip(self.blocks, time_datasets):
            block.time_data = time_ds

# ---------------------------------------------------------------------------
# Парсер
# ---------------------------------------------------------------------------

class Rocket_parser:
    def __init__(self, rocket=basis.current_rocket):
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
        self.separation_time   = r_data["separation_time"]

        self.block_number = len(self.block_mass)

        is_equal = len(set(map(len,
            [self.oxidizer_type,
             self.fuel_type,
             self.structural_values,
             self.thrust,
             self.exhaust_velocity,
             self.block_mass,
             self.prop_reserve,
             self.separation_time]))) == 1

        if not is_equal:
            raise ValueError("Не хватает данных для всех ступеней!")

        self.csv_filename = "rockets/" + rocket + "/distributed.csv"

        self.is_packet   = self.boosters_number > 1
        self.is_reusable = [p > 0.05 for p in self.prop_reserve]
        self.full_mass   = (self.payload_mass
                            + sum(self.block_mass[1:])
                            + self.boosters_number * self.block_mass[0])

        self.prop_reserve.append(0)

        if self.is_packet and self.separation_time[0] > 0:
            raise ValueError("Центральный блок работает непрерывно!")


        print(f"Payload mass: {self.payload_mass}")
        print(f"Block mass:   {self.block_mass}")
        print(f"Full mass:    {self.full_mass}")
        print(f"Reusable:     {self.is_reusable}")

        self._calculate_stage_parameters()

        self.united_data = United_data(self.block_number)
        self.united_data.set_length_dataset(self._distributed_handler())
        self.united_data.set_time_dataset(self._flight_handler())

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
                rocket_parser_utils.read_propellant_density(self.fuel_type[k]))
            self.oxidizer_density.append(
                rocket_parser_utils.read_propellant_density(self.oxidizer_type[k]))
            mixture_ratio.append(
                rocket_parser_utils.read_mixture_ratio(self.fuel_type[k],
                                                       self.oxidizer_type[k]))

            # ИЗМЕНЕНО: полное топливо, не уменьшаем на reserve
            full_propellant = (
                self.block_mass[k] * self.structural_values[k]
                / (self.structural_values[k] + 1)
            )

            burnable = full_propellant * (1.0 - self.prop_reserve[k])
            reserve  = full_propellant * self.prop_reserve[k]

            self.propellant_mass.append(full_propellant)
            self.burnable_mass.append(burnable)
            self.reserve_mass.append(reserve)

            # ИЗМЕНЕНО: mass_ox/mass_fu — полное топливо
            self.mass_ox.append(
                full_propellant * mixture_ratio[k] / (mixture_ratio[k] + 1))
            self.mass_fu.append(
                full_propellant / (mixture_ratio[k] + 1))

            self.delta_mass.append(self.thrust[k] / self.exhaust_velocity[k])

            # ИЗМЕНЕНО: время горения — только сгораемое топливо
            self.work_time.append(burnable / self.delta_mass[k])

            # ИЗМЕНЕНО: structural_mass — сухая конструкция
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
        start_ = Length_dataset()

        df = pd.read_csv(self.csv_filename)
        start_.lengths    = pd.to_numeric(df["L"], errors='coerce').tolist()
        start_.diameters  = pd.to_numeric(df["D"], errors='coerce').tolist()
        start_.classes    = df["Class"].tolist()
        start_.stages     = df["Stage"].tolist()
        start_.cumlengths = np.cumsum(start_.lengths).tolist()
        start_.numbers    = list(range(len(start_.lengths)))

        if len(start_.cumlengths) != len(set(start_.cumlengths)):
            raise ValueError("Найдены одинаковые кумулятивные длины")

        self.stages  = basis.get_stages_list(self.block_number)
        self.classes = basis.get_classes_list()

        self.stages_lengths = []
        class_stage_masses  = defaultdict(dict)
        for s in self.stages:
            self.stages_lengths.append(
                rocket_parser_utils.get_stage_length(s, start_))
            for c in self.classes:
                class_stage_masses[c][s] = rocket_parser_utils.get_class_stage_mass(
                    self, c, s)

        shared_mass = sum(sum(inner.values()) for inner in class_stage_masses.values())
        booster_mass = sum(class_stage_masses[c].get("First", 0) for c in self.classes)
        shared_mass += booster_mass * (self.boosters_number - 1)

        if abs(shared_mass - self.full_mass) > 0.01:
            raise ValueError(f"Неправильное распределение масс по ступеням: "
                             f"{shared_mass} vs {self.full_mass}")

        dis_partial_ = [Length_dataset() for _ in range(len(self.stages))]

        for b in range(len(dis_partial_)):
            li  = 0.0
            num = 0

            while li < self.stages_lengths[b] - basis.accuracy:
                step_len = min(basis.lenstep, self.stages_lengths[b] - li)

                cur_class    = rocket_parser_utils.get_local_class(
                    self, self.stages[b], li, start_)
                local_length = rocket_parser_utils.get_stageclass_length(
                    self.stages[b], cur_class, start_)

                if local_length > basis.accuracy:
                    mass_value = (rocket_parser_utils.get_class_stage_mass(
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

                diameter = rocket_parser_utils.get_stage_diameter(self.stages[b], start_)
                dis_partial_[b].diameters.append(diameter)
                dis_partial_[b].stiffnesses.append(basis.calculate_stiffness(diameter))
                area = basis.cross_sectional_area(diameter)
                dis_partial_[b].areas.append(area)
                dis_partial_[b].volumes.append(area * step_len)

                li  += step_len
                num += 1

        targets = [sum(class_stage_masses[c][s] for c in self.classes)
                   for s in self.stages]
        for b, ds in enumerate(dis_partial_):
            basis.normalize_list(ds.masses, targets[b])

        return dis_partial_

    # ------------------------------------------------------------------ #
    # Временные параметры + распределение массы (модель уровня топлива)
    # ------------------------------------------------------------------ #
    def _flight_handler(self):
        n_stages      = len(self.stages)
        time_partial_ = [Time_dataset() for _ in range(n_stages)]

        prop_reserve_full = list(self.prop_reserve) + \
                            [0.0] * (n_stages - len(self.prop_reserve))

        init_masses = [list(self.united_data.blocks[b].length_data.masses)
                       for b in range(n_stages)]

        # Геометрия для моментов
        x_coords  = []
        r_half_sq = []
        for b in range(n_stages):
            ld = self.united_data.blocks[b].length_data
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

                is_burning = False

                # --- Выгорание ---
                if index != n_stages - 1 and self.work_time[index] > 0.0:
                    active_dt = rocket_parser_utils.stage_active_dt(
                        self, index, time, basis.timestep)
                    if active_dt > 0.0:
                        remaining = self.work_time[index] - burn_time[index]
                        if remaining > 0.0:
                            effective_dt = min(active_dt, remaining)
                            tmp_mass[index] -= self.delta_mass[index] * effective_dt
                            burn_time[index] += effective_dt
                            is_burning = True

                # --- Тяга: ненулевая только пока ступень горит ---
                thrust_val = self.thrust[index] if is_burning else 0.0
                tp.thrusts.append(thrust_val)

                tp.masses.append(tmp_mass[index])

                # --- Модель уровня топлива с учётом резерва ---
                if index != n_stages - 1 and self.work_time[index] > 0.0:
                    burn_ratio = min(1.0, burn_time[index] / self.work_time[index])
                else:
                    burn_ratio = 0.0

                ld     = self.united_data.blocks[index].length_data
                distro = rocket_parser_utils.fuel_distribution_by_level(
                    self, ld, init_masses[index], burn_ratio,
                    reserve_fraction=prop_reserve_full[index],
                )
                basis.normalize_list(distro, tmp_mass[index])
                tp.distromasses.append(distro)

                # --- Моменты ---
                m = np.asarray(distro, dtype=float)
                M = float(m.sum())
                if M > basis.accuracy:
                    x = x_coords[index]
                    R2 = r_half_sq[index]                       # Rᵢ²

                    # Центр масс ступени
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

            time += basis.timestep

        # ------------------------------------------------------------------
        # Диагностика
        # ------------------------------------------------------------------
        for index in range(n_stages):
            tp = time_partial_[index]
            for t_idx in range(len(tp.times)):
                s = sum(tp.distromasses[t_idx])
                m = tp.masses[t_idx]
                if abs(s - m) > basis.accuracy * max(1.0, abs(m)):
                    print(f"[WARN] Ступень {index}, t={tp.times[t_idx]:.2f}: "
                          f"sum(distro)={s:.6f} vs mass={m:.6f}")
                    break

        # ИЗМЕНЕНО: конечная масса = сухая конструкция + reserve
        for k in range(n_stages - 1):
            expected = self.structural_mass[k] + self.reserve_mass[k]
            if abs(tmp_mass[k] - expected) > basis.accuracy:
                print(f"[WARN] Ступень {k}: {tmp_mass[k]} vs {expected}")

        return time_partial_


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_united_parameters(rp):
    fig1, ax1 = plt.subplots(3, 2)

    n_blocks = len(rp.united_data.blocks)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[0,0].plot(rp.united_data.blocks[i].time_data.times,
                      rp.united_data.blocks[i].time_data.masses,
                      label=label)
    ax1[0,0].set_xlabel("Время, с")
    ax1[0,0].set_ylabel("Масса, кг")
    ax1[0,0].legend()
    ax1[0,0].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[1,0].plot(rp.united_data.blocks[i].time_data.times,
                      rp.united_data.blocks[i].time_data.thrusts,
                      label=label)
    ax1[1,0].set_xlabel("Время, с")
    ax1[1,0].set_ylabel("Тяга, Н")
    ax1[1,0].legend()
    ax1[1,0].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[2,0].plot(rp.united_data.blocks[i].time_data.times,
                      rp.united_data.blocks[i].time_data.statics,
                      label=label)
    ax1[2,0].set_xlabel("Время, с")
    ax1[2,0].set_ylabel("Статический момент, кг·м")
    ax1[2,0].legend()
    ax1[2,0].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[0,1].plot(rp.united_data.blocks[i].time_data.times,
                      rp.united_data.blocks[i].time_data.inertionsX,
                      label=label)
    ax1[0,1].set_xlabel("Время, с")
    ax1[0,1].set_ylabel("Продольный момент, кг·м2")
    ax1[0,1].legend()
    ax1[0,1].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[1,1].plot(rp.united_data.blocks[i].time_data.times,
                      rp.united_data.blocks[i].time_data.inertionsY,
                      label=label)
    ax1[1,1].set_xlabel("Время, с")
    ax1[1,1].set_ylabel("Поперечный момент, кг·м2")
    ax1[1,1].legend()
    ax1[1,1].grid(True)

    for i in range(n_blocks):
        label = "ГО" if i == n_blocks - 1 else f"Ступень {i + 1}"
        ax1[2,1].plot(rp.united_data.blocks[i].time_data.times,
                      rp.united_data.blocks[i].time_data.centers,
                      label=label)
    ax1[2,1].set_xlabel("Время, с")
    ax1[2,1].set_ylabel("Центр масс, м")
    ax1[2,1].legend()
    ax1[2,1].grid(True)

    fig1.tight_layout()
    plt.show()

    

def plot_distromasses_parameters(rp):
    n_blocks = len(rp.united_data.blocks)
    fig2, axes = plt.subplots(n_blocks, 1, squeeze=False)
    axes = axes.ravel()

    start_color = [0.55, 0.85, 1.00]
    end_color   = [0.05, 0.05, 0.30]

    for b in range(n_blocks):
        block = rp.united_data.blocks[b]
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
        mask    = (times_arr >= t_start - basis.accuracy) & (times_arr <= t_end + basis.accuracy)
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
                    basis.interpolate_color(start_color, end_color, i, n_snap)
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
    rp = Rocket_parser()

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
                      command=lambda: plot_united_parameters(rp))
    btn2.pack(pady=5, ipadx=10, ipady=5)

    btn3 = ttk.Button(button_frame, text="МЦИХ ракеты",
                      command=lambda: plot_united_parameters(rp))
    btn3.pack(pady=5, ipadx=10, ipady=5)

    root.mainloop()


if __name__ == "__main__":
    main()