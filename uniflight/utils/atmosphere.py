import numpy as np
import matplotlib.pyplot as plt


class atmosphere:
    def __init__(self, H):
        self.H = H

        self.HT = [
            1,
            11019,
            32000,
            47350,
            71802,
            86152,
            104128,
            120000,
            140000,
            160000,
            200000,
            325000,
            400000,
            600000,
            1200000,
        ]
        self.TT = [
            288.15,
            216.65,
            228.65,
            270.65,
            214.65,
            186.65,
            203.81,
            334.417,
            559.6,
            695.6,
            834.4,
            941.9,
            984.65,
            995.9,
            1000,
        ]
        self.TMM = [288.15, 216.65, 228.65, 270.65, 214.65, 186.65, 212.0, 380.60]

        self.dynH = [
            0,
            5000,
            10000,
            15000,
            20000,
            25000,
            30000,
            40000,
            50000,
            60000,
            80000,
        ]
        self.dynT = [
            17.89,
            15.07,
            13.06,
            11.45,
            10.17,
            9.18,
            8.38,
            7.18,
            6.25,
            5.52,
            4.52,
        ]

        self.windH = [
            100,
            6000,
            12000,
            18000,
            24000,
            30000,
            36000,
            42000,
            48000,
            54000,
            60000,
            66000,
            72000,
            78000,
            84000,
            90000,
            94000,
        ]
        self.windV = [1.9, 8, 15, 18, 24, 12, 10, 8, 6, 5, 4, 3, 3, 2, 2, 1, 0.1]

        # Константы
        self.Mc = 28.964420
        self.gc = 9.80665
        self.ac = 340.294
        self.Hpc = 8434.5
        self.nc = 25.471 * 10**24
        self.pc = 101325.0
        self.Tc = 288.15
        self.vc = 458.94
        self.yc = 12.013
        self.nuc = 14.607 * 10**-6
        self.muc = 17.894 * 10**-6
        self.lac = 25.343 * 10**-3
        self.omegac = 6.9193 * 10**9
        self.poc = 1.2250
        self.Na = 602.257 * 10**24
        self.RB = 8314.32
        self.r = 287.05287
        self.SOS = 110.4
        self.BS = 1.458 * 10**-6
        self.hi = 1.4
        self.b = 0.365 * 10**-9
        self.R = 6371000

        self.PI = np.pi
        self.D = 2.66
        self.S = self.PI * self.D**2 / 4

        # Инициализация переменных
        self.T = self.Tc
        self.Tm = self.Tc * self.Mc / self.Mc  # будет просто Tc
        self.P = self.pc
        self.n = self.nc
        self.g = self.gc
        self.Mol = self.Mc
        self.po = None
        self.a = None
        self.dyn = None
        self.wind_velocity = 0
        # Вычисления
        self._calculate()

    def _calculate(self):
        H = self.H

        # Температура T
        for i in range(1, 15):
            if H >= self.HT[i - 1] and H < self.HT[i]:
                self.T = self.TT[i - 1] + (H - self.HT[i - 1]) * (
                    self.TT[i] - self.TT[i - 1]
                ) / (self.HT[i] - self.HT[i - 1])
                break

        # Средняя температура Tm
        for i in range(1, 8):
            if H >= self.HT[i - 1] and H < self.HT[i]:
                self.Tm = self.TMM[i - 1] + (H - self.HT[i - 1]) * (
                    self.TMM[i] - self.TMM[i - 1]
                ) / (self.HT[i] - self.HT[i - 1])
                break

        # Скорость ветра W
        for i in range(1, 17):
            if H >= self.windH[i - 1] and H < self.windH[i]:
                self.wind_velocity = self.windV[i - 1] + (H - self.windH[i - 1]) * (
                    self.windV[i] - self.windV[i - 1]
                ) / (self.windH[i] - self.windH[i - 1])
                break

        if H < 94000:
            # Коэффициенты полинома молярной массы
            B = [46.9083, -29.71210e-5, 12.08693e-10, -1.85675e-15]

            self.Mol = self.Mc

            Bett = (
                7466 * H**3
                - 1262795028 * H**2
                + 61597340039789 * H
                - 833732588564247562
            ) * 1e-20

            if abs(Bett) < 1e-7:
                pp = np.log(101325)
                Hs = H - 0.1
                self.P = np.exp(
                    pp - (0.434294 * self.gc / (self.r * self.T)) * (H - 0)
                )
            else:
                pp = np.log(101325)
                Hs = H - 0.1
                self.P = np.exp(
                    pp
                    - (self.gc * np.log((self.Tm + Bett * (H - 0)) / self.Tm))
                    / (Bett * self.r)
                )

            self.Pap = 101325 * np.exp(-self.gc * H * self.Mc / (self.RB * self.T))

            self.po = (self.P * self.Mol) / (self.RB * self.T)

            self.n = 7.243611e22 * self.P / self.T

            self.tCel = self.T - 273.15
            self.yyd = self.po * self.g
            self.Hmas = (self.RB / self.Mol) * (self.T / self.g)
            self.a = 20.046796 * np.sqrt(self.T)
            self.vsred = 145.50685 * self.T / self.Mol
            self.lsred = 2.332376e-5 * self.T / self.P
            self.omega = 6.238629e6 * self.P / np.sqrt(self.T * self.Mol)
            self.lamb = (2.648151e-3 * self.T ** (3 / 2)) / (
                self.T + 245.4 * 10 ** (-(12 / self.T))
            )

            if H <= self.dynH[0]:
                self.dyn = self.dynT[0] * 1e-6
            elif H >= self.dynH[-1]:
                self.dyn = self.dynT[-1] * 1e-6
            else:
                for i in range(1, len(self.dynH)):
                    if H < self.dynH[i]:
                        self.dyn = (
                            self.dynT[i - 1]
                            + (H - self.dynH[i - 1])
                            * (self.dynT[i] - self.dynT[i - 1])
                            / (self.dynH[i] - self.dynH[i - 1])
                        ) * 1e-6
                    break

        self.g = self.gc * (self.R / (self.R + H)) ** 2

    def get_T(self):
        return self.T

    def get_n(self):
        return self.n

    def get_pressure(self):
        return self.P

    def get_density(self):
        return self.po

    def get_AOG(self):
        return self.g

    def get_SV(self):
        return self.a

    def get_dyn(self):
        return self.dyn

    def get_wind(self):
        return self.wind_velocity


if __name__ == "__main__":
    atm = atmosphere(10000)
    print("Temperature (K):", atm.get_T())
    print("Pressure (Pa):", atm.get_pressure())
    print("Density (kg/m³):", atm.get_density())
    print("Gravity (m/s²):", atm.get_AOG())
    print("Speed of sound (m/s):", atm.get_SV())

    w_femap = [11.86, 32.51, 60.66, 86.75, 124.37]
    colors = ["b", "r", "m"]

    altitude_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    density = []
    temperature = []
    gravity = []
    pressure = []
    speed_of_sound = []
    atomic_speed = []

    for altitude in altitude_list:
        atm = atmosphere(altitude * 1000)
        density.append(atm.get_density())
        temperature.append(atm.get_T())
        gravity.append(atm.get_AOG())
        pressure.append(atm.get_pressure())
        speed_of_sound.append(atm.get_SV())
        atomic_speed.append(atm.get_n())

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

    axes[0, 0].plot(altitude_list, density, color="b")
    axes[0, 0].set_title("Плотность воздуха", fontsize=10)
    axes[0, 0].grid()

    axes[0, 1].plot(altitude_list, temperature, color="r")
    axes[0, 1].set_title("Температура", fontsize=10)
    axes[0, 1].grid()

    axes[1, 0].plot(altitude_list, gravity, color="g")
    axes[1, 0].set_title("Ускорение свободного падения", fontsize=10)
    axes[1, 0].grid()

    axes[1, 1].plot(altitude_list, pressure, color="m")
    axes[1, 1].set_title("Давление", fontsize=10)
    axes[1, 1].grid()

    axes[0, 2].plot(altitude_list, speed_of_sound, color="g")
    axes[0, 2].set_title("Скорость звука", fontsize=10)
    axes[0, 2].grid()

    axes[1, 2].plot(altitude_list, atomic_speed, color="c")
    axes[1, 2].set_title("Скорость частиц воздуха", fontsize=10)
    axes[1, 2].grid()

    plt.show()
