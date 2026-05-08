import numpy as np

class alpha:
    def __init__(self, k1: float, k2: float, septime: float, IsRichOrbit: bool):
        self.k1 = k1
        self.k2 = k2
        self.septime = septime
        self.IsRichOrbit = IsRichOrbit

    def calculate_alpha(self, velocity: float, time: float) -> float:
        ans = 0.0
        z = 0.0
        che = 0.0

        if 50 < velocity < 270 and time <= self.septime:
            z = np.pi * (velocity - 50)
            che = (velocity - 50) + 0.25 * (270 - velocity)
            ans = -self.k1 * (np.sin(z / che) ** 2)
        elif time >= self.septime:
            if not self.IsRichOrbit:
                ans = self.k2 * (time - self.septime) ** (1 / 2)
            else:
                if time - self.septime < 60:
                    z = np.pi * (time - self.septime)
                    che = (time - self.septime) + 0.25 * (self.septime + 60 - time)
                    ans = 90 * (np.sin(z / che) ** 2)

        # Применяем ограничения в зависимости от времени
        if time <= self.septime:
            # На активном участке - строгое ограничение до -5°
            return max(-5.0, ans)
        else:
            # На пассивном участке - без ограничений (или мягкое ограничение)
            return min(35.0, ans)  # Можно добавить max(-90, min(90, ans)) если нужно


# Пример использования
if __name__ == "__main__":
    a = alpha(1.0, 0.5, 100.0, True)

    # Тестируем разные случаи
    print(f"До septime (t=95): {a.calculate_alpha(120.0, 95.0):.2f}°")
    print(f"После septime (t=105): {a.calculate_alpha(120.0, 105.0):.2f}°")
    print(
        f"До septime с большим отрицательным значением: {a.calculate_alpha(200.0, 50.0):.2f}°"
    )
