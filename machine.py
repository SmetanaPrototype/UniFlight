import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import json

def read_rocket(rocket):
    path = "rocket_lib/" + rocket + "/constant.json"
    with open(path, "r") as r_file:
        r_data = json.load(r_file)
    s = r_data["structural_values"]
    mb = r_data["block_mass"]
    mpn = r_data["payload_mass"]
    thrust = r_data["thrust"]
    coefs  = r_data["attack_coefs"]
    mstep = []
    mstep.append(mpn + sum(mb))
    mstep.append(mpn + sum(mb) - mb[0])
    N = []
    for i in range(len(mb)):
        N.append(thrust[i]/(mstep[i]*9.81))
    return s[0], s[1], N[0], N[1], coefs[0], coefs[1]

rockets_list = ["amur", "falcon", "cz2c", "fireflyalpha", "electron"]

results = [read_rocket(rocket) for rocket in rockets_list]

data = {
    "s1": np.array([r[0] for r in results]),
    "s2": np.array([r[1] for r in results]),
    "N1": np.array([r[2] for r in results]),
    "N2": np.array([r[3] for r in results]),
    "k1": np.array([r[4] for r in results]),
    "k2": np.array([r[5] for r in results]),
}
print(pd.DataFrame(data))
# ============= 1. СОЗДАЕМ СИНТЕТИЧЕСКУЮ ВЫБОРКУ =============
np.random.seed(42)
n_rockets = 200  # количество ракет

# Входные параметры (реалистичные диапазоны)
data = {
    "N1": np.random.uniform(0.8, 2.5, n_rockets),  # тяговооруженность 1 ступени
    "N2": np.random.uniform(0.5, 1.8, n_rockets),  # тяговооруженность 2 ступени
    "s1": np.random.uniform(0.05, 0.25, n_rockets),  # конструктивное совершенство 1
    "s2": np.random.uniform(0.04, 0.20, n_rockets),  # конструктивное совершенство 2
    "Ratio": np.random.choice(
        [1.5, 2.0, 2.5, 3.0, 3.5], n_rockets
    ),  # соотношение компонентов
}


df = pd.DataFrame(data)


# ============= 2. ФИЗИЧЕСКАЯ МОДЕЛЬ (теперь k1 и k2 раздельно) =============
def generate_coefficients(row):
    # k1 (активный участок) зависит от s1, N1 и Ratio
    k1 = (
        0.6 * row["N1"]  # тяговооруженность первой ступени
        - 0.2 * row["s1"]  # конструктивное совершенство
        + 0.15 * np.sin(row["Ratio"] * np.pi / 4)  # влияние топлива
        + np.random.normal(0, 0.05)
    )

    # k2 (пассивный участок) зависит от s2, N2 и Ratio
    k2 = (
        0.4 * row["N2"]  # остаточные возмущения от второй ступени
        + 0.5 * row["s2"]  # аэродинамика пассивного участка
        - 0.1 * (row["Ratio"] - 2.5)  # химия сгоревших газов
        + np.random.normal(0, 0.05)
    )

    # Добавляем нелинейные эффекты
    if row["Ratio"] > 2.8:
        k1 += 0.12  # богатая смесь улучшает управляемость на активном участке
        k2 -= 0.08  # но ухудшает на пассивном

    return k1, k2


# Применяем модель
df[["k1", "k2"]] = df.apply(lambda row: pd.Series(generate_coefficients(row)), axis=1)

# Нормализуем в интервалы
df["k1"] = 2.0 * (df["k1"] - df["k1"].min()) / (df["k1"].max() - df["k1"].min()) - 1.0
df["k2"] = 1.5 * (df["k2"] - df["k2"].min()) / (df["k2"].max() - df["k2"].min()) - 0.5

print("Первые 5 строк датасета:")
print(df.head())
print(f"\nРазмер выборки: {len(df)} ракет")
print(f"Диапазон k1: [{df['k1'].min():.2f}, {df['k1'].max():.2f}]")
print(f"Диапазон k2: [{df['k2'].min():.2f}, {df['k2'].max():.2f}]")

# ============= 3. ОБУЧАЕМ МОДЕЛИ С РАЗНЫМИ ПРИЗНАКАМИ =============
# Для k1 используем только s1, N1, Ratio
X_k1 = df[["s1", "N1", "Ratio"]].values
# Для k2 используем только s2, N2, Ratio
X_k2 = df[["s2", "N2", "Ratio"]].values

y_k1 = df["k1"].values
y_k2 = df["k2"].values

# Разделяем на обучающую и тестовую выборки
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X_k1, y_k1, test_size=0.2, random_state=42
)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_k2, y_k2, test_size=0.2, random_state=42
)

# ============= 4. ОБУЧАЕМ МОДЕЛИ ДЛЯ k1 =============
print("\n" + "=" * 50)
print("МОДЕЛЬ ДЛЯ k1 (АКТИВНЫЙ УЧАСТОК)")
print("=" * 50)

# Random Forest
rf_k1 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_k1.fit(X1_train, y1_train)
y1_pred_rf = rf_k1.predict(X1_test)

# Linear Regression
lr_k1 = LinearRegression()
lr_k1.fit(X1_train, y1_train)
y1_pred_lr = lr_k1.predict(X1_test)

# Оценка
r2_rf_k1 = r2_score(y1_test, y1_pred_rf)
r2_lr_k1 = r2_score(y1_test, y1_pred_lr)
mae_k1 = mean_absolute_error(y1_test, y1_pred_rf)

print(f"Параметры для k1: s1, N1, Ratio")
print(f"Random Forest R² = {r2_rf_k1:.3f}")
print(f"Linear Regression R² = {r2_lr_k1:.3f}")
print(f"MAE (Random Forest) = {mae_k1:.3f}")

# Важность признаков для k1
features_k1 = ["s1", "N1", "Ratio"]
importance_k1 = rf_k1.feature_importances_
print(f"\nВажность признаков (Random Forest):")
for name, imp in zip(features_k1, importance_k1):
    print(f"  {name}: {imp:.3f}")

# ============= 5. ОБУЧАЕМ МОДЕЛИ ДЛЯ k2 =============
print("\n" + "=" * 50)
print("МОДЕЛЬ ДЛЯ k2 (ПАССИВНЫЙ УЧАСТОК)")
print("=" * 50)

# Random Forest
rf_k2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_k2.fit(X2_train, y2_train)
y2_pred_rf = rf_k2.predict(X2_test)

# Linear Regression
lr_k2 = LinearRegression()
lr_k2.fit(X2_train, y2_train)
y2_pred_lr = lr_k2.predict(X2_test)

# Оценка
r2_rf_k2 = r2_score(y2_test, y2_pred_rf)
r2_lr_k2 = r2_score(y2_test, y2_pred_lr)
mae_k2 = mean_absolute_error(y2_test, y2_pred_rf)

print(f"Параметры для k2: s2, N2, Ratio")
print(f"Random Forest R² = {r2_rf_k2:.3f}")
print(f"Linear Regression R² = {r2_lr_k2:.3f}")
print(f"MAE (Random Forest) = {mae_k2:.3f}")

# Важность признаков для k2
features_k2 = ["s2", "N2", "Ratio"]
importance_k2 = rf_k2.feature_importances_
print(f"\nВажность признаков (Random Forest):")
for name, imp in zip(features_k2, importance_k2):
    print(f"  {name}: {imp:.3f}")

# ============= 6. ПРЕДСКАЗАНИЕ ДЛЯ НОВОЙ РАКЕТЫ =============
print("\n" + "=" * 50)
print("ПРИМЕР ПРЕДСКАЗАНИЯ ДЛЯ НОВОЙ РАКЕТЫ")
print("=" * 50)

# Пример новой ракеты
new_rocket_k1 = np.array([[0.15, 1.2, 2.3]])  # s1, N1, Ratio
new_rocket_k2 = np.array([[0.10, 0.9, 2.3]])  # s2, N2, Ratio

pred_k1 = rf_k1.predict(new_rocket_k1)[0]
pred_k2 = rf_k2.predict(new_rocket_k2)[0]

print(f"Параметры ракеты:")
print(f"  s1 (констр.соверш. 1): 0.15")
print(f"  N1 (тяговоор-ть 1 ст): 1.20")
print(f"  s2 (констр.соверш. 2): 0.10")
print(f"  N2 (тяговоор-ть 2 ст): 0.90")
print(f"  Ratio (соотн.компон.): 2.30")
print(f"\nПредсказанные коэффициенты:")
print(f"  k1 (активный участок): {pred_k1:.3f}")
print(f"  k2 (пассивный участок): {pred_k2:.3f}")

# ============= 7. ВИЗУАЛИЗАЦИЯ =============
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# График для k1 (Random Forest)
axes[0, 0].scatter(y1_test, y1_pred_rf, alpha=0.6)
axes[0, 0].plot(
    [y1_test.min(), y1_test.max()], [y1_test.min(), y1_test.max()], "r--", lw=2
)
axes[0, 0].set_xlabel("Истинные k1")
axes[0, 0].set_ylabel("Предсказанные k1")
axes[0, 0].set_title(f"Random Forest: k1 (R² = {r2_rf_k1:.3f})")
axes[0, 0].grid(True, alpha=0.3)

# График для k1 (Linear Regression)
axes[0, 1].scatter(y1_test, y1_pred_lr, alpha=0.6)
axes[0, 1].plot(
    [y1_test.min(), y1_test.max()], [y1_test.min(), y1_test.max()], "r--", lw=2
)
axes[0, 1].set_xlabel("Истинные k1")
axes[0, 1].set_ylabel("Предсказанные k1")
axes[0, 1].set_title(f"Linear Regression: k1 (R² = {r2_lr_k1:.3f})")
axes[0, 1].grid(True, alpha=0.3)

# График для k2 (Random Forest)
axes[1, 0].scatter(y2_test, y2_pred_rf, alpha=0.6)
axes[1, 0].plot(
    [y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], "r--", lw=2
)
axes[1, 0].set_xlabel("Истинные k2")
axes[1, 0].set_ylabel("Предсказанные k2")
axes[1, 0].set_title(f"Random Forest: k2 (R² = {r2_rf_k2:.3f})")
axes[1, 0].grid(True, alpha=0.3)

# График для k2 (Linear Regression)
axes[1, 1].scatter(y2_test, y2_pred_lr, alpha=0.6)
axes[1, 1].plot(
    [y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], "r--", lw=2
)
axes[1, 1].set_xlabel("Истинные k2")
axes[1, 1].set_ylabel("Предсказанные k2")
axes[1, 1].set_title(f"Linear Regression: k2 (R² = {r2_lr_k2:.3f})")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============= 8. ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ =============
print("\n" + "=" * 50)
print("ВЫВОДЫ ПО МОДЕЛЯМ")
print("=" * 50)

if r2_rf_k1 > r2_lr_k1:
    print(f"✅ Для k1 лучше использовать Random Forest (R² = {r2_rf_k1:.3f})")
else:
    print(f"✅ Для k1 лучше использовать Linear Regression (R² = {r2_lr_k1:.3f})")

if r2_rf_k2 > r2_lr_k2:
    print(f"✅ Для k2 лучше использовать Random Forest (R² = {r2_rf_k2:.3f})")
else:
    print(f"✅ Для k2 лучше использовать Linear Regression (R² = {r2_lr_k2:.3f})")

print(f"\nРекомендуемый минимальный набор признаков:")
print(f"  Для k1: {', '.join(features_k1)}")
print(f"  Для k2: {', '.join(features_k2)}")
