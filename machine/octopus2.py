import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import json
import basis


def read_rocket(rocket):
    """Читает данные ракеты из JSON и вычисляет параметры."""
    path = "rocket_lib/" + rocket + "/constant.json"
    with open(path, "r") as r_file:
        r_data = json.load(r_file)
    s = r_data["structural_values"]
    mb = r_data["block_mass"]
    mpn = r_data["payload_mass"]
    thrust = r_data["thrust"]
    coefs = r_data["attack_coefs"]
    mstep = []
    mstep.append(mpn + sum(mb))
    mstep.append(mpn + sum(mb) - mb[0])
    N = []
    for i in range(len(mb)):
        N.append(thrust[i] / (mstep[i] * 9.81))
    return s[0], s[1], N[0], N[1], coefs[0], coefs[1]


# --- Загрузка данных ---
rockets_list = [
    "amur",
    "falcon",
    "cz2c",
    "fireflyalpha",
    "electron",
    "antares",
    "zenit",
    "titan2",
    "isa",
    "angara12",
    "blackarrow",
    "starship",
    "soyuz21v",
    "cyclone2",
    "terran1",
    "soyuz5",
    "cosmos3m",
    "kinetika2",
    "stalker",
    "vectorr",
]

# Создаем словарь для хранения данных по каждой ракете
rocket_data = {}
for rocket in rockets_list:
    s1, s2, N1, N2, k1, k2 = read_rocket(rocket)
    rocket_data[rocket] = {"s1": s1, "s2": s2, "N1": N1, "N2": N2, "k1_actual": k1, "k2_actual": k2}

print("\n" + "=" * 100)
print("🚀 ЗАПУСК КРОСС-ВАЛИДАЦИИ (Leave-One-Out)")
print("📊 МОДЕЛЬ:  k1, s2 и N2")
print("=" * 100)

# --- Кросс-валидация ---
results = []
for rocket_held_out in rockets_list:
    X_train = []
    y_train = []
    for rocket, data in rocket_data.items():
        if rocket == rocket_held_out:
            continue
        X_train.append([data["s1"], data["N1"], data["k1_actual"]])
        y_train.append(data["k2_actual"])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # 2. Обучаем модель KNN
    pipeline = make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=3, weights="distance")
    )
    pipeline.fit(X_train, y_train)

    # 3. Предсказываем для исключенной ракеты
    held_out_data = rocket_data[rocket_held_out]
    X_test = np.array([[held_out_data["s2"], held_out_data["N2"],  held_out_data["k1_actual"]]])
    predicted_k2 = pipeline.predict(X_test)[0]


    actual_k2 = held_out_data["k2_actual"]
    error = predicted_k2 - actual_k2
    abs_error = abs(error)

    results.append(
        {
            "rocket": rocket_held_out,
            "s2": held_out_data["s2"],
            "N2": held_out_data["N2"],
            "k1_actual": held_out_data["k1_actual"],
            "actual_k2": actual_k2,
            "predicted_k2": predicted_k2,
            "error": error,
            "abs_error": abs_error,
        }
    )

    # Красивый вывод для каждой ракеты
    status = "✅" if abs_error < 0.5 else "⚠️"
    print(
        f"{status} {rocket_held_out:15s} | s2={held_out_data['s2']:5.1f} N2={held_out_data['N2']:5.3f} | actual={actual_k2:4.2f} pred={predicted_k2:4.2f} | err={error:+.2f}"
    )

# --- Итоговая статистика ---
print("\n" + "=" * 100)
print("📊 ИТОГОВАЯ СТАТИСТИКА (модель: s2 + N2 + k1)")
print("=" * 100)

df_results = pd.DataFrame(results)
mean_abs_error = df_results["abs_error"].mean()
max_abs_error = df_results["abs_error"].max()
std_abs_error = df_results["abs_error"].std()
within_05 = (df_results["abs_error"] < 0.5).sum()
within_10 = (df_results["abs_error"] < 1.0).sum()

print(f"Средняя абсолютная ошибка (MAE): {mean_abs_error:.3f}")
print(f"Максимальная абсолютная ошибка:   {max_abs_error:.3f}")
print(f"Стандартное отклонение ошибки:    {std_abs_error:.3f}")
print(
    f"Предсказаний с ошибкой < 0.5:     {within_05} из {len(rockets_list)} ({within_05/len(rockets_list)*100:.1f}%)"
)
print(
    f"Предсказаний с ошибкой < 1.0:     {within_10} из {len(rockets_list)} ({within_10/len(rockets_list)*100:.1f}%)"
)

# Самые плохие предсказания
print("\n🔴 ХУДШИЕ ПРЕДСКАЗАНИЯ (max ошибка):")
worst = df_results.nlargest(3, "abs_error")
for _, row in worst.iterrows():
    print(
        f"   {row['rocket']:15s} | s2={row['s2']:5.1f} N2={row['N2']:5.3f} k1_actual={row['k1_actual']:5.3f} | actual={row['actual_k2']:4.2f} pred={row['predicted_k2']:4.2f} | err={row['error']:+.2f}"
    )

# Самые хорошие предсказания
print("\n🟢 ЛУЧШИЕ ПРЕДСКАЗАНИЯ (min ошибка):")
best = df_results.nsmallest(3, "abs_error")
for _, row in best.iterrows():
    print(
        f"   {row['rocket']:15s} | s2={row['s2']:5.1f} N2={row['N2']:5.3f} k1_actual={row['k1_actual']:5.3f} | actual={row['actual_k2']:4.2f} pred={row['predicted_k2']:4.2f} | err={row['error']:+.2f}"
    )

# ============================================================================
# ГРАФИК: ПРЕДСКАЗАННЫЕ vs ФАКТИЧЕСКИЕ ЗНАЧЕНИЯ
# ============================================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# Диапазон для линии y=x
all_vals = df_results['actual_k2'].tolist() + df_results['predicted_k2'].tolist()
min_val = min(all_vals) - 0.2
max_val = max(all_vals) + 0.2

# Точки с цветом в зависимости от ошибки
scatter = plt.scatter(df_results['actual_k2'], df_results['predicted_k2'],
                      c=df_results['abs_error'], cmap='RdYlGn_r', s=120,
                      edgecolors='black', linewidth=1.2, alpha=0.9)

# Линия идеального предсказания
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.5, label='Идеальное предсказание (y=x)')

# Подписи точек
for _, row in df_results.iterrows():
    plt.annotate(row['rocket'],
                 (row['actual_k2'], row['predicted_k2']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=9, alpha=0.8)

plt.xlabel('Фактический k2', fontsize=14)
plt.ylabel('Предсказанный k2', fontsize=14)
plt.title('Предсказанные vs Фактические значения k2 (модель: k1 + s2 + N2)', fontsize=14)

# Добавляем цветовую шкалу
cbar = plt.colorbar(scatter)
cbar.set_label('Абсолютная ошибка', fontsize=12)

# Добавляем текстовую информацию
plt.text(0.05, 0.95, f'MAE = {df_results["abs_error"].mean():.3f}',
         transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()