import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import json
import basis

def read_rocket(rocket):
    """Читает данные ракеты из JSON и вычисляет параметры."""
    # print(rocket) # Закомментировал для чистоты вывода
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
        N.append(thrust[i]/(mstep[i]*9.81))
    return s[0], s[1], N[0], N[1], coefs[0]

# --- Загрузка данных ---
rockets_list = ["amur", "falcon", "cz2c", "fireflyalpha", "electron", "antares", "zenit", "titan2", "isa",
                "vectorr", "angara12", "blackarrow", "epsilon", "starship", "soyuz21v", "cyclone2",
                "terran1", "soyuz5", "cosmos3m", "kinetika2"]

# Создаем словарь для хранения данных по каждой ракете
rocket_data = {}
for rocket in rockets_list:
    s1, s2, N1, N2, k1 = read_rocket(rocket)
    rocket_data[rocket] = {
        "s1": s1, "s2": s2, "N1": N1, "N2": N2, "k1_actual": k1
    }

print("\n" + "="*80)
print("🚀 ЗАПУСК КРОСС-ВАЛИДАЦИИ (Leave-One-Out)")
print("="*80)

# --- Кросс-валидация ---
results = []
for rocket_held_out in rockets_list:
    # 1. Формируем обучающую выборку (все, кроме текущей ракеты)
    X_train = []
    y_train = []
    for rocket, data in rocket_data.items():
        if rocket == rocket_held_out:
            continue
        X_train.append([data["s1"], data["s2"], data["N1"], data["N2"]])
        y_train.append(data["k1_actual"])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # 2. Обучаем модель KNN
    pipeline = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=3, weights='distance')
    )
    pipeline.fit(X_train, y_train)
    
    # 3. Предсказываем для исключенной ракеты
    held_out_data = rocket_data[rocket_held_out]
    X_test = np.array([[held_out_data["s1"], held_out_data["s2"], 
                        held_out_data["N1"], held_out_data["N2"]]])
    predicted_k1 = pipeline.predict(X_test)[0]
    
    # Ограничения для предсказания
    if predicted_k1 >= 4.1:
        predicted_k1 = 4.0
    if predicted_k1 <= 0.0:
        predicted_k1 = 0.1
    
    actual_k1 = held_out_data["k1_actual"]
    error = predicted_k1 - actual_k1
    abs_error = abs(error)
    
    results.append({
        "rocket": rocket_held_out,
        "actual_k1": actual_k1,
        "predicted_k1": predicted_k1,
        "error": error,
        "abs_error": abs_error
    })
    
    # Красивый вывод для каждой ракеты
    status = "✅" if abs_error < 0.5 else "⚠️"
    print(f"{status} {rocket_held_out:15s} | actual: {actual_k1:4.2f} | predicted: {predicted_k1:4.2f} | error: {error:+.2f}")

# --- Итоговая статистика ---
print("\n" + "="*80)
print("📊 ИТОГОВАЯ СТАТИСТИКА")
print("="*80)

df_results = pd.DataFrame(results)
mean_abs_error = df_results["abs_error"].mean()
max_abs_error = df_results["abs_error"].max()
std_abs_error = df_results["abs_error"].std()
within_05 = (df_results["abs_error"] < 0.5).sum()
within_10 = (df_results["abs_error"] < 1.0).sum()

print(f"Средняя абсолютная ошибка (MAE): {mean_abs_error:.3f}")
print(f"Максимальная абсолютная ошибка:   {max_abs_error:.3f}")
print(f"Стандартное отклонение ошибки:    {std_abs_error:.3f}")
print(f"Предсказаний с ошибкой < 0.5:     {within_05} из {len(rockets_list)} ({within_05/len(rockets_list)*100:.1f}%)")
print(f"Предсказаний с ошибкой < 1.0:     {within_10} из {len(rockets_list)} ({within_10/len(rockets_list)*100:.1f}%)")

# Самые плохие предсказания
print("\n🔴 ХУДШИЕ ПРЕДСКАЗАНИЯ (max ошибка):")
worst = df_results.nlargest(3, "abs_error")
for _, row in worst.iterrows():
    print(f"   {row['rocket']:15s} | actual: {row['actual_k1']:4.2f} | predicted: {row['predicted_k1']:4.2f} | error: {row['error']:+.2f}")

# Самые хорошие предсказания
print("\n🟢 ЛУЧШИЕ ПРЕДСКАЗАНИЯ (min ошибка):")
best = df_results.nsmallest(3, "abs_error")
for _, row in best.iterrows():
    print(f"   {row['rocket']:15s} | actual: {row['actual_k1']:4.2f} | predicted: {row['predicted_k1']:4.2f} | error: {row['error']:+.2f}")
