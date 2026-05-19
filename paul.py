import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import json
import basis

def compute_polynomial_coefficients(X, y, degree=2):
    """
    Вычисляет коэффициенты полиномиальной регрессии на основе обучающих данных.
    Возвращает функцию predict(x) и коэффициенты.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    def predict(s1, s2, N1, N2):
        x = np.array([[s1, s2, N1, N2]])
        x_poly = poly.transform(x)
        return model.predict(x_poly)[0]

    return predict, model.coef_, model.intercept_, poly

def read_rocket(rocket):
    print(rocket)
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

# Загружаем данные всех ракет
rockets_list = ["amur", "falcon", "cz2c", "fireflyalpha", "electron", "antares", "zenit", "titan2", "isa",
                "vectorr", "angara12", "blackarrow", "epsilon", "starship", "soyuz21v", "cyclone2",
                "terran1", "soyuz5", "cosmos3m", "kinetika2"]

results = [read_rocket(rocket) for rocket in rockets_list]

data = {
    "s1": np.array([r[0] for r in results]),
    "s2": np.array([r[1] for r in results]),
    "N1": np.array([r[2] for r in results]),
    "N2": np.array([r[3] for r in results]),
    "k1": np.array([r[4] for r in results])
}

print("\n" + "="*80)
print("Собранные данные по ракетам:")
print(pd.DataFrame(data))
print("="*80)

X = np.column_stack([data["s1"], data["s2"], data["N1"], data["N2"]])
y = data["k1"]

# ==================== 1. ПОЛИНОМИАЛЬНАЯ ИНТЕРПОЛЯЦИЯ (динамическая) ====================
print("\n📐 Расчет полиномиальных коэффициентов на основе имеющихся ракет...")
poly_predict, poly_coefs, poly_intercept, poly = compute_polynomial_coefficients(X, y, degree=2)

print(f"Коэффициенты полинома (первые 10 из {len(poly_coefs)}): {poly_coefs[:10]}")
print(f"Intercept: {poly_intercept:.6f}")
print(f"Порядок полинома: {poly.powers_.shape}")

# ==================== 2. KNN-РЕГРЕССИЯ ====================
print("\n🤖 Обучение KNN-регрессора...")
pipeline_knn = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=3, weights='distance')
)

# LOOCV для оценки качества
loo = LeaveOneOut()
y_pred_loo = np.zeros(len(y))
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    pipeline_loo = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=3, weights='distance')
    )
    pipeline_loo.fit(X_train, y_train)
    y_pred_loo[test_idx[0]] = pipeline_loo.predict(X_test)[0]

r2_knn = r2_score(y, y_pred_loo)
print(f"R² для k1 (LOOCV, KNN): {r2_knn:.3f}")

# Обучаем KNN на ВСЕХ данных
pipeline_knn.fit(X, y)

# ==================== 3. ПРОГНОЗ ДЛЯ ТЕКУЩЕЙ РАКЕТЫ ====================
nr = read_rocket(basis.current_rocket)
new_rocket = np.array([[nr[0], nr[1], nr[2], nr[3]]])

# Прогноз через полином (на основе всех ракет)
predicted_k1_poly = poly_predict(nr[0], nr[1], nr[2], nr[3])

# Прогноз через KNN
predicted_k1_knn = pipeline_knn.predict(new_rocket)[0]

# Ограничения для полинома
if predicted_k1_poly >= 4.1:
    predicted_k1_poly = 4.0
    print("\n⚠️ Полиномиальное предсказание было скорректировано (>4)")
if predicted_k1_poly <= 0.0:
    predicted_k1_poly = 0.1
    print("⚠️ Полиномиальное предсказание было скорректировано (<=0)")

# Ограничения для KNN
if predicted_k1_knn >= 4.1:
    predicted_k1_knn = 4.0
    print("⚠️ KNN предсказание было скорректировано (>4)")
if predicted_k1_knn <= 0.0:
    predicted_k1_knn = 0.1
    print("⚠️ KNN предсказание было скорректировано (<=0)")

print("\n" + "="*80)
print(f"🎯 Ракета: {basis.current_rocket}")
print(f"📊 Параметры: s1={nr[0]}, s2={nr[1]}, N1={nr[2]:.4f}, N2={nr[3]:.4f}")
print("-"*80)
print(f"🔮 Предсказанный k1 (ПОЛИНОМ, вычислен на {len(rockets_list)} ракетах): {predicted_k1_poly:.2f}")
print(f"🤖 Предсказанный k1 (KNN, n_neighbors=3):                    {predicted_k1_knn:.2f}")
print("="*80)

# Проверка качества обучения KNN
print(f"\n📈 Качество KNN на тестовых данных (LOOCV): R² = {r2_knn:.3f}")
if r2_knn < 0.5:
    print("⚠️ Внимание: модель плохо предсказывает. Возможно, нужно больше данных или другой метод.")