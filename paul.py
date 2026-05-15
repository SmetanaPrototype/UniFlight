import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import json
import basis

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

rockets_list = ["amur", "falcon", "cz2c", "fireflyalpha", "electron", "antares", "zenit", "titan2", "isa", "jielong3", "starship", "vectorr"]

# rockets_list = [
#     "amur", "angara12", "antares", "blackarrow",
#     "cz2c", "delta2", "electron",
#     "epsilon", "falcon", "fireflyalpha", "isa", "jielong3",
#     "mlv", "newglenn", "rockot", "soyuz21v", "starship", "titan2",
#     "vectorr", "vulcanc", "zenit"
# ]

results = [read_rocket(rocket) for rocket in rockets_list]

data = {
    "s1": np.array([r[0] for r in results]),
    "s2": np.array([r[1] for r in results]),
    "N1": np.array([r[2] for r in results]),
    "N2": np.array([r[3] for r in results]),
    "k1": np.array([r[4] for r in results])
}
mask = data["k1"] < 4.1
data = {key: value[mask] for key, value in data.items()}

print(pd.DataFrame(data))

X = np.column_stack([data["s1"], data["s2"], data["N1"], data["N2"]])
y = data["k1"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)

print(f"R² для k1 (на тесте): {model.score(X_test, y_test):.3f}")

nr = read_rocket(basis.current_rocket)
new_rocket = [[nr[0], nr[1], nr[2], nr[3]]]
predicted_k1 = model.predict(new_rocket)[0]

# Принудительно ограничиваем предсказание значением < 4
if predicted_k1 >= 4.1:
    predicted_k1 = 4.0
    print(f"Предсказание было скорректировано (было >=4)")

print(f"\nПредсказанный k1: {predicted_k1:.2f}")

plt.figure(figsize=(6, 5))
plt.scatter(y_test, model.predict(X_test))
plt.plot([0, 3.5], [0, 3.5], 'r--')
plt.xlabel("Истинные k1 (тест)")
plt.ylabel("Предсказанные k1")
plt.tight_layout()