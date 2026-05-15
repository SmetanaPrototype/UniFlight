import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

def read_rocket(rocket):
    print(rocket)
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

rockets_list = ["amur", "falcon", "cz2c", "fireflyalpha", "electron", "antares", "zenit"]

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
    "k1": np.array([r[4] for r in results]),
    "k2": np.array([r[5] for r in results]),
}
print(pd.DataFrame(data))

X = np.column_stack([data["s1"], data["s2"], data["N1"], data["N2"]])
y1 = data["k1"]
y2 = data["k2"]

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)
_, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)

model_k1 = LinearRegression().fit(X_train, y1_train)
model_k2 = LinearRegression().fit(X_train, y2_train)

print(f"R² для k1 (на тесте): {model_k1.score(X_test, y1_test):.3f}")
print(f"R² для k2 (на тесте): {model_k2.score(X_test, y2_test):.3f}")

nr = read_rocket("vulcanc")
new_rocket = [[nr[0], nr[1], nr[2], nr[3]]]
print(f"\nПредсказанные k1, k2: {model_k1.predict(new_rocket)[0]:.2f}, {model_k2.predict(new_rocket)[0]:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.scatter(y1_test, model_k1.predict(X_test))
ax1.plot([0, 3], [0, 3], 'r--')
ax1.set_xlabel("Истинные k1 (тест)")
ax1.set_ylabel("Предсказанные k1")
ax2.scatter(y2_test, model_k2.predict(X_test))
ax2.plot([0, 1.5], [0, 1.5], 'r--')
ax2.set_xlabel("Истинные k2 (тест)")
ax2.set_ylabel("Предсказанные k2")
plt.tight_layout()
plt.show()