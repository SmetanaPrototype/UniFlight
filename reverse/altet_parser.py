import matplotlib.pyplot as plt

top_folders = ['П0', 'П1', 'П2', 'П3', 'П4', 'С5', 'С6', 'С7']
fig, ax = plt.subplots(2, 1)

for f_ in top_folders:
    time_vec = []
    alt_vec = []
    tet_vec = []
    with open(f_ + '/АЛТЭТ.txt', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) == 3:
                time_vec.append(float(parts[0]))
                alt_vec.append(float(parts[1]))
                tet_vec.append(float(parts[2]))


    ax[0].plot(time_vec, alt_vec, label=f_)
    ax[1].plot(time_vec, tet_vec, label=f_)


ax[0].set_title('Alt(t)')
ax[0].legend()

ax[1].set_title('Tet(t)')
ax[1].legend()

plt.tight_layout()
plt.show()