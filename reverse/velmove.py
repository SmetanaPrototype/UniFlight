

import matplotlib.pyplot as plt

top_folders = ['П0', 'П1', 'П2', 'П3', 'П4', 'С5', 'С6', 'С7']
fig, ax = plt.subplots(2, 3)

def to_float(val):
    try:
        return float(val)
    except ValueError:
        return float('nan')

for f_ in top_folders:
    C1 = []
    C2 = []
    C3 = []
    C4 = []
    C5 = []
    C6 = []
    C7 = []
    C8 = []

    with open(f_ + '/VXTI1, VYTI1, VZTI1, XTI1, YTI1, ZTI1.txt', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) == 8:
                C1.append(to_float(parts[0]))
                C2.append(to_float(parts[1]))
                C3.append(to_float(parts[2]))
                C4.append(to_float(parts[3]))
                C5.append(to_float(parts[4]))
                C6.append(to_float(parts[5]))
                C7.append(to_float(parts[6]))
                C8.append(to_float(parts[7]))

    # plt.plot(C1, C8, label=f_)
    ax[0,0].plot(C1, C3, label=f_)
    ax[0,1].plot(C1, C4, label=f_)
    ax[0,2].plot(C1, C5, label=f_)
    ax[1,0].plot(C1, C6, label=f_)
    ax[1,1].plot(C1, C7, label=f_)
    ax[1,2].plot(C1, C8, label=f_)


ax[0,0].set_title('Velocity(t)')
ax[0,0].legend()

ax[0,1].set_title('C4')
ax[0,1].legend()

ax[0,2].set_title('C5')
ax[0,2].legend()

ax[1,0].set_title('C6')
ax[1,0].legend()

ax[1,1].set_title('C7')
ax[1,1].legend()

ax[1,2].set_title('C8')
ax[1,2].legend()

plt.tight_layout()
plt.show()