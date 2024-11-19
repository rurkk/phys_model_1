import matplotlib.pyplot as plt
import numpy as np

# Определяем параметры
u_values = [1, 4, 9]
Q_small = 1
Q_big = 4
delta = 10
t = np.linspace(-2 * np.pi, 4 * np.pi, 1000)

# Создаем подграфики
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
q_values = [Q_small, Q_big]
colors = {'even': ['purple', 'blue'], 'odd': ['orange', 'green']}

for row, q in enumerate(q_values):
    for parity, parity_label in enumerate(['even', 'odd']):
        ax = axes[row, parity]
        ax.set_title(f"{'Четные' if parity == 0 else 'Нечетные'} состояния, Q = {q}")
        ax.set_xlim(0, delta)
        ax.set_ylim(-delta / 2, delta)
        ax.grid()

        for u in u_values:
            y = np.sqrt(u * q ** 2 - t ** 2) / t
            condition = t ** 2 < u * q ** 2
            ax.plot(t[condition], y[condition], color=colors[parity_label][row], label=f'u={u}')

        function = np.tan if parity == 0 else lambda x: -1 / np.tan(x)
        func_values = function(t)
        func_values[:-1][np.diff(func_values) < 0] = np.nan
        ax.plot(t, func_values, label='tan(t)' if parity == 0 else 'cot(t)', color='red' if parity == 0 else 'cyan')

plt.tight_layout()
plt.legend()
plt.show()


# Второй график
def second_figure():
    plt.figure(figsize=(8, 6))
    delta = 10
    Q = 3
    u_values = [1, 2, 9]

    t = np.linspace(-2 * np.pi, 4 * np.pi, 1000)
    tan_values = np.tan(t)
    tan_values[:-1][np.diff(tan_values) < 0] = np.nan

    plt.xlim(-delta / 2, delta / 2)
    plt.ylim(-delta, delta)
    plt.axvline(x=0, color='black')
    plt.axhline(y=0, color='black')

    for u in u_values:
        odd_values = np.sqrt(u * Q ** 2 - t ** 2) / t
        even_values = -(t / np.sqrt(u * Q ** 2 - t ** 2))

        condition = t ** 2 < u * Q ** 2
        plt.plot(t[condition], odd_values[condition], color='magenta', label=f'Нечетные u={u}')
        plt.plot(t[condition], even_values[condition], color='lime', label=f'Четные u={u}')

    plt.plot(t, tan_values, color='orange', label='tan(t)')
    plt.grid()
    plt.title('График для второй фигуры')
    plt.legend()
    plt.show()


second_figure()


# Третий график
def third_figure():
    plt.figure(figsize=(8, 6))

    h = 1.054
    w = 1
    n = 10

    x = []
    y = []
    for i in range(-n, n + 1):
        temp = (abs(i) + 0.5) * h * w
        y.append(temp)
        x.append(i / temp)

    plt.hlines(y, -np.array(x), x, colors='red')

    x = []
    y = []
    step = 0.1
    for i in range(-n, n):
        start = i
        end = i + 1
        while start < end:
            temp = (abs(start) + 0.5) * h * w
            y.append(temp)
            x.append(start / temp)
            start += step

    plt.plot(x, y, color='purple')
    plt.xlabel('x')
    plt.ylabel('U(x)')
    plt.title('График U(x)')
    plt.grid()
    plt.show()


third_figure()


# Четвертый график
def fourth_figure():
    fig, axes = plt.subplots(2, figsize=(8, 6))

    x1 = np.linspace(-1, 0, 1000)
    x2 = np.linspace(0, 1, 1000)
    k_values = [1.8365, 4.8158]
    a = 1

    for i, k in enumerate(k_values):
        y1 = -np.sqrt(2 * k / (2 * k * a - np.sin(2 * k * a))) * np.sin(k * (x1 + a))
        y2 = np.sqrt(2 * k / (2 * k * a - np.sin(2 * k * a))) * np.sin(k * (x2 - a))

        axes[i].plot(x1, y1, color='blue')
        axes[i].plot(x2, y2, color='green')
        axes[i].set_title(f'k = {k}')
        axes[i].set_xlim(-1, 1)
        axes[i].grid()

    plt.tight_layout()
    plt.show()


fourth_figure()