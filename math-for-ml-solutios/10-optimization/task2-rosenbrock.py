import numpy as np
import matplotlib.pyplot as plt 

def func(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])

def gd(grad, x0, lr=0.001, n_steps=10000):
    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(n_steps):
        g = grad(x)
        x -= lr * g
        trajectory.append(x)
    return np.array(trajectory)

def sgd(grad, x0, lr=0.001, n_steps=10000, noise_std=0.1):
    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(n_steps):
        g = grad(x)
        noisy_g = g + np.random.normal(0, noise_std, size=g.shape)
        x -= lr * noisy_g
        trajectory.append(x)
    return np.array(trajectory)

def adam(grad, x0, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, n_steps=10000):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    trajectory = [x.copy()]

    for t in range(1, n_steps + 1):
        g = grad(x)

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2

        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)

        x -= lr * m_hat / (np.sqrt(v_hat) + eps)

        trajectory.append(x.copy())
    
    return np.array(trajectory)
    
x0 = np.array([-1.5, 2.0])

traj_gd = gd(grad, x0)
traj_sgd = sgd(grad, x0)
traj_adam = adam(grad, x0)

print(f"GD:   конечная точка {traj_gd[-1]}, ошибка: {func(traj_gd[-1]):.6f}")
print(f"SGD:  конечная точка {traj_sgd[-1]}, ошибка: {func(traj_sgd[-1]):.6f}")
print(f"Adam: конечная точка {traj_adam[-1]}, ошибка: {func(traj_adam[-1]):.6f}")

plt.figure(figsize=(10, 8))

x_range = np.linspace(-2., 2., 200)
y_range = np.linspace(-1., 3., 200)
X, Y = np.meshgrid(x_range, y_range)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

contours = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='Blues', alpha=0.6)
plt.clabel(contours, inline=True, fontsize=8)

plt.plot(traj_gd[:, 0], traj_gd[:, 1], 'r.-', label='GD (lr=0.01)', markersize=4)
plt.plot(traj_sgd[:, 0], traj_sgd[:, 1], 'g.--', label='SGD (lr=0.05, noise)', markersize=4)
plt.plot(traj_adam[:, 0], traj_adam[:, 1], 'm.-', label='Adam (lr=0.1)', markersize=4)

plt.scatter(x0[0], x0[1], color='black', s=100, marker='*', label='Старт')
plt.scatter(0, 0, color='red', s=100, marker='X', label='Минимум (0,0)')

plt.title('Траектории на функции Розенброка')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-2, 2)
plt.ylim(-1, 3)
plt.show()


plt.figure(figsize=(10, 5))
steps = range(0, 10001, 10)
plt.semilogy(steps, [func(traj_gd[i]) for i in steps], 'r-', label='GD', linewidth=2)
plt.semilogy(steps, [func(traj_sgd[i]) for i in steps], 'g-', label='SGD', linewidth=2)
plt.semilogy(steps, [func(traj_adam[i]) for i in steps], 'm-', label='Adam', linewidth=2)
plt.xlabel('Шаг оптимизации')
plt.ylabel('Ошибка (логарифмическая шкала)')
plt.title('Скорость сходимости')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()