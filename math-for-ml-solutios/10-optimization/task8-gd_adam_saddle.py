import numpy as np 
import matplotlib.pyplot as plt

def func(x):
    return x[0]**2 - x[1]**2

def grad(x):
    return np.array([2 * x[0], -2 * x[1]])

def gd(grad, x0, lr=0.1, n_steps=500):
    x = x0.copy()
    traj = [x.copy()]
    for _ in range(n_steps):
        x -= lr * grad(x)
        traj.append(x.copy())
    return np.array(traj)

def adam_with_noise(grad, x0, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, n_steps=500, noise_std=1e-4):
    x = x0.copy()
    traj = [x.copy()]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, n_steps + 1):
        g = grad(x) + np.random.normal(0, noise_std, size=x.shape)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(x.copy())
    return np.array(traj)

np.random.seed(42)

x0 = np.array([1e-4, 0])

traj_gd = gd(grad, x0)
traj_adam = adam_with_noise(grad, x0)

print(f"GD финальная точка:   [{traj_gd[-1, 0]:.2e} {traj_gd[-1, 1]:.2f}] (f = {func(traj_gd[-1]):.2f})")
print(f"Adam финальная точка: [{traj_adam[-1, 0]:.2e} {traj_adam[-1, 1]:.2f}] (f = {func(traj_adam[-1]):.2f})")

plt.figure(figsize=(10, 8))

x_range = np.linspace(-1, 1, 100)
y_range = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 - Y**2

contours = plt.contour(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.6)
plt.clabel(contours, inline=True, fontsize=8)

plt.plot(traj_gd[:, 0], traj_gd[:, 1], 'b-', label='GD', linewidth=2)
plt.plot(traj_adam[:, 0], traj_adam[:, 1], 'r-', label='Adam', linewidth=2)

plt.scatter(x0[0], x0[1], color='black', s=100, marker='*', label='Start', zorder=5)
plt.scatter(traj_gd[-1, 0], traj_gd[-1, 1], color='blue', s=100, marker='o', zorder=5)
plt.scatter(traj_adam[-1, 0], traj_adam[-1, 1], color='red', s=100, marker='o', zorder=5)
plt.scatter(0, 0, color='green', s=100, marker='X', label='(0, 0)', zorder=5)

plt.title('GD vs Adam на седловой точке')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.show()