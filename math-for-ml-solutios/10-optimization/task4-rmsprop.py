import numpy as np 
import matplotlib.pyplot as plt 

def func(x):
    return x[0]**2 + 100 * x[1]**2

def grad(x):
    return np.array([2 * x[0], 200 * x[1]])

def gd(grad, x0, lr=0.01, n_steps=100):
    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(n_steps):
        x -= lr * grad(x)
        trajectory.append(x.copy())
    return np.array(trajectory)

def rmsprop(grad, x0, lr=0.01, r=0.9, eps=1e-8, n_steps=100):
    x = x0.copy()
    s = np.zeros_like(x)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        g = grad(x)
        s = r * s + (1 - r) * g**2
        x -= lr * g / (np.sqrt(s + 1) + eps)
        trajectory.append(x.copy())
    return np.array(trajectory)

x0 = np.array([-4.0, 4.0])

traj_gd = gd(grad, x0, lr=0.01, n_steps=100)
traj_rmsprop = rmsprop(grad, x0, lr=0.1, r=0.9, n_steps=100)

print(f"GD:      конечная точка {traj_gd[-1]}, ошибка: {func(traj_gd[-1]):.4f}")
print(f"RMSProp: конечная точка {traj_rmsprop[-1]}, ошибка: {func(traj_rmsprop[-1]):.4f}")

plt.figure(figsize=(10, 8))

x_range = np.linspace(-5, 5, 200)
y_range = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + 100 * Y**2

contours = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='Blues', alpha=0.6)
plt.clabel(contours, inline=True, fontsize=8)

plt.plot(traj_gd[:, 0], traj_gd[:, 1], 'r.-', label='GD (lr=0.01)', markersize=4)
plt.plot(traj_rmsprop[:, 0], traj_rmsprop[:, 1], 'b.--', label='RMSProp (lr=0.1, ρ=0.9)', markersize=4)

plt.scatter(x0[0], x0[1], color='black', s=100, marker='*', label='Старт')
plt.scatter(0, 0, color='red', s=100, marker='X', label='Минимум (0,0)')

plt.title('GD vs RMSProp на параболе-овраге')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()