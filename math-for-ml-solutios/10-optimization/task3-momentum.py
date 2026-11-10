import numpy as np 
import matplotlib.pyplot as plt 

def func(x):
    return x[0]**2 + 100 * x[1]**2

def grad(x):
    return np.array([2 * x[0], 200 * x[1]])

def momentum_gd(grad, x0, lr=0.01, b=0.9, n_steps=100):
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(n_steps):
        v = b * v + (1 - b) * grad(x)
        x -= lr * v
        history.append(x.copy())
    return np.array(history)

x0 = np.array([-4.0, 4.0])

traj_momentum = momentum_gd(grad, x0, lr=0.01, b=0.9, n_steps=100)

print(f"Momentum: конечная точка {traj_momentum[-1]}, ошибка: {func(traj_momentum[-1]):.4f}")

plt.figure(figsize=(10, 8))

x_range = np.linspace(-5, 5, 200)
y_range = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + 100 * Y**2

contours = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='Blues', alpha=0.6)
plt.clabel(contours, inline=True, fontsize=8)

plt.plot(traj_momentum[:, 0], traj_momentum[:, 1], 'b.--', label='Momentum (lr=0.01, β=0.9)', markersize=4)

plt.scatter(x0[0], x0[1], color='black', s=100, marker='*', label='Старт')
plt.scatter(0, 0, color='red', s=100, marker='X', label='Минимум (0,0)')

plt.title('Momentum на параболе-овраге')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()