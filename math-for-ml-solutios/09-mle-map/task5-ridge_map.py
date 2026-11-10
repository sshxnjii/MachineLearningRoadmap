import numpy as np 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

n_samples = 100
n_features = 10

x1 = np.random.normal(0, 1, n_samples)
x2 = x1 + np.random.normal(0, 0.1, n_samples)
X_rest = np.random.normal(0, 1, (n_samples, n_features - 2))
X = np.column_stack([x1, x2, X_rest])

true_w = np.zeros(n_features)
true_w[0] = 3.0
true_w[1] = -2.0
true_b = 1.0

noise = np.random.normal(0, 0.5, n_samples)
y = X @ true_w + true_b + noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model_mle = LinearRegression()
model_mle.fit(X_train, y_train)
w_mle = model_mle.coef_
b_mle = model_mle.intercept_

model_ridge = Ridge(0.01)
model_ridge.fit(X_train, y_train)
w_ridge = model_ridge.coef_
b_ridge = model_ridge.intercept_

print(f"{'Признак':<10} | {'Истинный':<10} | {'MLE':<12} | {'Ridge':<12}")
for i in range(n_features):
    print(f"x{i+1:<9} | {true_w[i]:<10.3f} | {w_mle[i]:<12.3f} | {w_ridge[i]:<12.3f}")

mse_mle = np.mean((y_test - model_mle.predict(X_test))**2)
mse_ridge = np.mean((y_test - model_ridge.predict(X_test))**2)

print(f"\nMSE на тестовой выборке:")
print(f"  MLE (без регуляризации): {mse_mle:.4f}")
print(f"  Ridge (L2-регуляризация): {mse_ridge:.4f}")
print(f"  Улучшение: {((mse_mle - mse_ridge) / mse_mle * 100):.1f}%")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
x_idx = np.arange(1, n_features + 1)
plt.bar(x_idx - 0.2, true_w, width=0.4, label='Истинные веса', color='green', alpha=0.7)
plt.bar(x_idx + 0.2, w_mle, width=0.4, label='MLE (LinearRegression)', color='red', alpha=0.7)
plt.xlabel('Признак')
plt.ylabel('Вес')
plt.title('Веса без регуляризации (MLE)')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(1, 2, 2)
plt.bar(x_idx - 0.2, true_w, width=0.4, label='Истинные веса', color='green', alpha=0.7)
plt.bar(x_idx + 0.2, w_ridge, width=0.4, label='Ridge (MAP)', color='blue', alpha=0.7)
plt.xlabel('Признак')
plt.ylabel('Вес')
plt.title('Веса с L2-регуляризацией (MAP)')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()