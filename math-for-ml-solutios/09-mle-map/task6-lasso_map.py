import numpy as np 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

n_samples = 100
n_features = 50

X = np.random.normal(0, 1, (n_samples, n_features))

true_w = np.zeros(n_features)
true_w[0] = 5.0
true_w[1] = -3.0
true_w[2] = 2.0
true_b = 1.0

noise = np.random.normal(0, 0.5, n_samples)
y = X @ true_w + true_b + noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model_mle = LinearRegression()
model_mle.fit(X_train, y_train)

model_ridge = Ridge(0.01)
model_ridge.fit(X_train, y_train)

model_lasso = Lasso(0.01)
model_lasso.fit(X_train, y_train)

print("СРАВНЕНИЕ: MLE vs Ridge vs Lasso")
print(f"{'Признак':<10} | {'Истинный':<10} | {'MLE':<10} | {'Ridge':<10} | {'Lasso':<10}")
for i in range(n_features):
    print(f"x{i+1:<9} | {true_w[i]:<10.3f} | {model_mle.coef_[i]:<10.3f} | "
          f"{model_ridge.coef_[i]:<10.3f} | {model_lasso.coef_[i]:<10.3f}")

n_nonzero_mle = np.sum(model_mle.coef_ != 0)
n_nonzero_ridge = np.sum(model_ridge.coef_ != 0)
n_nonzero_lasso = np.sum(model_lasso.coef_ != 0)

print(f"\nКоличество ненулевых весов:")
print(f"  MLE:   {n_nonzero_mle} / {n_features}")
print(f"  Ridge: {n_nonzero_ridge} / {n_features}")
print(f"  Lasso: {n_nonzero_lasso} / {n_features} ← feature selection!")

mse_mle = np.mean((y_test - model_mle.predict(X_test))**2)
mse_ridge = np.mean((y_test - model_ridge.predict(X_test))**2)
mse_lasso = np.mean((y_test - model_lasso.predict(X_test))**2)

print(f"\nMSE на тесте:")
print(f"  MLE:   {mse_mle:.4f}")
print(f"  Ridge: {mse_ridge:.4f}")
print(f"  Lasso: {mse_lasso:.4f}")

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
x_idx = np.arange(1, n_features + 1)
plt.scatter(x_idx, true_w, color='green', s=50, label='Истинные', zorder=5)
plt.plot(x_idx, model_mle.coef_, 'r.', markersize=8, label='MLE', alpha=0.7)
plt.plot(x_idx, model_ridge.coef_, 'b.', markersize=8, label='Ridge', alpha=0.7)
plt.plot(x_idx, model_lasso.coef_, 'm.', markersize=8, label='Lasso', alpha=0.7)
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
plt.xlabel('Признак')
plt.ylabel('Вес')
plt.title('Сравнение весов всех моделей')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
models = ['MLE', 'Ridge', 'Lasso']
nonzero = [n_nonzero_mle, n_nonzero_ridge, n_nonzero_lasso]
colors = ['red', 'blue', 'magenta']
plt.bar(models, nonzero, color=colors, alpha=0.7)
plt.axhline(y=3, color='green', linestyle='--', label='Истинное кол-во важных признаков (3)')
plt.ylabel('Количество ненулевых весов')
plt.title('Feature Selection: Lasso обнуляет лишнее')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()