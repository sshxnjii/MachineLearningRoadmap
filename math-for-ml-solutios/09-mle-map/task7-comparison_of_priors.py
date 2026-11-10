import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm, laplace, t

w = np.linspace(-5, 5, 1000)

pdf_norm = norm.pdf(w, 0, 1)

pdf_laplace = laplace.pdf(w, 0, 1 / np.sqrt(2))

pdf_student = t.pdf(w / np.sqrt(3), df=3) / np.sqrt(3)

plt.figure(figsize=(12, 6))

plt.plot(w, pdf_norm, 'b--', linewidth=2, label='N(0, 1) -> Ridge (L2)')
plt.plot(w, pdf_laplace, 'r--', linewidth=2, label='Laplace(0, 1/sqrt(2)) -> Lasso (L1)')
plt.plot(w, pdf_student, 'g-.', linewidth=2, label='Student-t(df=3) -> Robust')

plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Значение веса (w)')
plt.ylabel('Плотность вероятности p(w)')
plt.title('Сравнение априорных распределений (MAP priors)')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 0.8)

plt.tight_layout()
plt.show()

print("АНАЛИЗ ФОРМЫ РАСПРЕДЕЛЕНИЙ:")
print("-" * 50)
print(f"Плотность в нуле (w=0):")
print(f"  Гаусс:   {pdf_norm[500]:.3f}")
print(f"  Лаплас:  {pdf_laplace[500]:.3f} ← ОСТРЫЙ ПИК!")
print(f"  Стьюдент:{pdf_student[500]:.3f}")

print(f"\nПлотность на хвосте (w=4):")
print(f"  Гаусс:   {pdf_norm[900]:.5f}")
print(f"  Лаплас:  {pdf_laplace[900]:.5f}")
print(f"  Стьюдент:{pdf_student[900]:.5f} ← ТЯЖЕЛЫЙ ХВОСТ!")