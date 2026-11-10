import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

n_tosses = 5
k_heads = 4
k_tails = n_tosses - k_heads

p_mle = k_heads / n_tosses
print(f"Данные: {k_heads} орлов из {n_tosses} бросков")
print(f"MLE оценка p̂ = {p_mle:.3f}")

alpha_prior = 2
beta_prior = 2

alpha_post = alpha_prior + k_heads
beta_post = beta_prior + k_tails

p_map = (alpha_post - 1) / (alpha_post + beta_post - 2)
print(f"Prior: Beta({alpha_prior}, {beta_prior})")
print(f"Posterior: Beta({alpha_post}, {beta_post})")
print(f"MAP оценка p̂ = {p_map:.3f}")

p_values = np.linspace(0, 1, 500)

prior_pdf = beta.pdf(p_values, alpha_prior, beta_prior)
posterior_pdf = beta.pdf(p_values, alpha_post, beta_post)

plt.figure(figsize=(12, 6))

plt.plot(p_values, prior_pdf, 'gray', linestyle='--', linewidth=2, 
         label=f'Prior Beta({alpha_prior},{beta_prior})', alpha=0.7)

plt.plot(p_values, posterior_pdf, 'blue', linewidth=3, 
         label=f'Posterior Beta({alpha_post},{beta_post})')

plt.axvline(x=p_mle, color='red', linestyle=':', linewidth=2, 
            label=f'MLE (p={p_mle:.2f})')
plt.axvline(x=p_map, color='green', linestyle='-', linewidth=2, 
            label=f'MAP (p={p_map:.2f})')

plt.xlabel('Вероятность орла (p)')
plt.ylabel('Плотность вероятности')
plt.title('Bayes vs MLE на маленькой выборке (5 бросков)')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)

plt.tight_layout()
plt.show()

print("\nИНТЕРПРЕТАЦИЯ:")
print(f"• MLE дает p=0.8 — мы поверили в 4 орла из 5 как в истину.")
print(f"• MAP дает p={p_map:.2f} — Prior 'утянул' оценку к 0.5.")
print(f"• Разница: {abs(p_mle - p_map):.3f}")
print(f"\nЕсли бы было 100 бросков и 80 орлов:")
p_mle_big = 80/100
p_map_big = (2 + 80 - 1) / (2 + 80 + 2 + 20 - 2)
print(f"  MLE = {p_mle_big:.3f}, MAP = {p_map_big:.3f}")
print(f"  Разница: {abs(p_mle_big - p_map_big):.3f} ← Prior почти не влияет!")