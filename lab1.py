import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


data = [0.233, 0.300, 0.145, 0.679, 0.423, 0.295, 2.018, 0.147, 1.060, 0.509,
        0.109, 0.359, 0.740, 0.742, 0.920, 1.082, 0.339, 0.673, 0.668, 0.372,
        0.160, 0.315, 0.567, 0.692, 0.104, -0.198, 0.980, -0.326, 0.635, 0.804,
        0.538, 0.717, 0.714, -0.075, 0.925, 0.222, 0.416, 0.584, 0.187, 0.069]

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# boxplot
axes[0].boxplot(data, patch_artist=True,
boxprops=dict(facecolor='lightblue', color='blue'),
medianprops=dict(color='red', linewidth=2),
whiskerprops=dict(color='blue'),
capprops=dict(color='blue'),
flierprops=dict(marker='o', markerfacecolor='orange',
markersize=8, linestyle='none', markeredgecolor='black'))

axes[0].axhline(np.mean(data), color='green', linestyle='--', linewidth=2, label='Среднее')
axes[0].axhline(np.median(data), color='red', linestyle='-', linewidth=2, label='Медиана')
axes[0].set_title('Boxplot', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Значения')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# violin plot
axes[1].violinplot(data, positions=[1], widths=0.8, showmeans=False, showmedians=True, showextrema=True)
axes[1].scatter(np.random.normal(1, 0.02, len(data)), data, alpha=0.3, color='purple', s=30, label='Данные')
axes[1].axhline(np.median(data), color='red', linestyle='-', linewidth=2, label='Медиана')
axes[1].axhline(np.percentile(data, 25), color='gray', linestyle='--', linewidth=1.5, label='z1 (25%)')
axes[1].axhline(np.percentile(data, 75), color='gray', linestyle='--', linewidth=1.5, label='z2 (75%)')
axes[1].set_title('Скрипичная диаграмма', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Значения')
axes[1].set_xticks([1])
axes[1].set_xticklabels([''])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# вывод статистики
print("Статистика данных:\n")
print(f"Количество наблюдений (n): {len(data)}")
print(f"Минимум: {min(data):.3f}")
print(f"Максимум: {max(data):.3f}")
print(f"Среднее: {np.mean(data):.3f}")
print(f"Медиана: {np.median(data):.3f}")
print(f"z1 (25%): {np.percentile(data, 25):.3f}")
print(f"z2 (75%): {np.percentile(data, 75):.3f}")
print(f"Стд. отклонение: {np.std(data, ddof=1):.3f}")
print(f"Дисперсия: {np.var(data, ddof=1):.3f}")