import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

manuf = [0.233, 0.300, 0.145, 0.679, 0.423, 0.295, 2.018, 0.147, 1.060, 0.509,
         0.109, 0.359, 0.740, 0.742, 0.920, 1.082, 0.339, 0.673, 0.668, 0.372,
         0.160, 0.315, 0.567, 0.692, 0.104, -0.198, 0.980, -0.326, 0.635, 0.804,
         0.538, 0.717, 0.714, -0.075, 0.925, 0.222, 0.416, 0.584, 0.187, 0.069]

serv = [0.794, 0.749, 0.312, 0.237, 0.337, 0.232, 0.056, 0.529, 0.001, 0.248,
        0.448, -0.127, 0.236, 0.370, 0.769, 0.236, 0.528, 0.524, 0.598, 0.278,
        0.518, 0.275, 0.632, 1.146, 0.719, 0.914, 0.832, 0.642, 0.667, 0.176,
        0.353, 1.039, 0.893, 0.915, 1.386, 0.655, 0.610, 0.950, 0.681, -0.042]

agri = [0.262, 0.281, 0.094, 0.573, 0.401, 0.195, 0.892, 0.358, 0.534, 0.320,
        0.156, 0.488, 0.361, 0.197, 0.413, 0.210, 0.430, 0.315, 0.287, 0.254,
        0.098, 0.178, 0.312, 0.540, 0.085, 0.621, 0.445, 0.390, 0.410, 0.290,
        0.225, 0.580, 0.367, 0.710, 0.503, 0.165, 0.176, 0.330, 0.112, -0.156]

# 1.4
sectors = ['Manufacturing', 'Services', 'Agriculture']
means = [np.mean(manuf), np.mean(serv), np.mean(agri)]

plt.figure(figsize=(8,5))
bars = plt.bar(sectors, means, color=['steelblue','coral','seagreen'], edgecolor='black')
plt.ylabel('Средняя эластичность')
plt.title('Средняя эластичность занятости по секторам')
for b, m in zip(bars, means):
    plt.text(b.get_x()+b.get_width()/2, m+0.01, f'{m:.3f}', ha='center')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.boxplot([manuf, serv, agri], tick_labels=sectors, showmeans=True)
plt.ylabel('Эластичность')
plt.title('Распределение эластичности по секторам')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 1.5.1
alpha = 0.05

stat_m, p_m = stats.shapiro(manuf)
stat_s, p_s = stats.shapiro(serv)
stat_a, p_a = stats.shapiro(agri)

print(f"Manufacturing: W = {stat_m:.4f}, p = {p_m:.4f}")
print(f"Services:      W = {stat_s:.4f}, p = {p_s:.4f}")
print(f"Agriculture:   W = {stat_a:.4f}, p = {p_a:.4f}")

# 1.5.2
rho_ms, p_ms = stats.spearmanr(manuf, serv)
rho_ma, p_ma = stats.spearmanr(manuf, agri)
rho_sa, p_sa = stats.spearmanr(serv, agri)

print(f"Manufacturing-Services:    rho = {rho_ms:.4f}, p = {p_ms:.4f}")
print(f"Manufacturing-Agriculture: rho = {rho_ma:.4f}, p = {p_ma:.4f}")
print(f"Services-Agriculture:      rho = {rho_sa:.4f}, p = {p_sa:.4f}")

# 1.5.3
u_ms, p_ms = stats.mannwhitneyu(manuf, serv, alternative='two-sided')
u_ma, p_ma = stats.mannwhitneyu(manuf, agri, alternative='two-sided')
u_sa, p_sa = stats.mannwhitneyu(serv, agri, alternative='two-sided')

print(f"Manufacturing vs Services:    U = {u_ms:.1f}, p = {p_ms:.4f}")
print(f"Manufacturing vs Agriculture: U = {u_ma:.1f}, p = {p_ma:.4f}")
print(f"Services vs Agriculture:      U = {u_sa:.1f}, p = {p_sa:.4f}")

# 1.5.4
w_ms, p_ms = stats.wilcoxon(manuf, serv, alternative='two-sided')
w_ma, p_ma = stats.wilcoxon(manuf, agri, alternative='two-sided')
w_sa, p_sa = stats.wilcoxon(serv, agri, alternative='two-sided')

print(f"Manufacturing vs Services:    T = {w_ms:.1f}, p = {p_ms:.4f}")
print(f"Manufacturing vs Agriculture: T = {w_ma:.1f}, p = {p_ma:.4f}")
print(f"Services vs Agriculture:      T = {w_sa:.1f}, p = {p_sa:.4f}")

# 2.2.1
tau_ma, p_tau = stats.kendalltau(manuf, agri)
print(f"Kendall Manufacturing-Agriculture: tau = {tau_ma:.4f}, p = {p_tau:.4f}")

# 2.2.2
h_stat, p_kw = stats.kruskal(manuf, serv, agri)
print(f"Kruskal-Wallis: H = {h_stat:.4f}, p = {p_kw:.4f}")

# 2.2.3
stat_med, p_med, med_val, table = stats.median_test(manuf, serv, agri)
print(f"Median test: stat = {stat_med:.4f}, p = {p_med:.4f}, grand  = {med_val:.4f}")

# 2.3
plt.figure(figsize=(7,5))
plt.scatter(manuf, agri, alpha=0.7, edgecolors='black', s=50)
z = np.polyfit(manuf, agri, 1)
p_line = np.poly1d(z)
x_line = np.linspace(min(manuf), max(manuf), 100)
plt.plot(x_line, p_line(x_line), 'r--', label=f'Тренд: y={z[0]:.2f}x+{z[1]:.2f}')
plt.xlabel('Manufacturing')
plt.ylabel('Agriculture')
plt.title('Связь эластичности: Manufacturing vs Agriculture')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
