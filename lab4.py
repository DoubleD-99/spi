"""
Анализ временного ряда: Годовой рост ВВП Российской Федерации (2000–2024)
Источник: Всемирный банк, индикатор NY.GDP.MKTP.KD.ZG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, anderson, norm
import warnings

warnings.filterwarnings('ignore')

# Настройка стилей для графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

# =============================================================================
# 1. ПОДГОТОВКА ДАННЫХ
# =============================================================================

# Исходные данные: годовой рост ВВП России, % (2000–2024)
data = {
    'year': list(range(2000, 2025)),
    'gdp_growth': [10.0, 5.1, 4.7, 7.3, 7.2, 6.4, 8.2, 8.5, 5.2, -7.8,
                   4.5, 4.3, 4.0, 1.8, 0.7, -2.0, 0.2, 1.8, 2.8, 2.2,
                   -2.7, 5.9, -1.4, 4.1, 4.3]
}

df = pd.DataFrame(data)
df['t'] = np.arange(1, len(df) + 1)  # Временной индекс для регрессии

print("=== Исходные данные ===")
print(df.head(25))
print(f"\nОбъём выборки: {len(df)} наблюдений")
print(f"Период: {df['year'].min()}–{df['year'].max()}")


# =============================================================================
# 2. ВИЗУАЛИЗАЦИЯ ВРЕМЕННОГО РЯДА
# =============================================================================

def plot_time_series(df):
    """Построение графика временного ряда с трендом и сглаживанием"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Фактические значения
    ax.plot(df['year'], df['gdp_growth'], 'bo-', label='Фактические значения',
            markersize=4, linewidth=1.5)

    # Линейный тренд
    coeffs = np.polyfit(df['t'], df['gdp_growth'], 1)
    trend = np.poly1d(coeffs)
    ax.plot(df['year'], trend(df['t']), 'r--', label='Линейный тренд', linewidth=2)

    # Скользящая средняя (5 лет)
    df['ma5'] = df['gdp_growth'].rolling(window=5, center=True, min_periods=1).mean()
    ax.plot(df['year'], df['ma5'], 'g-.', label='Скользящая средняя (5 лет)', linewidth=2)

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('Рост ВВП, %', fontsize=12)
    ax.set_title('Динамика годового роста ВВП России (2000–2024)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    return coeffs


trend_coeffs = plot_time_series(df)
print(f"\nУравнение тренда: y = {trend_coeffs[0]:.4f}·t + {trend_coeffs[1]:.4f}")


def plot_bar_chart(df):
    """Построение столбчатой диаграммы годового роста ВВП"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Цветовое кодирование: зелёный — рост, красный — спад
    colors = ['green' if val >= 0 else 'red' for val in df['gdp_growth']]

    # Построение столбчатой диаграммы
    bars = ax.bar(df['year'], df['gdp_growth'],
                  color=colors,
                  edgecolor='black',
                  linewidth=0.5,
                  alpha=0.8)

    # Оформление
    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('Рост ВВП, %', fontsize=12)
    ax.set_title('Столбчатая диаграмма годового роста ВВП России (2000–2024)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Добавление горизонтальной линии среднего значения
    mean_val = df['gdp_growth'].mean()
    ax.axhline(y=mean_val, color='blue', linestyle=':', linewidth=1.5,
               label=f'Среднее значение: {mean_val:.2f}%')

    # Легенда для цветов
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Экономический рост'),
        Patch(facecolor='red', edgecolor='black', label='Экономический спад'),
        plt.Line2D([0], [0], color='blue', linestyle=':', linewidth=1.5, label='Среднее значение')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Поворот подписей лет для читаемости
    ax.set_xticks(df['year'][::2])  # Подписываем каждый второй год
    ax.set_xticklabels(df['year'][::2], rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    return fig


# Вызов функции для построения столбчатой диаграммы
plot_bar_chart(df)

# =============================================================================
# 3. ПРОВЕРКА НА СТАЦИОНАРНОСТЬ: АВТОКОРРЕЛЯЦИИ
# =============================================================================

def calculate_autocorrelations(series, max_lag=10):
    """Расчёт выборочных автокорреляций"""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series, ddof=0)

    acf = []
    for k in range(max_lag + 1):
        if k == 0:
            acf.append(1.0)
        else:
            numerator = np.sum((series[:n - k] - mean) * (series[k:] - mean))
            acf.append(numerator / (n * var))

    return np.array(acf)


# Расчёт автокорреляций
acf_values = calculate_autocorrelations(df['gdp_growth'].values, max_lag=10)


# Построение коррелограммы
def plot_correlogram(acf, n, alpha=0.05):
    """Построение коррелограммы с доверительными интервалами"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Доверительные границы
    conf_bound = 2 / np.sqrt(n)

    lags = np.arange(len(acf))
    ax.bar(lags, acf, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(y=conf_bound, color='red', linestyle='--', label=f'95% доверительный интервал')
    ax.axhline(y=-conf_bound, color='red', linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xlabel('Лаг (k)', fontsize=11)
    ax.set_ylabel('Коэффициент автокорреляции', fontsize=11)
    ax.set_title('Коррелограмма временного ряда', fontsize=13, fontweight='bold')
    ax.set_xticks(lags)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Вывод значений
    print("\n=== Автокорреляции ===")
    for k, val in enumerate(acf[:11]):
        print(f"Лаг {k}: rₖ = {val:.3f}")

    return conf_bound


conf_int = plot_correlogram(acf_values, len(df))


# =============================================================================
# 4. ОБНАРУЖЕНИЕ АНОМАЛИЙ: МЕТОД ИРВИНА
# =============================================================================

def irwin_test(series, alpha=0.05):
    """Метод Ирвина для обнаружения аномальных наблюдений"""

    n = len(series)
    mean = np.mean(series)
    std = np.std(series, ddof=1)

    # Критическое значение (приближённое для n=25, α=0.05)
    lambda_crit = 2.5

    results = []
    anomalies = []

    for i, y in enumerate(series):
        lambda_t = abs(y - mean) / std
        is_anomaly = lambda_t > lambda_crit
        results.append({'index': i, 'value': y, 'lambda': lambda_t, 'anomaly': is_anomaly})
        if is_anomaly:
            anomalies.append(i)

    return pd.DataFrame(results), anomalies, lambda_crit


irwin_results, anomalies, lambda_crit = irwin_test(df['gdp_growth'].values)

print(f"\n=== Метод Ирвина (критическое значение λ = {lambda_crit}) ===")
print(irwin_results[irwin_results['lambda'] > 2.0][['index', 'value', 'lambda', 'anomaly']])


# Визуализация аномалий
def plot_anomalies(df, anomalies):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(df['year'], df['gdp_growth'], 'bo-', label='Значения', markersize=5)

    # Выделение аномалий
    if anomalies:
        anom_df = df.iloc[anomalies]
        ax.scatter(anom_df['year'], anom_df['gdp_growth'], color='red', s=100,
                   marker='*', label='Аномалия', zorder=5, edgecolors='black')

    ax.set_xlabel('Год')
    ax.set_ylabel('Рост ВВП, %')
    ax.set_title('Обнаружение аномальных уровней (метод Ирвина)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':')

    plt.tight_layout()
    plt.show()


plot_anomalies(df, anomalies)


# =============================================================================
# 5. ПРОВЕРКА НАЛИЧИЯ ТРЕНДА: МЕТОД СРАВНЕНИЯ СРЕДНИХ
# =============================================================================

def trend_test_comparison_of_means(series, alpha=0.05):
    """Проверка наличия тренда методом сравнения средних"""

    n = len(series)
    mid = n // 2

    part1 = series[:mid]
    part2 = series[mid:]

    mean1, mean2 = np.mean(part1), np.mean(part2)
    var1, var2 = np.var(part1, ddof=1), np.var(part2, ddof=1)
    n1, n2 = len(part1), len(part2)

    # t-статистика для неравных дисперсий (Уэлча)
    t_stat = abs(mean1 - mean2) / np.sqrt(var1 / n1 + var2 / n2)

    # Критическое значение (приближённо)
    df_welch = (var1 / n1 + var2 / n2) ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df_welch)

    return {
        'mean1': mean1, 'mean2': mean2,
        't_stat': t_stat, 't_crit': t_crit, 'df': df_welch,
        'trend_present': t_stat > t_crit
    }


trend_test = trend_test_comparison_of_means(df['gdp_growth'].values)

print(f"\n=== Метод сравнения средних ===")
print(f"Среднее (1-я половина): {trend_test['mean1']:.3f}%")
print(f"Среднее (2-я половина): {trend_test['mean2']:.3f}%")
print(f"t-статистика: {trend_test['t_stat']:.3f}")
print(f"t-критическое: {trend_test['t_crit']:.3f}")
print(f"Тренд присутствует: {trend_test['trend_present']}")


# =============================================================================
# 6 СРАВНЕНИЕ МЕТОДОВ СГЛАЖИВАНИЯ
# =============================================================================

def moving_averages_comparison(df, windows=[3, 5], alpha_exp=0.3):
    """Сравнение различных методов сглаживания с визуализацией"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Исходный ряд
    ax.plot(df['year'], df['gdp_growth'], 'k-', label='Исходный ряд',
            linewidth=1.5, alpha=0.8)

    # Скользящие средние
    colors = ['blue', 'green', 'orange']
    mse_results = {}

    for w, color in zip(windows, colors):
        df[f'ma{w}'] = df['gdp_growth'].rolling(window=w, center=True,
                                                min_periods=1).mean()
        ax.plot(df['year'], df[f'ma{w}'], color=color, label=f'MA({w})',
                linewidth=2.5, linestyle='-')

        # Расчёт MSE
        mse = np.mean((df['gdp_growth'] - df[f'ma{w}']) ** 2)
        mse_results[f'MA({w})'] = mse

    # Экспоненциальное сглаживание
    df['exp_smooth'] = df['gdp_growth'].ewm(alpha=alpha_exp, adjust=False).mean()
    ax.plot(df['year'], df['exp_smooth'], color='orange',
            label=f'Экспоненциальное (α={alpha_exp})',
            linewidth=2.5, linestyle='--')

    mse_exp = np.mean((df['gdp_growth'] - df['exp_smooth']) ** 2)
    mse_results['Exponential'] = mse_exp

    # Оформление графика
    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('Рост ВВП, %', fontsize=12)
    ax.set_title('Сравнение методов сглаживания временного ряда',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # Вывод результатов
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ КАЧЕСТВА СГЛАЖИВАНИЯ (MSE)")
    print("=" * 60)
    for method, mse in mse_results.items():
        print(f"{method:20s}: MSE = {mse:.3f}")
    print("=" * 60)

    return df, mse_results


# Вызов функции для этапа 6
df, mse_results = moving_averages_comparison(df, windows=[3, 5], alpha_exp=0.3)

# =============================================================================
# 7. ПОСТРОЕНИЕ МОДЕЛИ ТРЕНДА (МНК)
# =============================================================================

def fit_trend_model(df, degree=1):
    """Подбор полиномиальной модели тренда методом наименьших квадратов"""

    # Подготовка данных
    X = df['t'].values
    y = df['gdp_growth'].values

    # Полиномиальная регрессия
    coeffs = np.polyfit(X, y, degree)
    model = np.poly1d(coeffs)
    y_pred = model(X)

    # Расчёт метрик
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Остатки
    residuals = y - y_pred

    return {
        'coeffs': coeffs,
        'model': model,
        'y_pred': y_pred,
        'residuals': residuals,
        'r_squared': r_squared,
        'mse': np.mean(residuals ** 2)
    }


trend_model = fit_trend_model(df, degree=1)

print(f"\n=== Параметры линейной модели тренда ===")
print(f"Уравнение: ŷ = {trend_model['coeffs'][0]:.4f}·t + {trend_model['coeffs'][1]:.4f}")
print(f"R² = {trend_model['r_squared']:.4f}")
print(f"MSE = {trend_model['mse']:.4f}")


# =============================================================================
# 8. АНАЛИЗ ОСТАТКОВ
# =============================================================================

def analyze_residuals(residuals, alpha=0.05):
    """Комплексный анализ остаточной компоненты"""

    n = len(residuals)
    results = {}

    # 1. Проверка на случайность (критерий серий)
    median = np.median(residuals)
    signs = ['+' if r > median else '-' if r < median else '' for r in residuals]

    # Подсчёт серий
    series_count = 1
    max_series_len = 1
    current_len = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1] and signs[i] != '':
            current_len += 1
            max_series_len = max(max_series_len, current_len)
        else:
            series_count += 1
            current_len = 1

    # Критерии для случайности
    n_effective = len([s for s in signs if s != ''])
    crit_N_lower = (n_effective + 1 - 1.96 * np.sqrt(n_effective - 1)) / 2
    crit_K_upper = 3.3 * (np.log(n_effective) + 1)

    results['randomness'] = {
        'series_count': series_count,
        'max_series_len': max_series_len,
        'criterion_N': series_count > crit_N_lower,
        'criterion_K': max_series_len < crit_K_upper,
        'passed': series_count > crit_N_lower and max_series_len < crit_K_upper
    }

    # 2. Проверка нормальности (критерий Шапиро-Уилка)
    shapiro_stat, shapiro_p = shapiro(residuals)
    results['normality'] = {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'passed': shapiro_p > alpha
    }

    # 3. Проверка математического ожидания = 0 (t-критерий)
    mean_res = np.mean(residuals)
    std_res = np.std(residuals, ddof=1)
    t_stat = abs(mean_res) / (std_res / np.sqrt(n))
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    results['mean_zero'] = {
        'mean': mean_res,
        't_stat': t_stat,
        't_crit': t_crit,
        'passed': t_stat < t_crit
    }

    # 4. Проверка независимости (критерий Дарбина-Уотсона)
    dw_num = np.sum(np.diff(residuals) ** 2)
    dw_denom = np.sum(residuals ** 2)
    dw_stat = dw_num / dw_denom
    # Приближённые критические значения для n=25, k=1
    d1, d2 = 1.29, 1.45  # из таблиц Дарбина-Уотсона
    if dw_stat < d1:
        autocorr = 'positive'
    elif dw_stat > 4 - d1:
        autocorr = 'negative'
    elif d1 <= dw_stat <= d2 or (4 - d2) <= dw_stat <= (4 - d1):
        autocorr = 'inconclusive'
    else:
        autocorr = 'none'

    results['independence'] = {
        'dw_stat': dw_stat,
        'autocorrelation': autocorr,
        'passed': autocorr == 'none'
    }

    return results


residual_analysis = analyze_residuals(trend_model['residuals'])

print(f"\n=== Анализ остаточной компоненты ===")
print(f"1. Случайность колебаний: {'✓' if residual_analysis['randomness']['passed'] else '✗'}")
print(f"   - Число серий: {residual_analysis['randomness']['series_count']}")
print(f"   - Макс. длина серии: {residual_analysis['randomness']['max_series_len']}")

print(f"\n2. Нормальность распределения: {'✓' if residual_analysis['normality']['passed'] else '✗'}")
print(f"   - Критерий Шапиро-Уилка: p = {residual_analysis['normality']['shapiro_p']:.4f}")

print(f"\n3. Мат. ожидание = 0: {'✓' if residual_analysis['mean_zero']['passed'] else '✗'}")
print(f"   - t-статистика: {residual_analysis['mean_zero']['t_stat']:.3f}")

print(f"\n4. Независимость уровней: {residual_analysis['independence']['autocorrelation']}")
print(f"   - Статистика Дарбина-Уотсона: d = {residual_analysis['independence']['dw_stat']:.3f}")


# Визуализация остатков
def plot_residuals_analysis(df, residuals, y_pred):
    """Комплексная визуализация анализа остатков"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Остатки во времени
    axes[0, 0].plot(df['year'], residuals, 'bo-')
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Год')
    axes[0, 0].set_ylabel('Остатки')
    axes[0, 0].set_title('Остатки модели')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Гистограмма остатков с нормальной кривой
    axes[0, 1].hist(residuals, bins=8, density=True, alpha=0.7, edgecolor='black')
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x_norm, norm.pdf(x_norm, np.mean(residuals), np.std(residuals)),
                    'r-', linewidth=2, label='Нормальное распределение')
    axes[0, 1].set_xlabel('Значение остатка')
    axes[0, 1].set_ylabel('Плотность')
    axes[0, 1].set_title('Распределение остатков')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Остатки vs Предсказанные значения
    axes[1, 0].scatter(y_pred, residuals, alpha=0.7, edgecolors='black')
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Предсказанные значения')
    axes[1, 0].set_ylabel('Остатки')
    axes[1, 0].set_title('Остатки против предсказаний')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Q-Q plot для проверки нормальности
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q plot остатков')

    plt.tight_layout()
    plt.show()


plot_residuals_analysis(df, trend_model['residuals'], trend_model['y_pred'])


# =============================================================================
# 9. ЦИКЛИЧЕСКАЯ КОМПОНЕНТА: ОТКЛОНЕНИЯ ОТ ТРЕНДА
# =============================================================================

def extract_cyclical_component(df, trend_model):
    """Выделение циклической компоненты как отклонений от тренда"""

    df['trend'] = trend_model['y_pred']
    df['cyclical'] = df['gdp_growth'] - df['trend']

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.bar(df['year'], df['cyclical'], color='skyblue', edgecolor='black', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel('Год')
    ax.set_ylabel('Циклическое отклонение, %')
    ax.set_title('Циклическая компонента временного ряда')
    ax.grid(True, alpha=0.3, axis='y')

    # Выделение рецессионных периодов
    recession_years = df[df['cyclical'] < -2]['year']
    for year in recession_years:
        ax.axvspan(year - 0.4, year + 0.4, color='red', alpha=0.2)

    plt.tight_layout()
    plt.show()

    return df


df = extract_cyclical_component(df, trend_model)

print(f"\n=== Циклическая компонента ===")
print("Периоды значительных отрицательных отклонений (рецессии):")
print(df[df['cyclical'] < -2][['year', 'gdp_growth', 'cyclical']])

# =============================================================================
# 10. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
# =============================================================================

print("\n" + "=" * 60)
print("СВОДНЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА")
print("=" * 60)

summary = {
    'Объём выборки': len(df),
    'Период': f"{df['year'].min()}–{df['year'].max()}",
    'Среднее значение': f"{df['gdp_growth'].mean():.3f}%",
    'Стандартное отклонение': f"{df['gdp_growth'].std():.3f}%",
    'Коэффициент вариации': f"{df['gdp_growth'].std() / abs(df['gdp_growth'].mean()) * 100:.1f}%",
    'Минимум': f"{df['gdp_growth'].min():.1f}% ({df.loc[df['gdp_growth'].idxmin(), 'year']} г.)",
    'Максимум': f"{df['gdp_growth'].max():.1f}% ({df.loc[df['gdp_growth'].idxmax(), 'year']} г.)",
    'Уравнение тренда': f"ŷ = {trend_model['coeffs'][0]:.4f}·t + {trend_model['coeffs'][1]:.4f}",
    'Коэффициент детерминации (R²)': f"{trend_model['r_squared']:.4f}",
    'Аномалии (метод Ирвина)': f"{len(anomalies)} ({[df.iloc[a]['year'] for a in anomalies] if anomalies else 'нет'})",
    'Тренд (метод сравнения средних)': 'Присутствует (нисходящий)' if trend_test['trend_present'] else 'Отсутствует',
    'Остатки: нормальность': '✓' if residual_analysis['normality']['passed'] else '✗',
    'Остатки: независимость': residual_analysis['independence']['autocorrelation'],
}

for key, value in summary.items():
    print(f"{key:40s}: {value}")
