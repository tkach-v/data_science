import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
import json


def analyze_real_data(dates, real_inflation_values):
    real_mean_inflation = np.mean(real_inflation_values)
    real_std_deviation_inflation = np.std(real_inflation_values)
    real_median_inflation = np.median(real_inflation_values)
    real_min_inflation = np.min(real_inflation_values)
    real_max_inflation = np.max(real_inflation_values)
    real_variance_inflation = np.var(real_inflation_values)

    print("Статистика даних з парсингу:")
    print("Середнє значення інфляції:", real_mean_inflation)
    print("Стандартне відхилення інфляції:", real_std_deviation_inflation)
    print("Медіанна інфляція:", real_median_inflation)
    print("Мінімальне значення інфляції:", real_min_inflation)
    print("Максимальне значення інфляції:", real_max_inflation)
    print("Відхилення інфляції:", real_variance_inflation)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dates, real_inflation_values, marker='o', linestyle='-', color='black', markersize=4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Inflation")
    plt.xticks(rotation=45)
    ax.grid(True)


def generate_and_analyze_synthetic_data(dates, real_inflation_values):
    degree = 12
    noise_std = 1

    np.random.seed(0)
    coefficients = np.polyfit(np.arange(len(dates)), real_inflation_values, degree)
    synthetic_trend_values = np.polyval(coefficients, np.arange(len(dates)))
    print("\nМодель:")
    print(np.poly1d(coefficients))
    synthetic_inflation_values = synthetic_trend_values + np.random.normal(0, noise_std, len(dates))

    synthetic_mean_inflation = np.mean(synthetic_inflation_values)
    synthetic_std_deviation_inflation = np.std(synthetic_inflation_values)
    synthetic_median_inflation = np.median(synthetic_inflation_values)
    synthetic_min_inflation = np.min(synthetic_inflation_values)
    synthetic_max_inflation = np.max(synthetic_inflation_values)
    synthetic_variance_inflation = np.var(synthetic_inflation_values)

    print("\nСтатистика синтезованих даних:")
    print("Середнє значення інфляції:", synthetic_mean_inflation)
    print("Стандартне відхилення інфляції:", synthetic_std_deviation_inflation)
    print("Медіана інфляції:", synthetic_median_inflation)
    print("Мінімальне значення інфляції:", synthetic_min_inflation)
    print("Максимальне значення інфляції:", synthetic_max_inflation)
    print("Відхилення інфляції:", synthetic_variance_inflation)

    return synthetic_trend_values, synthetic_inflation_values


if __name__ == '__main__':
    with open('data.json', 'r') as f1:
        data = json.load(f1)

    parsed_dates = data['year']
    parsed_inflation_values = data['value']

    root = tk.Tk()
    root.title("Діаграма інфляції гривні із трендом")

    frame1 = tk.Frame(root)
    frame1.pack(side=tk.LEFT)

    analyze_real_data(dates=parsed_dates, real_inflation_values=parsed_inflation_values)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(parsed_dates, parsed_inflation_values, marker='o', linestyle='-', label='Дані парсингу')
    ax1.set_xlabel("Дата")
    ax1.set_ylabel("Інфляція")
    plt.xticks(rotation=45)
    ax1.grid(True)
    plt.legend()
    plt.title("Графік інфляції гривні (Парсинг)")

    canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
    canvas_widget1 = canvas1.get_tk_widget()
    canvas_widget1.pack()

    frame2 = tk.Frame(root)
    frame2.pack(side=tk.RIGHT)

    synthetic_trend_values, synthetic_inflation_values = generate_and_analyze_synthetic_data(
        dates=parsed_dates,
        real_inflation_values=parsed_inflation_values)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(parsed_dates, synthetic_inflation_values, marker='o', label='Синтезовані дані')
    ax2.plot(parsed_dates, synthetic_trend_values, label='Синтезований тренд')
    ax2.set_xlabel("Дата")
    ax2.set_ylabel("Інфляція")
    plt.xticks(rotation=45)
    ax2.grid(True)
    plt.legend()
    plt.title("Графік інфляції гривні (Синтезований)")

    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas_widget2 = canvas2.get_tk_widget()
    canvas_widget2.pack()

    root.mainloop()
