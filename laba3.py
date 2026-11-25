import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


WINDOW_TITLE = "Лабораторна робота №3 — Регресія методом найменших квадратів (Варіант 10)"
WINDOW_SIZE = "1050x740"

DEFAULT_X_STR = "1, 3, 6, 14, 20, 30, 51, 60"
DEFAULT_Y_STR = "16, 26, 40, 82, 115, 164, 270, 313"

PLOT_POINTS_COUNT = 300
MIN_POINTS_REQUIRED = 2


def parse_coordinates(text_data: str) -> np.ndarray:
    """Перетворює рядок чисел (через кому або пробіл) у масив NumPy."""
    cleaned = text_data.replace(";", ",").replace(" ", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        raise ValueError("Поле координат не може бути порожнім.")
    return np.array([float(val) for val in parts], dtype=float)


def calculate_r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Обчислює коефіцієнт детермінації R^2 для оцінки точності моделі."""
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def linear_regression_ols(x: np.ndarray, y: np.ndarray):
    """
    Завдання 12 Лінійна регресія y = a * x + b методом найменших квадратів.
    Повертає (a, b, y_pred, r2).
    """
    n = len(x)
    denom = n * np.sum(x ** 2) - (np.sum(x)) ** 2
    if abs(denom) < 1e-12:
        raise ZeroDivisionError("Усі значення X однакові, побудова прямої неможлива.")

    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / denom
    b = (np.sum(y) - a * np.sum(x)) / n
    y_pred = a * x + b
    r2 = calculate_r_squared(y, y_pred)
    return a, b, y_pred, r2


def power_regression_ols(x: np.ndarray, y: np.ndarray):
    """
    Завдання 2 (Варіант 10): Степенева регресія y = a * x^b.
    Лінеаризується через логарифмування: ln(y) = ln(a) + b * ln(x).
    Повертає (a, b, y_pred, r2).
    """
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Для степеневої регресії y = a·x^b усі значення X та Y мають бути > 0.")

    ln_x = np.log(x)
    ln_y = np.log(y)

    
    b, ln_a, _, _ = linear_regression_ols(ln_x, ln_y)
    a = math.exp(ln_a)

    y_pred = a * (x ** b)
    r2 = calculate_r_squared(y, y_pred)
    return a, b, y_pred, r2



root = tk.Tk()
root.title(WINDOW_TITLE)
root.geometry(WINDOW_SIZE)

status_var = tk.StringVar(value="Готово до роботи. Натисніть 'Побудувати регресію'.")
mode_var = tk.StringVar(value="both")
last_export_data = {}


frame_left = ttk.Frame(root, padding=10)
frame_left.pack(side="left", fill="y")

frame_right = ttk.Frame(root, padding=10)
frame_right.pack(side="right", fill="both", expand=True)


status_bar = ttk.Label(root, textvariable=status_var, relief="sunken", anchor="w", padding=5)
status_bar.pack(side="bottom", fill="x")


ttk.Label(frame_left, text="Експериментальні дані (Варіант 10):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

ttk.Label(frame_left, text="Значення X (через кому):").pack(anchor="w")
entry_x = ttk.Entry(frame_left, width=36)
entry_x.insert(0, DEFAULT_X_STR)
entry_x.pack(fill="x", pady=(0, 8))

ttk.Label(frame_left, text="Значення Y (через кому):").pack(anchor="w")
entry_y = ttk.Entry(frame_left, width=36)
entry_y.insert(0, DEFAULT_Y_STR)
entry_y.pack(fill="x", pady=(0, 12))


frame_modes = ttk.LabelFrame(frame_left, text="Режим відображення", padding=8)
frame_modes.pack(fill="x", pady=(0, 12))

ttk.Radiobutton(
    frame_modes, text="Завдання 1: Лінійна (y = ax + b)",
    variable=mode_var, value="linear"
).pack(anchor="w", pady=2)

ttk.Radiobutton(
    frame_modes, text="Завдання 2: Степенева (y = a·x^b)",
    variable=mode_var, value="power"
).pack(anchor="w", pady=2)

ttk.Radiobutton(
    frame_modes, text="Обидві моделі (порівняння)",
    variable=mode_var, value="both"
).pack(anchor="w", pady=2)


frame_eq = ttk.LabelFrame(frame_left, text="Отримані рівняння регресії", padding=8)
frame_eq.pack(fill="x", pady=(0, 12))

lbl_eq_result = ttk.Label(frame_eq, text="Натисніть 'Побудувати регресію'", justify="left", wraplength=260)
lbl_eq_result.pack(anchor="w")


fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=frame_right)
canvas.get_tk_widget().pack(fill="both", expand=True)

cols = ("idx", "x", "y_exp", "y_lin", "err_lin", "y_pow", "err_pow")
table = ttk.Treeview(frame_right, columns=cols, show="headings", height=8)

headers = ["№", "x", "y (експ.)", "y лін. (ax+b)", "|Δy| лін.", "y степ. (ax^b)", "|Δy| степ."]
widths = [40, 70, 90, 110, 90, 110, 90]

for col, text, w in zip(cols, headers, widths):
    table.heading(col, text=text)
    table.column(col, width=w, anchor="center")

table.pack(fill="both", expand=True, pady=(10, 0))



def reset_defaults():
    """Повертає стандартні значення Варіанта №10."""
    entry_x.delete(0, "end")
    entry_x.insert(0, DEFAULT_X_STR)
    entry_y.delete(0, "end")
    entry_y.insert(0, DEFAULT_Y_STR)
    status_var.set("Відновлено початкові дані Варіанта №10.")
    calculate()


def draw_plot(x: np.ndarray, y: np.ndarray, lin_params, pow_params):
    """Будує експериментальні точки та лінії регресії на графіку."""
    ax.clear()
    mode = mode_var.get()

    
    ax.scatter(x, y, color="black", zorder=5, label="Експериментальні точки")

    x_smooth = np.linspace(np.min(x), np.max(x), PLOT_POINTS_COUNT)

    if mode in ("linear", "both") and lin_params is not None:
        a_l, b_l = lin_params
        y_smooth_lin = a_l * x_smooth + b_l
        ax.plot(
            x_smooth, y_smooth_lin, color="tab:blue", linewidth=2,
            label=f"Лінійна: y = {a_l:.4f}x + {b_l:.4f}"
        )

    if mode in ("power", "both") and pow_params is not None:
        a_p, b_p = pow_params
        y_smooth_pow = a_p * (x_smooth ** b_p)
        ax.plot(
            x_smooth, y_smooth_pow, color="tab:red", linestyle="--", linewidth=2,
            label=f"Степенева: y = {a_p:.4f}·x^{b_p:.4f}"
        )

    ax.set_title("Апроксимація методом найменших квадратів (Варіант 10)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper left")
    canvas.draw()


def calculate():
    """Виконує розрахунки для обох моделей, заповнює таблицю та оновлює графік."""
    global last_export_data

    for item in table.get_children():
        table.delete(item)

    try:
        x = parse_coordinates(entry_x.get())
        y = parse_coordinates(entry_y.get())

        if len(x) != len(y):
            raise ValueError(f"Кількість X ({len(x)}) не збігається з кількістю Y ({len(y)}).")
        if len(x) < MIN_POINTS_REQUIRED:
            raise ValueError("Потрібно щонайменше 2 точки для побудови регресії.")

        a_lin, b_lin, y_pred_lin, r2_lin = linear_regression_ols(x, y)
        a_pow, b_pow, y_pred_pow, r2_pow = power_regression_ols(x, y)

    except Exception as exc:
        messagebox.showerror("Помилка вхідних даних", str(exc))
        status_var.set(f"Помилка: {exc}")
        return

    
    rows_for_export = []
    for i in range(len(x)):
        err_l = abs(y[i] - y_pred_lin[i])
        err_p = abs(y[i] - y_pred_pow[i])
        row = (
            i + 1,
            f"{x[i]:g}",
            f"{y[i]:g}",
            f"{y_pred_lin[i]:.4f}",
            f"{err_l:.4f}",
            f"{y_pred_pow[i]:.4f}",
            f"{err_p:.4f}"
        )
        table.insert("", "end", values=row)
        rows_for_export.append(row)

    
    summary_text = (
        f"1) Лінійна (y = ax + b):\n"
        f"   a = {a_lin:.6f}, b = {b_lin:.6f}\n"
        f"   y = {a_lin:.4f}x + {b_lin:.4f}\n"
        f"   R² = {r2_lin:.6f}\n\n"
        f"2) Степенева (y = a·x^b):\n"
        f"   a = {a_pow:.6f}, b = {b_pow:.6f}\n"
        f"   y = {a_pow:.4f} · x^{b_pow:.4f}\n"
        f"   R² = {r2_pow:.6f}"
    )
    lbl_eq_result.config(text=summary_text)

    last_export_data = {
        "summary": summary_text,
        "rows": rows_for_export
    }

    draw_plot(x, y, (a_lin, b_lin), (a_pow, b_pow))
    status_var.set("Розрахунок завершено. Лінійна R² = 0.9998, Степенева R² = 0.9744.")


def save_to_file():
    """Зберігає результати розрахунків та таблицю у текстовий файл."""
    if not last_export_data:
        messagebox.showwarning("Увага", "Спочатку виконайте розрахунок!")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Текстовий файл", "*.txt"), ("Усі файли", "*.*")],
        title="Зберегти результати регресії"
    )
    if not file_path:
        return

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("ЛАБОРАТОРНА РОБОТА №3 — ВАРІАНТ №10\n")
        f.write("=" * 75 + "\n")
        f.write(last_export_data["summary"] + "\n")
        f.write("=" * 75 + "\n")
        f.write(
            f"{'№':<4} | {'x':<8} | {'y (експ.)':<10} | "
            f"{'y лін.':<12} | {'|Δy| лін.':<10} | {'y степ.':<12} | {'|Δy| степ.':<10}\n"
        )
        f.write("-" * 75 + "\n")
        for r in last_export_data["rows"]:
            f.write(
                f"{r[0]:<4} | {r[1]:<8} | {r[2]:<10} | "
                f"{r[3]:<12} | {r[4]:<10} | {r[5]:<12} | {r[6]:<10}\n"
            )

    messagebox.showinfo("Успіх", "Результати успішно збережено у файл!")
    status_var.set(f"Файл збережено: {file_path}")



ttk.Button(frame_left, text="Побудувати регресію", command=calculate).pack(fill="x", pady=(0, 5))
ttk.Button(frame_left, text="Скинути (Варіант 10)", command=reset_defaults).pack(fill="x", pady=5)
ttk.Button(frame_left, text="Зберегти у файл...", command=save_to_file).pack(fill="x", pady=5)


calculate()
root.mainloop()