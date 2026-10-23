import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import quad


INTEGRALS = {
    "№10. 1) ∫(0.6..1.5) dx / √(2x + 1)": {
        "f": lambda x: 1.0 / np.sqrt(2 * x + 1),
        "f_single": lambda x: 1.0 / math.sqrt(2 * x + 1),
        "a": 0.6,
        "b": 1.5,
        "title": r"$\int_{0.6}^{1.5} \frac{dx}{\sqrt{2x + 1}}$"
    },
    "№10. 2) ∫(0.4..0.8) tg(x² + 0.5) / (1 + 2x²) dx": {
        "f": lambda x: np.tan(x**2 + 0.5) / (1 + 2 * x**2),
        "f_single": lambda x: math.tan(x**2 + 0.5) / (1 + 2 * x**2),
        "a": 0.4,
        "b": 0.8,
        "title": r"$\int_{0.4}^{0.8} \frac{\tan(x^2 + 0.5)}{1 + 2x^2} dx$"
    },
    "№10. 3) ∫(0.8..1.7) dx / √(2x² + 0.3)": {
        "f": lambda x: 1.0 / np.sqrt(2 * x**2 + 0.3),
        "f_single": lambda x: 1.0 / math.sqrt(2 * x**2 + 0.3),
        "a": 0.8,
        "b": 1.7,
        "title": r"$\int_{0.8}^{1.7} \frac{dx}{\sqrt{2x^2 + 0.3}}$"
    }
}


def method_rectangles(f, a, b, n):
    """Метод серединних прямокутників"""
    h = (b - a) / n
    x_mid = a + (np.arange(n) + 0.5) * h
    return h * np.sum(f(x_mid))

def method_trapezoids(f, a, b, n):
    """Метод трапецій"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def method_monte_carlo(f, a, b, n, seed=42):
    """Метод Монте-Карло (метод середнього значення)"""
    np.random.seed(seed)
    x_rand = np.random.uniform(a, b, n)
    return (b - a) * np.mean(f(x_rand))

def analytical_value(f_single, a, b):
    """Точне значення інтеграла за допомогою SciPy Quad"""
    val, _ = quad(f_single, a, b)
    return val


root = tk.Tk()
root.title("Лабораторна робота №2 — Числові методи обчислення інтегралів")
root.geometry("1000x720")

status_var = tk.StringVar(value="Готово. Оберіть інтеграл та натисніть 'Обчислити'")
calculated_rows = []


frame_left = ttk.Frame(root, padding=10)
frame_left.pack(side="left", fill="y")


frame_right = ttk.Frame(root, padding=10)
frame_right.pack(side="right", fill="both", expand=True)


status_bar = ttk.Label(root, textvariable=status_var, relief="sunken", anchor="w", padding=5)
status_bar.pack(side="bottom", fill="x")


ttk.Label(frame_left, text="Оберіть завдання (Варіант №10):").pack(anchor="w")
combo_int = ttk.Combobox(frame_left, values=list(INTEGRALS.keys()), state="readonly", width=38)
combo_int.current(0)
combo_int.pack(fill="x", pady=(0, 10))

ttk.Label(frame_left, text="Нижня межа (a):").pack(anchor="w")
entry_a = ttk.Entry(frame_left); entry_a.pack(fill="x", pady=(0, 5))

ttk.Label(frame_left, text="Верхня межа (b):").pack(anchor="w")
entry_b = ttk.Entry(frame_left); entry_b.pack(fill="x", pady=(0, 10))

def update_limits_from_combo():
    data = INTEGRALS[combo_int.get()]
    entry_a.delete(0, "end"); entry_a.insert(0, str(data["a"]))
    entry_b.delete(0, "end"); entry_b.insert(0, str(data["b"]))
    status_var.set(f"Обрано інтеграл: {combo_int.get()}")

update_limits_from_combo()


fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=frame_right)
canvas.get_tk_widget().pack(fill="both", expand=True)


cols = ("n", "analit", "rect", "trap", "mc")
table = ttk.Treeview(frame_right, columns=cols, show="headings", height=8)

headers = ["N", "Аналіт. значення", "Метод прямокутників", "Метод трапецій", "Метод Монте-Карло"]
widths = [60, 160, 170, 160, 160]

for col, text, w in zip(cols, headers, widths):
    table.heading(col, text=text)
    table.column(col, width=w, anchor="center")

table.pack(fill="both", expand=True, pady=(10, 0))

def draw_plot(f, a, b, title_str):
    ax.clear()
    x_vals = np.linspace(a, b, 300)
    y_vals = f(x_vals)

    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=r'$f(x)$')
    ax.fill_between(x_vals, y_vals, color='skyblue', alpha=0.4, label='Площа інтегрування')
    
    ax.set_title(f"Графік функції: {title_str}", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right")
    canvas.draw()

def on_select_change(event):
    update_limits_from_combo()
    data = INTEGRALS[combo_int.get()]
    draw_plot(data["f"], data["a"], data["b"], data["title"])

combo_int.bind("<<ComboboxSelected>>", on_select_change)

def calculate():
    global calculated_rows
    calculated_rows.clear()
    
    for item in table.get_children():
        table.delete(item)

    selected_key = combo_int.get()
    data = INTEGRALS[selected_key]
    f = data["f"]
    f_single = data["f_single"]

    try:
        a = float(entry_a.get())
        b = float(entry_b.get())
        if a >= b:
            raise ValueError("a має бути меншим за b!")
    except ValueError as e:
        messagebox.showerror("Помилка", f"Некоректні межі інтегрування: {e}")
        status_var.set("Помилка у введенні меж інтегрування.")
        return

    
    exact_val = analytical_value(f_single, a, b)

    n_values = [10, 20, 50, 100, 1000]

    for n in n_values:
        val_rect = method_rectangles(f, a, b, n)
        val_trap = method_trapezoids(f, a, b, n)
        val_mc = method_monte_carlo(f, a, b, n)

        row = (
            n,
            f"{exact_val:.6f}",
            f"{val_rect:.6f}",
            f"{val_trap:.6f}",
            f"{val_mc:.6f}"
        )
        table.insert("", "end", values=row)
        calculated_rows.append(row)

    draw_plot(f, a, b, data["title"])
    status_var.set(f"Розрахунок успішно виконано для {selected_key}")

def save_to_file():
    if not calculated_rows:
        messagebox.showwarning("Увага", "Спочатку виконай обчислення!")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Текстовий файл", "*.txt"), ("Усі файли", "*.*")],
        title="Зберегти результати"
    )

    if file_path:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"РЕЗУЛЬТАТИ ОБЧИСЛЕННЯ ІНТЕГРАЛА\n")
            file.write(f"Завдання: {combo_int.get()}\n")
            file.write(f"Межі: a = {entry_a.get()}, b = {entry_b.get()}\n")
            file.write("=" * 80 + "\n")
            file.write(f"{'N':<6} | {'Аналіт. значення':<18} | {'Прямокутників':<18} | {'Трапецій':<18} | {'Монте-Карло':<18}\n")
            file.write("-" * 80 + "\n")
            for r in calculated_rows:
                file.write(f"{r[0]:<6} | {r[1]:<18} | {r[2]:<18} | {r[3]:<18} | {r[4]:<18}\n")

        messagebox.showinfo("Успіх", "Результати збережено у файл!")
        status_var.set(f"Збережено: {file_path}")


btn_calc = ttk.Button(frame_left, text="Обчислити", command=calculate)
btn_calc.pack(fill="x", pady=(15, 5))

btn_save = ttk.Button(frame_left, text="Зберегти у файл...", command=save_to_file)
btn_save.pack(fill="x", pady=5)


initial_data = INTEGRALS[combo_int.get()]
draw_plot(initial_data["f"], initial_data["a"], initial_data["b"], initial_data["title"])

root.mainloop()