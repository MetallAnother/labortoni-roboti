import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



def f(x, eq_type):
    if eq_type == "1. x + sqrt(x) + x^(1/3) + x^(1/4) = 5":
        return x + x**(1/2) + x**(1/3) + x**(1/4) - 5

    elif eq_type == "2. ln((1+x^2)/(1-x^2)) = 1-x":
        return math.log((1 + x**2) / (1 - x**2)) - 1 + x



def df(x, eq_type):
    if eq_type == "1. x + sqrt(x) + x^(1/3) + x^(1/4) = 5":
        return (
            1
            + 1 / (2 * x**(1/2))
            + 1 / (3 * x**(2/3))
            + 1 / (4 * x**(3/4))
        )

    elif eq_type == "2. ln((1+x^2)/(1-x^2)) = 1-x":
        return 4 * x / (1 - x**4) + 1



def phi(x, eq_type):
    if eq_type == "1. x + sqrt(x) + x^(1/3) + x^(1/4) = 5":
        return 5 - x**(1/2) - x**(1/3) - x**(1/4)

    elif eq_type == "2. ln((1+x^2)/(1-x^2)) = 1-x":

        e = math.exp(1 - x)
        return math.sqrt((e - 1) / (e + 1))


def dihtomiya(a, b, eps, eq_type):
    n = 0

    if f(a, eq_type) * f(b, eq_type) > 0:
        raise ValueError("No root on interval")

    while (b - a) / 2 > eps and n < 500:
        n += 1
        c = (a + b) / 2

        if f(a, eq_type) * f(c, eq_type) < 0:
            b = c
        else:
            a = c

    x = (a + b) / 2

    return x, n, f(x, eq_type)


def iteraciya(x0, eps, eq_type):
    x = x0
    n = 0

    while n < 500:
        n += 1

        x_next = phi(x, eq_type)

        if abs(x_next - x) < eps:
            return x_next, n, f(x_next, eq_type)

        x = x_next

    return x, n, f(x, eq_type)


def nuton(x0, eps, eq_type):
    x = x0
    n = 0

    while n < 500:
        n += 1

        derivative = df(x, eq_type)

        if abs(derivative) < 1e-12:
            raise ValueError("Derivative is zero")

        x_next = x - f(x, eq_type) / derivative

        if abs(x_next - x) < eps:
            return x_next, n, f(x_next, eq_type)

        x = x_next

    return x, n, f(x, eq_type)



root = tk.Tk()
root.title("Laboratorna 1")
root.geometry("900x650")

frame_left = ttk.Frame(root, padding=10)
frame_left.pack(side="left", fill="y")

frame_right = ttk.Frame(root, padding=10)
frame_right.pack(side="right", fill="both", expand=True)



lbl_eq = ttk.Label(frame_left, text="Select example:")
lbl_eq.pack(anchor="w")

combo_eq = ttk.Combobox(
    frame_left,
    values=[
        "1. x + sqrt(x) + x^(1/3) + x^(1/4) = 5",
        "2. ln((1+x^2)/(1-x^2)) = 1-x"
    ],
    state="readonly",
    width=40
)

combo_eq.current(0)
combo_eq.pack(fill="x", pady=(0, 10))



lbl_method = ttk.Label(frame_left, text="Select method:")
lbl_method.pack(anchor="w")

combo_method = ttk.Combobox(
    frame_left,
    values=[
        "All methods",
        "Dihtomiya",
        "Iteration",
        "Nuton"
    ],
    state="readonly"
)

combo_method.current(0)
combo_method.pack(fill="x", pady=(0, 10))



lb_a = ttk.Label(frame_left, text="a:")
lb_a.pack(anchor="w")

entry_a = ttk.Entry(frame_left)
entry_a.insert(0, "1.0")
entry_a.pack(fill="x")



lb_b = ttk.Label(frame_left, text="b:")
lb_b.pack(anchor="w")

entry_b = ttk.Entry(frame_left)
entry_b.insert(0, "2.0")
entry_b.pack(fill="x")



lb_x0 = ttk.Label(frame_left, text="x0:")
lb_x0.pack(anchor="w")

entry_x0 = ttk.Entry(frame_left)
entry_x0.insert(0, "1.5")
entry_x0.pack(fill="x")



fig = plt.figure(figsize=(4, 2.5), dpi=90)

ax = fig.add_subplot(1, 1, 1)

canvas = FigureCanvasTkAgg(fig, master=frame_right)

canvas.get_tk_widget().pack(
    fill="both",
    expand=True
)



cols = ("eps", "method", "n", "x", "fx")

table = ttk.Treeview(
    frame_right,
    columns=cols,
    show="headings",
    height=8
)

for col, text in zip(
    cols,
    ["ε", "Метод", "n", "x", "f(x)"]
):
    table.heading(col, text=text)
    table.column(
        col,
        width=90,
        anchor="center"
    )

table.pack(
    fill="both",
    expand=True
)


def draw_graph(points=None):
    eq = combo_eq.get()

    try:
        a = float(entry_a.get())
        b = float(entry_b.get())
    except ValueError:
        a = 0.1
        b = 2.0

    ax.clear()

    if eq == "2. ln((1+x^2)/(1-x^2)) = 1-x":
        left = max(a - 0.2, -0.99)
        right = min(b + 0.2, 0.99)
    else:
        left = max(a - 0.5, 0.001)
        right = b + 0.5

    x_vals = np.linspace(left, right, 200)
    y_vals = []

    for x in x_vals:
        try:
            y_vals.append(f(x, eq))
        except (ValueError, ZeroDivisionError):
            y_vals.append(np.nan)

    ax.plot(x_vals, y_vals, label="f(x)")
    ax.axhline(0, color="gray", linestyle="--")

    
    if points:
        for x, method in points:
            y = f(x, eq)

            ax.scatter(x, y, s=70, zorder=5)

            ax.annotate(
                f"{method}\nx = {x:.4f}",
                (x, y),
                xytext=(10, 15),
                textcoords="offset points",
                fontsize=8
            )

    ax.grid(True)
    ax.legend()
    canvas.draw()



def on_eq_change(event=None):
    eq = combo_eq.get()

    entry_a.delete(0, "end")
    entry_b.delete(0, "end")
    entry_x0.delete(0, "end")

    if eq == "1. x + sqrt(x) + x^(1/3) + x^(1/4) = 5":

        
        entry_a.insert(0, "1.0")
        entry_b.insert(0, "2.0")
        entry_x0.insert(0, "1.5")

    else:

        
        entry_a.insert(0, "0.1")
        entry_b.insert(0, "0.9")
        entry_x0.insert(0, "0.5")

    draw_graph()


combo_eq.bind(
    "<<ComboboxSelected>>",
    on_eq_change
)


def calculate():

    
    for item in table.get_children():
        table.delete(item)

    eq = combo_eq.get()
    method = combo_method.get()

    try:
        a = float(entry_a.get())
        b = float(entry_b.get())
        x0 = float(entry_x0.get())

    except ValueError:
        messagebox.showerror(
            "Помилка",
            "Введіть числові значення!"
        )
        return

    
    points = []

    for eps in [0.01, 0.001, 0.0001]:

        
        if method == "All methods" or method == "Dihtomiya":

            try:
                x, n, fx = dihtomiya(
                    a,
                    b,
                    eps,
                    eq
                )

                table.insert(
                    "",
                    "end",
                    values=(
                        eps,
                        "Dihtomiya",
                        n,
                        round(x, 5),
                        f"{fx:.2e}"
                    )
                )

                points.append((x, f"Dihtomiya ε={eps}"))

            except Exception:
                table.insert(
                    "",
                    "end",
                    values=(
                        eps,
                        "Dihtomiya",
                        "Err",
                        "-",
                        "-"
                    )
                )

        
        if method == "All methods" or method == "Iteration":

            try:
                x, n, fx = iteraciya(
                    x0,
                    eps,
                    eq
                )

                table.insert(
                    "",
                    "end",
                    values=(
                        eps,
                        "Iteration",
                        n,
                        round(x, 5),
                        f"{fx:.2e}"
                    )
                )

                points.append((x, f"Iteration ε={eps}"))

            except Exception:
                table.insert(
                    "",
                    "end",
                    values=(
                        eps,
                        "Iteration",
                        "Err",
                        "-",
                        "-"
                    )
                )

        
        if method == "All methods" or method == "Nuton":

            try:
                x, n, fx = nuton(
                    x0,
                    eps,
                    eq
                )

                table.insert(
                    "",
                    "end",
                    values=(
                        eps,
                        "Nuton",
                        n,
                        round(x, 5),
                        f"{fx:.2e}"
                    )
                )

                points.append((x, f"Nuton ε={eps}"))

            except Exception:
                table.insert(
                    "",
                    "end",
                    values=(
                        eps,
                        "Nuton",
                        "Err",
                        "-",
                        "-"
                    )
                )

    
    draw_graph(points)


def save_file():

    path = filedialog.asksaveasfilename(
        defaultextension=".txt"
    )

    if path:

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f_out:

            for row in table.get_children():

                f_out.write(
                    str(
                        table.item(row)["values"]
                    ) + "\n"
                )

        messagebox.showinfo(
            "Success",
            "Saved!"
        )



btn_calc = ttk.Button(
    frame_left,
    text="Calculate",
    command=calculate
)

btn_calc.pack(
    fill="x",
    pady=10
)


btn_save = ttk.Button(
    frame_left,
    text="Save in a file",
    command=save_file
)

btn_save.pack(fill="x")


draw_graph()

root.mainloop()