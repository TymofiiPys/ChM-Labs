import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy.plotting import plot
from sympy.abc import x
import lab4_round_funcs as rf

np.set_printoptions(precision=4, suppress=True)

# Зображувана функція f(x)
function_str = "x**3 - x - 1 - cos(x)"
func = sp.sympify(function_str)

# Отримання точних розв'язків рівняння f(x) = 0
f_zero = sp.nsolve(func, x, 1, dict=True)
# Формуємо список точок (x, 0) таких, що f(x) = 0
pointlist_f = list()
for sol in f_zero:
    point = {
        'args': [sol[x], 0],
        'color': "black",
        'ms': 5,
        'marker': "o"
    }
    pointlist_f.append(point)

print("Функція f(x): ")
sp.pprint(func)
print("\nРозв'язки рівняння f(x) = 0: ", f_zero)

a = float(input("\nВведіть ліву межу інтервалу (а): "))
b = float(input("Введіть праву межу інтервалу (b): "))
if b < a:
    print("Помилка: права межа має бути не меншою за ліву")
    exit(1)

sol_belongs_to_interv = False
for sol in f_zero:
    if sp.Interval(a, b).contains(sol[x]):
        sol_belongs_to_interv = True
        break

if not sol_belongs_to_interv:
    print("Помилка: жоден розв'язок не належить інтервалу [", a, ",", b, "]")
    exit(1)

# Число вузлів інтерполяції
m = 10
# Степінь інтерполяційного полінома
n = m - 1

# Нулі полінома Чебишова на інтервалі [a; b]
zeros_cheb = []
for k in range(n + 1):
    zeros_cheb.append(
        ((a + b) / 2)
        + ((b - a) / 2)
        * sp.cos((2 * k + 1) * sp.pi
                 / (2 * (n + 1))).evalf()
    )

for zero in zeros_cheb:
    point = {
        'args': [zero, func.subs(x, zero)],
        'color': "purple",
        'ms': 5,
        'marker': "o",
        'label': 'bib'
    }
    pointlist_f.append(point)

print("\nНулі полінома Чебишова на інтервалі [", a, ",", b, "]")
print(rf.round_list(zeros_cheb))

# Розділені різниці

# 0-й стовпець - f(x)
# 1 - n стовпці - розділені різниці відповідних порядків
differs = np.ndarray(shape=(n + 1, n + 1), dtype=float)

for i in range(n + 1):
    for j in range(n + 1):
        differs[i][j] = 0

for k in range(n + 1):
    all_zero = True
    for i in range(n + 1 - k):
        if k == 0:
            differs[i][k] = func.subs(x, zeros_cheb[i])
        else:
            differs[i][k] = ((differs[i + 1][k - 1] - differs[i][k - 1])
                             / (zeros_cheb[i + k] - zeros_cheb[i]))
        if abs(differs[i][k]) > np.power(10., -16):
            all_zero = False
    if all_zero:
        break

print("\n\nРозділені різниці (k-й стовпець - k-й порядок): ")
print(differs)

# Пряма інтерполяція

polynom_forward = 0.
for k in range(n + 1):
    term_k = differs[0][k]
    for i in range(k):
        term_k *= (x - zeros_cheb[i])
    polynom_forward += term_k

polynom_forward = sp.poly(polynom_forward).as_expr()
print("\n\nПрямий інтерполянт Ньютона:")
sp.pprint(rf.round_expr(polynom_forward, 8))

# Обернена інтерполяція

polynom_rev = 0.
for k in range(n + 1):
    term_k = differs[n - k][k]
    for i in range(k):
        term_k *= (x - zeros_cheb[n - i])
    polynom_rev += term_k

polynom_rev = sp.poly(polynom_rev).as_expr()
print("\n\nОбернений інтерполянт Ньютона:")
sp.pprint(rf.round_expr(polynom_rev, 8))

# Пошук розв'язків функцій Pf(x) = 0 і Pr(x) = 0,
# де Pf(x) - прямий, а
# Pr(x) - обернений інтерполянт Ньютона

p_forw_zero = sp.nsolve(polynom_forward, x, 1, dict=True)
# Формуємо список точок (x, 0) таких, що f(x) = 0
pointlist_p_forw = list()
max_p_forw = p_forw_zero[0][x]
for sol in p_forw_zero:
    point = {
        'args': [sol[x], 0],
        'color': "orange",
        'ms': 5,
        'marker': "o"
    }
    if sol[x] > max_p_forw:
        max_p_forw = sol[x]
    pointlist_p_forw.append(point)

p_rev_zero = sp.nsolve(polynom_forward, x, 1, dict=True)
# Формуємо список точок (x, 0) таких, що f(x) = 0
pointlist_p_rev = list()
max_p_rev = p_rev_zero[0][x]
for sol in p_rev_zero:
    point = {
        'args': [sol[x], 0],
        'color': "yellow",
        'ms': 5,
        'marker': "o"
    }
    if sol[x] > max_p_rev:
        max_p_rev = sol[x]
    pointlist_p_rev.append(point)

print("\nРозв'язки рівняння Pf(x) = 0: ", p_forw_zero)
print("Найбільший корінь рівняння Pf(x) = 0: ", max_p_forw)
print("\nРозв'язки рівняння Pr(x) = 0: ", p_rev_zero)
print("Найбільший корінь рівняння Pf(x) = 0: ", max_p_rev)

# Графіки функцій

plt.style.use('_mpl-gallery')
plot_poly = plot(func,
                 line_color="red",
                 label="Початкова функція",
                 legend=True,
                 xlim=(-50, 50), ylim=(-50, 50),
                 markers=pointlist_f,
                 show=False)
plot_poly.append(plot(polynom_forward,
                      line_color="green",
                      label="Прямий інтерполянт Ньютона",
                      markers=pointlist_p_forw,
                      show=False)[0])
plot_poly.append(plot(polynom_rev,
                      line_color="blue",
                      label="Обернений інтерполянт Ньютона",
                      markers=pointlist_p_rev,
                      show=False)[0])
plot_poly.show()
