import matplotlib.pyplot as plt
import sympy as sp
import math
from sympy.plotting import plot
from sympy.abc import x

# Зображувана функція f(x)
functionstr = "x**4 + x**3 - 6 * x**2 + 20 * x - 16"
# Похибка
epsilon = 10 ** (-4)

func = sp.sympify(functionstr)
# Отримання точних розв'язків рівняння f(x) = 0
fzero = sp.solveset(func, x, domain=sp.S.Reals)
# Формуємо список точок (x, 0) таких, що f(x) = 0
sollist = list(fzero)
pointlist = list()
for sol in sollist:
    point = {
        'args': [sol, 0],
        'color': "red",
        'ms': 5,
        'marker': "o"
    }
    pointlist.append(point)

# Графік функції разом із точками x: f(x) = 0
plt.style.use('_mpl-gallery')
p1 = plot(func, xlim=(-50, 50), ylim=(-50, 50),
          markers=pointlist, show=False)
p1.show()

print("Функція f(x): ")
sp.pprint(func)
print("\nРозв'язки рівняння f(x) = 0: ", fzero)

a = float(input("\nВведіть ліву межу інтервалу (а): "))
b = float(input("Введіть праву межу інтервалу (b): "))
if b < a:
    print("Помилка: права межа має бути не меншою за ліву")
    exit(1)

sol_belongs_to_interv = False
for sol in sollist:
    if sp.Interval(a, b).contains(sol):
        sol_belongs_to_interv = True
        break

if not sol_belongs_to_interv:
    print("Помилка: жоден розв'язок не належить інтервалу (", a, ",", b, ")")
    exit(1)

x0 = float(input("Введіть початкове наближення (x0): "))
if x0 < a or x0 > b:
    print("Помилка: початкове наближення не належить інтервалу")
    exit(1)



# Метод простої ітерації

print("\nМетод простої ітерації:")

# Умова припинення ітераційного процесу
def fpi_stop_cond(current_x, prev_x, q, epsilon):
    if q < 0.5 and sp.Abs(current_x - prev_x) <= (1 - q) * epsilon / q:
        return False
    elif sp.Abs(current_x - prev_x) <= epsilon:
        return False
    return True


def fixed_point_iter(func, x, a, b, x0, sol):
    #Пошук дельти: |x - x0| <= delta
    g = x - x0
    gmin = sp.Abs(sp.minimum(g, x, sp.Interval(a,b)))
    gmax = sp.Abs(sp.maximum(g, x, sp.Interval(a,b)))
    if gmax < gmin:
        temp = gmin
        gmin = gmax
        gmax = temp
    delta = gmax
    print("delta =", delta)

    # Список можливих фі-функцій
    possible_phi_funcs = list()
    possible_phi_funcs.append(sp.sympify("1/20 * (- x**4 - x**3 + 6 * x**2 + 16)"))
    possible_phi_funcs.append(sp.sympify("1/6 * (x**3 + x**2 + 20 - (16/x))"))
    possible_phi_funcs.append(sp.sympify("1/6 * (x**4 + x**3 + 20 * x - 16)**(1/2)"))
    possible_phi_funcs.append(sp.sympify("(- x**4 + 6 * x**2 - 20 * x + 16)**(1/3)"))
    possible_phi_funcs.append(sp.sympify("(- x**3 + 6 * x**2 - 20 * x + 16)**(1/4)"))

    # Прохід по списку фі-функцій
    for phi_func in possible_phi_funcs:
        print("Фі-функція: ")
        sp.pprint(phi_func)
        # Перевірка фі-функції та похідної до неї на неперервність та знакосталість
        if (sp.calculus.util.continuous_domain(phi_func, x, sp.Interval(a, b))
            != sp.Interval(a, b)):
            print("Фі-функція", phi_func, "не є неперервною на проміжку (", a,",", b, ") - перехід до наступної")
            continue
        if (sp.calculus.util.continuous_domain(phi_func.diff(), x, sp.Interval(a, b))
            != sp.Interval(a, b)):
            print("Похідна від фі-функції", phi_func, "не є неперервною на проміжку (", a, ",", b, ") - перехід до наступної")
            continue
        try:
            if (not (sp.maximum(phi_func, x, sp.Interval(a, b))
                 * sp.minimum(phi_func, x, sp.Interval(a, b)) > 0)):
                print("Фі-функція не є знакосталою на інтервалі (", a, ", ", b,
                  ") - перехід до наступної")
                continue
        except:
            print("Помилка обчислення мінімуму/максимуму - перехід до наступної фі-функції")
            continue

        # Умови збіжності
        S = sp.Interval(x0 - delta, x0 + delta)
        # Визначення q
        qmin = sp.minimum(phi_func.diff(), x, S)
        q = sp.maximum(phi_func.diff(), x, S)
        if sp.Abs(qmin) > sp.Abs(q):
            # Якщо модуль мінімуму функції є більшим за модуль максимуму цієї ж функції,
            # то перший є максимумом модуля від функції (на заданому інтервалі)
            q = sp.Abs(qmin)
        print("q =", q.evalf())

        # Умова (1)
        if q >= 1:
            print("q >= 1 - перехід до наступної фі-функції")
            continue

        # Умова (2)
        if not sp.Abs(phi_func.subs(x, x0) - x0) <= (1 - q) * delta:
            print("|фі(x0) - x0| > (1-q)*delta - перехід до наступної фі-функції")
            continue

        # Обчислення апріорної оцінки
        apr_est = math.trunc((sp.ln((sp.Abs(phi_func.subs(x, x0) - x0) /
                                ((1 - q) * epsilon))) / sp.ln(1 / q)).evalf(1)) + 1
        print("Апріорна оцінка: n >=", apr_est)

        #Ітераційний процес
        current_x = phi_func.subs(x, x0)
        prev_x = x0
        n = 1
        print("x0 =", x0)
        print("x1 =", current_x)
        while fpi_stop_cond(current_x, prev_x, q, epsilon):
            prev_x = current_x
            n += 1
            current_x = phi_func.subs(x, current_x)
            print(f"x{n} =", current_x)
        print("Апостеріорна оцінка: n =", n)
        return
    print("Жодна фі-функція не задовольнила достатнім умовам збіжності")


# fixed_point_iter(func, x, a, b, x0, sollist[0])

# Метод релаксації
print("\nМетод релаксації:")


def relaxation(func, x, a, b, x0, sol):
    firstprime = func.diff()

    # Достатні умови збіжності
    # Обчислення m1, M1, tao та q
    m1 = sp.Abs(sp.minimum(firstprime, x, sp.Interval(a, b)))
    M1 = sp.Abs(sp.maximum(firstprime, x, sp.Interval(a, b)))
    if M1 < m1:
        # Якщо модуль максимуму функції є меншим за модуль мінімуму цієї ж функції,
        # то перший є мінімумом модуля від функції (на заданому інтервалі)
        temp = m1
        m1 = M1
        M1 = temp

    print("m1 =", m1.evalf())
    print("M1 =", M1.evalf())

    if m1 == 0:
        print("m1 = 0 - достатня умова збіжності не виконується")
    if m1 == M1:
        print("m1 = M1 - достатня умова збіжності не виконується")

    tao = 2 / (M1 + m1)
    print("tao =", tao.evalf())

    q = (M1 - m1) / (M1 + m1)
    print("q =", q.evalf())

    # Обчислення апріорної оцінки
    apr_est = math.trunc((sp.ln((sp.Abs(x0 - sol) / epsilon))
                     / sp.ln(1 / q)).evalf(1)) + 1
    print("Апріорна оцінка: n >=", apr_est)

    # Визначення знаку для ітераційного процесу:
    if sp.is_strictly_increasing(func, sp.Interval(a, b)):
        # f'(x) > 0
        tao = -1 * tao

    # Ітераційний процес
    current_x = x0 + tao * func.subs(x, x0)
    prev_x = x0
    n = 1
    print("x0 =", x0)
    print("x1 =", current_x)
    while sp.Abs(current_x - prev_x) > epsilon:
        prev_x = current_x
        n += 1
        current_x = prev_x + tao * func.subs(x, prev_x)
        print(f"x{n} =", current_x.evalf())
    print("Апостеріорна оцінка: n =", n)


relaxation(func, x, a, b, x0, sollist[1])

# Модифікований метод Ньютона
print("\nМодифікований метод Ньютона:")


def mod_newton(func, x, a, b, x0):
    # Достатні умови збіжності
    # Обчислення першої та другої похідної
    firstprime = func.diff()
    secondprime = firstprime.diff()
    if (sp.calculus.util.continuous_domain(func, x, sp.Interval(a, b))
            != sp.Interval(a, b)):
        print("Функція не є неперервною на інтервалі (", a, ",", b,
              ") - достатня умова збіжності не виконуються")
        return
    if (sp.calculus.util.continuous_domain(firstprime, x, sp.Interval(a, b))
            != sp.Interval(a, b)):
        print("Перша похідна не є неперервною на інтервалі (", a, ",", b,
              ") - достатня умова збіжності не виконуються")
        print("Перша похідна: ")
        sp.pprint(firstprime)
        return
    if (sp.calculus.util.continuous_domain(secondprime, x, sp.Interval(a, b))
            != sp.Interval(a, b)):
        print("Друга похідна не є неперервною на інтервалі (", a, ",", b,
              ") - достатня умова збіжності не виконуються")
        print("Друга похідна: ")
        sp.pprint(secondprime)
        return
    if (not (sp.maximum(firstprime, x, sp.Interval(a, b))
             * sp.minimum(firstprime, x, sp.Interval(a, b)) > 0)):
        print("Перша похідна не є знакосталою на інтервалі (", a, ", ", b,
              ") - достатня умова збіжності не виконуються")
        print("Перша похідна: ")
        sp.pprint(firstprime)
        return
    if (not (sp.maximum(secondprime, x, sp.Interval(a, b))
             * sp.minimum(secondprime, x, sp.Interval(a, b)) > 0)):
        print("Друга похідна не є знакосталою на інтервалі (", a, ",", b,
              ") - достатня умова збіжності не виконуються")
        print("Друга похідна: ")
        sp.pprint(secondprime)
        return
    if (sp.solveset(firstprime != 0, x, sp.Interval(a, b))
            != sp.Interval(a, b)):
        print("Не виконується f'(x) ≠ 0 на інтервалі (", a, ",", b,
              ") - достатня умова збіжності не виконуються")
        print("Перша похідна: ")
        sp.pprint(firstprime)
        return
    if not func.subs(x, x0) * secondprime.subs(x, x0) > 0:
        print("Не виконується f(x0) * f''(x0) > 0  - достатня умова збіжності не виконуються")
        print("Друга похідна: ")
        sp.pprint(secondprime)
        return

    # Ітераційний процес
    current_x = x0 - func.subs(x, x0) / firstprime.subs(x, x0)
    prev_x = x0
    n = 1
    print("x0 =", x0)
    print("x1 =", current_x)
    while sp.Abs(current_x - prev_x) > epsilon:
        prev_x = current_x
        n += 1
        current_x = current_x - func.subs(x, current_x) / firstprime.subs(x, x0)
        print(f"x{n} =", current_x.evalf())
    print("Апостеріорна оцінка: n =", n)


mod_newton(func, x, a, b, x0)
