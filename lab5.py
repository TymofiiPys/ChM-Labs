import sympy as sp
from sympy.plotting import plot
import matplotlib.pyplot as plt
from sympy.abc import x
import math


def do_plotting(function):
    # Побудова графіка підінтегральної функції

    plt.style.use('_mpl-gallery')
    plot_ = plot((function, (x, -50, 9)),
                 (function, (x, 9, 50)),
                 line_color="blue",
                 xlim=(-50, 50),
                 ylim=(-50, 50))


def comp_left_rect_integral(function, a, b, h):
    # Обчислення інтеграла методом лівих прямокутників з кроком h
    accumulate = 0.
    xi = a
    while xi < b:
        accumulate += h * function.subs(x, xi)
        xi += h
    return accumulate


def est_error(inth, inth2, p):
    # Оцінка похибкки
    numer = math.fabs(inth - inth2)
    denom = math.pow(2, p) - 1
    return numer / denom


def cia_stop_condition(inth, inth2, p, epsilon):
    # Умова припинення
    return est_error(inth, inth2, p) <= epsilon


def comp_integral_approx(function, a, b, epsilon, n):
    # Наближене обчислення, метод лівих прямокутників, правило Рунге

    # Порядок точності
    p = 1

    # Крок
    h = (b - a) / n

    print("n =", n)
    print("Крок h =", h)
    i = 1
    index = str(int(math.pow(2, i)))
    integral_h = comp_left_rect_integral(function, a, b, h)
    print("I h =", integral_h)
    integral_h_halved = comp_left_rect_integral(function, a, b, h / 2)
    print("I h/" + index + " = ", integral_h_halved)
    integral_error = est_error(integral_h, integral_h_halved, p)
    print("Оцінка похибки |I - I h/" + index + "| =", integral_error)
    while not cia_stop_condition(integral_h, integral_h_halved, p, epsilon):
        h = h / 2
        i += 1
        index = str(int(math.pow(2, i)))
        integral_h = integral_h_halved
        integral_h_halved = comp_left_rect_integral(function, a, b, h / 2)
        print("I h/" + index + " =", integral_h_halved)
        integral_error = est_error(integral_h, integral_h_halved, p)
        print("Оцінка похибки |I - I h/" + index + "| =", integral_error)

    return integral_h_halved


def main():
    # Ввід підінтегральної функції

    function_string = "1 / (9 - x)"

    function = sp.sympify(function_string)

    print("Підінтегральна функція f(x):")
    sp.pprint(function)

    do_plotting(function)

    # Межі інтегрування
    a = 4.
    b = 7.

    print()

    # Точний інтеграл

    integral_accurate = sp.integrate(function, (x, a, b))
    print("Точний інтеграл: ")
    sp.pprint(integral_accurate)

    # Наближене значення інтеграла:
    integral_approx = comp_integral_approx(function, a, b, epsilon=0.5, n=3)

    print("Наближене значення інтеграла: ")
    sp.pprint(integral_approx)


main()
