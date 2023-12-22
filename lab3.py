import numpy as np

# Ініціалізація матриці A та вектор-стовпчика b

filename = input("Введіть назву файлу, з якого зчитуються матриця A і b: ")
Ab = np.loadtxt(filename, delimiter=';')
A = Ab[:, :5].copy()
print("\nМатриця А:")
print()
n = A.shape[0] # shape[0] та shape[1] - кількість рядків/стопчиків відповідно

if n != A.shape[1]:
    print("Матриця не є квадратною!")
    exit(1)

np.set_printoptions(precision=8, suppress=True)

# Степеневий метод:

print("Степеневий метод")

# Максимальне значення

epsilon = float(input("Введіть точність ε обчисленння: "))
# epsilon = 0.5
print("Введіть поелементно початкове наближення х0: ")
x0_list = []
x0_is_nullvect = True
for i in range(n):
    x0_list.append(float(input("Елемент " + str(i) + ": ")))
    if x0_list[i] != 0.:
        x0_is_nullvect = False
if x0_is_nullvect:
    print("Введено нуль-вектор. Метод не буде виконано.")
    exit(1)
x0 = np.fromiter(x0_list, dtype=np.dtype(float, 1))
x0 = np.reshape(x0, (n, 1))

m = int(input("Який номер компоненти векторів x буде застосовано у методі? "))
while m < 1 or m > n:
    m = int(input("Який номер компоненти векторів x буде застосовано у методі? "))

m = m - 1


# Ітераційний процес
def iter_pow_it(matr, m, epsilon, x0, matr_name):
    print("x0:", x0)
    x1 = matr.dot(x0).copy()
    print("x1:", x1)
    prev_lambda = x1[m] / x0[m]
    print("λ1:", prev_lambda)
    x1 = x1 / np.linalg.norm(x1)
    print("Нормований x1:", x1)
    x2 = matr.dot(x1).copy()
    print()
    print("x2:", x2)
    cur_lambda = x2[m] / x1[m]
    print("λ2:", cur_lambda)
    cur_x = x2
    step = 2
    while np.abs(cur_lambda - prev_lambda) > epsilon:
        cur_x = cur_x / np.linalg.norm(cur_x)
        print("Нормований x" + str(step) + ":", cur_x)
        prev_x = cur_x
        step += 1
        cur_x = matr.dot(prev_x).copy()
        print()
        print("x" + str(step) + ":", cur_x)
        prev_lambda = cur_lambda
        cur_lambda = cur_x[m] / prev_x[m]
        print("λ" + str(step) + ":", cur_lambda)

    print("|λ" + str(step) + " - λ" + str(step - 1) + "| <= ε =", epsilon)

    print("Максимальне власне значення матриці", matr_name, ":", cur_lambda)

    return cur_lambda


lambda_A = iter_pow_it(A, m, epsilon, x0, "A")

# Мінімальне власне значення
print()
if not np.array_equal(A, A.transpose()):
    print("Матриця А не є симетричною - не можна знайти мінімальне власне значення")
    exit(1)

def is_positive_def(matr):
    n = matr.shape[0]
    for i in range(1, n + 1):
        # Обчислення кутових мінорів
        submatr = matr[:i, :i].copy()
        smdet = np.linalg.det(submatr)
        if smdet < 0:
            return False
    return True


if not is_positive_def(A):
    print("Матриця А не є додатно визначеною - не можна знайти мінімальне власне значення")
    exit(1)

# Матриця B

B = lambda_A * np.eye(n, n) - A

print("Матриця B:", B)

lambda_B = iter_pow_it(B, m, epsilon, x0, "B")

print()
print("Мінімальне власне значення матриці А:", (lambda_A - lambda_B))
