import math
import numpy as np
import sympy as sp

# opt = int(input("Оберіть спосіб задання матриці A і вектор-стовпчика b:\n"
#                 "1) генерація таким чином, щоб можна було застосувати методи розв'язання СЛАР\n"
#                 "2) самостійне задання матриці з файлу matrix.csv (розмірність 5x5 і 5x1)"))

opt = 2
# Ініціалізація матриці A та вектор-стовпчика b
Ab = np.zeros(shape=(5, 6))
A = np.zeros(shape=(5, 5))
b = np.zeros(shape=(5, 1))
# if opt == 1:
#     print()
# if opt == 2:
#     Ab = np.loadtxt('matrix.csv', delimiter=';')
#     A = Ab[:, :5].copy()
#     b = Ab[:, 5:].copy()

filename = input("Введіть назву файлу, з якого зчитуються матриця A і b: ")
Ab = np.loadtxt(filename, delimiter=';')
A = Ab[:, :5].copy()
b = Ab[:, 5:].copy()
print("\nМатриця А:")
sp.pprint(A)
print("Вектор-стовпчик b:")
sp.pprint(b)
print()

np.set_printoptions(precision=8, suppress=True)

if A.shape[0] != A.shape[1]:
    print("Матриця А не є квадратною")
    exit(1)

det = np.linalg.det(A)
if det == 0:
    print("Визначник матриці A дорівнює 0")
    exit(1)
else:
    print("Визначник матриці A =", math.trunc(det))

A_inv = np.linalg.inv(A)
print("\nОбернена матриця до А:")
sp.pprint(A_inv)
A_norm = np.linalg.norm(A, np.inf)
print("Норма матриці А: \n", A_norm)
A_inv_norm = np.linalg.norm(A_inv, np.inf)
print("Норма оберненої матриці до А: \n", A_inv_norm)
cond_A = A_inv_norm * A_norm
print("Число обумовленості матриці А =", cond_A)

n = A.shape[0]


# Метод Гауса з вибором головного за всією матрицею

def Gaussian():
    print("\nМетод Гауса з вибором головного за всією матрицею")
    # Розв'язок
    vars = np.arange(0, n, dtype=int)
    Ab_interm = Ab.copy()
    l = 0
    max_list = []
    for k in range(n):
        # Вибір максимального за модулем елемента
        print("\n Ітерація", k)
        print("A" + str(k) + "розшир.:\n", Ab_interm)
        max = -1
        imax = -1
        jmax = -1
        for i in range(k, n):
            for j in range(k, n):
                if np.abs(Ab_interm[i][j]) > max:
                    max = np.abs(Ab_interm[i][j])
                    imax = i
                    jmax = j
        print("Максимальний за модулем елемент:", np.round(Ab_interm[imax][jmax], 4), ", індекси: (", imax, ",", jmax, ")")
        max_list.append(max)
        # Перестановка
        P1 = np.eye(n, n, dtype=int)
        P2 = np.eye(n, n, dtype=int)
        P1[:, [k, imax]] = P1[:, [imax, k]]
        P2[:, [k, jmax]] = P2[:, [jmax, k]]
        if not k == imax:
            l += 1
        if not k == jmax:
            l += 1
        print("P" + str(k) + "1:\n", P1)
        print("P" + str(k) + "2:\n", P2)
        Ab_interm = P1.dot(Ab_interm)
        # tempA = Ab_interm[:, :3].copy().dot(P2)
        # tempB = Ab_interm[:, 3:].copy()
        Ab_interm[:, :n] = Ab_interm[:, :n].dot(P2)
        vars = vars.dot(P1)
        print("P" + str(k) + "1A" + str(k) + "розшир.P" + str(k) + "2:\n", Ab_interm)
        # Ab_interm = np.concatenate((tempA, tempB), axis=1)
        M = np.eye(n, n, dtype=float)
        M[k][k] = 1 / Ab_interm[k][k]
        for i in range(k + 1, n):
            M[i][k] = -Ab_interm[i][k] / Ab_interm[k][k]
        print("M" + str(k) + ":\n", M)
        Ab_interm = M.dot(Ab_interm)
    print("Aрозшир.:\n", Ab_interm)
    i = n - 1
    x = np.empty((n, 1), dtype=float)
    while (i >= 0):
        x[vars[i]] = Ab_interm[i][n]
        for j in range(i + 1, n):
            x[vars[i]] -= Ab_interm[i][j] * x[vars[j]]
        i -= 1
    print("\nРозв'язок: x = ", x)

    det = math.pow(-1, l)
    for max in max_list:
        det *= max
    print("\nВизначник = ", det)

Gaussian()

# Метод Зейделя

def Seidel():
    # Ab = np.loadtxt('matrix2.csv', delimiter=';')
    # A = Ab[:, :3].copy()
    # b = Ab[:, 3:].copy()
    print("\nМетод Зейделя")
    # Достатня умова збіжності 1
    cond_1 = True
    for i in range(n):
        sum = 0.
        j = 0
        for j in range(n):
            if j == i:
                continue
            sum += np.abs(A[i, j])
            j += 1
        if np.abs(A[i, i]) < sum:
            cond_1 = False
            break
        i += 1
    if not cond_1:
        print("Достатня умова збіжності |A(i,i)| >= sum(|A(i,j)|), j = 1 && j != i) не виконується")
        exit(1)
    # Достатня умова збіжності 2
    if not np.array_equal(A, A.transpose()):
        print("Матриця А не є симетричною - пошук мінімального власного значення степеневим методом неможливий")
        exit(1)
    if not np.all(np.linalg.eigvals(A)) > 0:
        print("Матриця А не є додатно визначеною - пошук мінімального власного значення степеневим методом неможливий")
        exit(1)
    # Необхідна і достатня умова збіжності
    lambdaA = np.empty(shape=(5, 5))
    A_sp = sp.Matrix(A)
    l_symb = sp.symbols("l")

    def lAset(i, j):
        if j <= i:
            return A_sp[i, j] * l_symb
        else:
            return A_sp[i, j]

    lambdaA = sp.Matrix(n, n, lAset)
    lamb_set = sp.solveset(lambdaA.det(), l_symb, domain=sp.S.Reals)
    # print(ll)
    cond_3 = True
    for lamb in lamb_set:
        if np.abs(lamb) > 1:
            cond_3 = False
            break
    if not cond_3:
        print("Необхідна і достатня умова збіжності |lambda| < 1 не виконується")

    epsilon = float(input("Введіть точність epsilon обчисленння: "))
    # epsilon = 0.5
    print("Введіть поелементно початкове наближення х0:")
    x0_list = []
    for i in range(n):
        x0_list.append(float(input("Елемент " + str(i) + ": ")))
    x0 = np.fromiter(x0_list, dtype=np.dtype(float, 1))
    x0 = np.reshape(x0, (n, 1))

    Ab_interm = Ab.copy()
    x_pred = x0.copy()
    x_cur = np.empty(shape=(n, 1))
    for i in range(n):
        sum_1 = 0.
        sum_2 = 0.
        for j in range(i):
            sum_1 += Ab_interm[i][j] * x_cur[j] / Ab_interm[i][i]
        for j in range(i + 1, n):
            sum_2 += Ab_interm[i][j] * x_pred[j] / Ab_interm[i][i]
        x_cur[i] = - sum_1 - sum_2 + (Ab_interm[i][n] / Ab_interm[i][i])

    print("x0: ", x_pred)
    print("x1: ", x_cur)
    print("Норма х0: ", np.linalg.norm(x_pred, np.inf))
    print("Норма х1: ", np.linalg.norm(x_cur, np.inf))

    step = 2
    print()
    while np.linalg.norm(x_cur - x_pred, np.inf) > epsilon:
        x_pred = x_cur.copy()
        x_cur = np.empty(shape=(n, 1))
        for i in range(n):
            sum_1 = 0.
            sum_2 = 0.
            for j in range(i):
                sum_1 += Ab_interm[i][j] * x_cur[j] / Ab_interm[i][i]
            for j in range(i + 1, n):
                sum_2 += Ab_interm[i][j] * x_pred[j] / Ab_interm[i][i]
            x_cur[i] = - sum_1 - sum_2 + (Ab_interm[i][n] / Ab_interm[i][i])
        print("x" + str(step) + ": ", x_cur)

    print("Норма х" + str(step - 1) + ": ", np.linalg.norm(x_pred, np.inf))
    print("Норма х" + str(step) + ": ", np.linalg.norm(x_cur, np.inf))


Seidel()
