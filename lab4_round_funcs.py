from sympy import Number

def round_list(list):
    new_list = []
    for item in list:
        new_list.append(round(item, 4))
    return new_list


def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(Number)})