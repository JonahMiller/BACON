import pandas as pd
from sympy import Eq, Add, Symbol, Number, simplify, nsimplify, expand, solve, fraction

from pysr import PySRRegressor


eta = Symbol("eta")
nu = Symbol("nu")


def main(initial_df):
    print(initial_df)

    symbols = list(initial_df)
    data = [initial_df[col_name] for col_name in symbols]

    y = data[0]

    model = PySRRegressor(
        niterations=15,
        maxsize=7,
        binary_operators=["+", "*", "/", "-"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="accuracy",
        temp_equation_file=True,
        delete_tempfiles=True
    )

    X = pd.DataFrame({str(nu): data[1]})

    model.fit(X, y)
    eqn_rhs = simplify(model.sympy())

    eqn = Eq(eta, eqn_rhs)

    rhs_symb, terms = eqn_rhs.count(nu), len(Add.make_args(eqn_rhs))

    if rhs_symb == 0:
        return [float(eqn_rhs.subs(nu, symbols[1]))], symbols[0], ""

    if rhs_symb == 1:
        if terms == 1:
            return single_product_division(eqn, symbols)
        elif terms == 2:
            return simple_linear(eqn, symbols)
        else:
            print("Unable to deal with non-linear relationship, returning blank")
            return None, None, None

    elif rhs_symb == 2:
        if terms == 1:
            return complex_linear_1_term(eqn, symbols)
        elif terms == 2:
            return complex_linear_2_term(eqn, symbols)
        else:
            print("Unable to deal with non-linear relationship, returning blank")
            return None, None, None

    else:
        print("Unable to deal with non-linear relationship, returning blank")
        return None, None, None


def single_product_division(eqn, symbols):
    arg_list = list(eqn.rhs.args)
    const = 1
    for arg in arg_list:
        if isinstance(arg, Number):
            const = arg
    print(eqn, const, solve(eqn, const))
    expr = solve(eqn, const)[0]
    expr = subs_expr(eqn.lhs, symbols)
    return [const], expr, None


def simple_linear(eqn, symbols):
    const = 1
    for arg in list(Add.make_args(eqn.rhs)):
        if isinstance(arg, Number):
            const = arg
    expr = solve(eqn, const)[0]
    coeff, var1, var2 = linear_term_coeff(expr, symbols)
    k = new_symbol(var1 + var2)
    lin_data = ["linear", k, var1 - nsimplify(k*var2), -coeff, var1, var2, symbols]
    return [coeff], subs_expr(expr, symbols), lin_data


def complex_linear_1_term(eqn, symbols):
    num, den = fraction(eqn.rhs)
    assert len(Add.make_args(num)) == 1
    top_coeff = 1
    for arg in list(num.args):
        if isinstance(arg, Number):
            top_coeff = arg
    den = den/top_coeff
    for arg in list(den.args):
        if isinstance(arg, Number):
            const = arg
    print(eqn.lhs, nsimplify(eqn.rhs/top_coeff), const)
    expr = solve(Eq(eqn.lhs, nsimplify(eqn.rhs/top_coeff)), const)[0]
    coeff, var1, var2 = linear_term_coeff(expr, symbols)
    k = new_symbol(var1 + var2)
    lin_data = ["linear", k, var1 - nsimplify(k*var2), -coeff, var1, var2, symbols]
    return [coeff], subs_expr(expr, symbols), lin_data


def complex_linear_2_term(eqn, symbols):
    raise Exception


def linear_term_coeff(expr, symbols):
    arg_list = Add.make_args(expr)
    assert len(arg_list) == 2
    var2 = arg_list[1]
    for arg in arg_list:
        if any(isinstance(term, Number) for term in list(arg.args)):
            for term in arg.args:
                if isinstance(term, Number):
                    coeff = term
                else:
                    var2 = term
        else:
            var1 = arg
            coeff = 1
    return coeff, subs_expr(var1, symbols), subs_expr(var2, symbols)


def subs_expr(expr, symbols):
    e1 = expr.subs(eta, symbols[0])
    e2 = e1.subs(nu, symbols[1])
    return e2

def new_symbol(symbols):
    '''
    Draws new variables to use in the case of linear relationships. Starts with
    "a" and draws onwards in the alphabet.
    '''
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    used_symbols = list(symbols.free_symbols)
    for sym in used_symbols:
        try:
            idx = letters.index(str(sym))
            letters = letters[idx + 1:]
        except ValueError:
            continue
    letter = Symbol(letters[0])
    return letter
