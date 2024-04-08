import pandas as pd
from sympy import Eq, Add, Symbol, Number, simplify, nsimplify, expand, solve

from pysr import PySRRegressor


def main(initial_df):
    print(initial_df)

    eta = Symbol("eta")

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

    X = pd.DataFrame({str(symbols[1]): data[1]})

    model.fit(X, y)
    eqn_rhs = simplify(model.sympy())

    eqn = Eq(eta, eqn_rhs)

    if len(eqn_rhs.free_symbols) == 0:
        return [float(eqn_rhs)], symbols[0], ""

    # if eqn_rhs.count(symbols[1]) != 1:
    #     print("Non-linear relation found, returning blank.")
    #     return None, None, None

    expanded_eqn, coeff = expand_form(eqn)

    return return_form(expanded_eqn, coeff, symbols[0])


def expand_form(eqn):
    zeta = Symbol("zeta")
    eqn_rhs = eqn.rhs
    eqn_lhs = eqn.lhs
    arg_list = list(Add.make_args(eqn_rhs))

    if len(arg_list) == 1:
        for term in list(arg_list[0].args):
            if isinstance(term, Number):
                break
            term = 1
        expr = eqn_rhs/term
        eqn_rhs = eqn_rhs.subs(expr, zeta)
        final_form = solve(Eq(eqn_lhs, eqn_rhs), term)[0]
        final_form = final_form.subs(zeta, expr)

    elif len(arg_list) == 2:
        eqn_rhs -= list(arg_list)[1]
        final_form = eqn_lhs - list(arg_list)[1]
        term = eqn_rhs

    expanded = expand(final_form)
    return expanded, term


def return_form(expanded_form, coeff, lhs):
    eta = Symbol("eta")
    arg_list = list(Add.make_args(expanded_form))
    expanded_form = expanded_form.subs(eta, lhs)
    if len(arg_list) == 1:
        return [coeff], nsimplify(simplify(expanded_form)), None

    elif len(arg_list) == 2:
        term = 1
        print(arg_list)
        for arg in arg_list:
            arg_ = nsimplify(arg)
            if any(isinstance(term, Number) for term in list(arg_.args)):
                var_arg = arg.subs(eta, lhs)
                for term in list(arg.args):
                    if isinstance(term, Number):
                        coeff = term
            else:
                fixed_arg = arg_.subs(eta, lhs)
        k = new_symbol(expanded_form)
        first, second, third = fixed_arg - nsimplify(k*(var_arg/coeff)), \
            fixed_arg, nsimplify(var_arg/coeff)
        lin_data = ["linear", k, first, -coeff, second, third]
        print([coeff], nsimplify(simplify(expanded_form)), lin_data)
        return [coeff], nsimplify(simplify(expanded_form)), lin_data


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
