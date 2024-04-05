import pandas as pd
from sympy import Eq, Add, Symbol, solve, Pow, simplify

from pysr import PySRRegressor


def main(initial_df):
    for col_name in initial_df.columns.tolist():
        if len(col_name.free_symbols) > 1:
            temp_df = initial_df.rename(columns={col_name: Symbol("eta")})
            save_expr = col_name
            subst = True
        else:
            temp_df = initial_df
            subst = False

    symbols = list(temp_df)
    symbols.reverse()
    data = [temp_df[col_name] for col_name in symbols]

    y = data[-1]

    model = PySRRegressor(
        niterations=5,
        maxsize=7,
        binary_operators=["+", "*", "/", "-"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="accuracy",
        temp_equation_file=True,
        delete_tempfiles=True
    )

    X = pd.DataFrame({str(v): d for v, d in zip(symbols[:-1], data[:-1])})

    model.fit(X, y)
    eqn_rhs = simplify(model.sympy())

    symbols = list(initial_df)
    symbols.reverse()

    eqn = Eq(symbols[1], eqn_rhs)

    if len(Add.make_args(eqn_rhs)) == 2:
        k = new_symbol(symbols)
        lin_data = ["linear", k, symbols[1] - k*symbols[0], 1, symbols[1], symbols[0]]
        symb = symbols[1] - k*symbols[0]
        return "", symb, lin_data

    elif len(Add.make_args(eqn_rhs)) == 1:
        if len(eqn_rhs.atoms(Pow)) == 0:
            var = symbols[0]
        else:
            var = list(eqn_rhs.atoms(Pow))[0]
        coeff = eqn_rhs.coeff(var)

        symb = solve(eqn, coeff)[0]

        if subst:
            symb = symb.subs(Symbol("eta"), save_expr)

        return [coeff], symb, ""
    else:
        raise Exception("Not equipped to deal with non-linear relations")


def new_symbol(symbols):
    '''
    Draws new variables to use in the case of linear relationships. Starts with
    "a" and draws onwards in the alphabet.
    '''
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    used_symbols = sum(sym for sym in symbols).free_symbols
    for sym in used_symbols:
        try:
            idx = letters.index(str(sym))
            letters = letters[idx + 1:]
        except ValueError:
            continue
    letter = Symbol(letters[0])
    return letter
