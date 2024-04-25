import pandas as pd
from sympy import Symbol, simplify
from pysr import PySRRegressor

from utils import laws_helper as laws_helper


eta = Symbol("eta")
nu = Symbol("nu")
psi = Symbol("psi")


class PYSR:
    def __init__(self, initial_df, all_found_symbols, verbose=False, return_print=False):
        self.initial_df = initial_df
        self.all_found_symbols = all_found_symbols
        self.symbols = list(self.initial_df)
        self.verbose = verbose
        self.return_print = return_print

    def run_iteration(self):
        data = [self.initial_df[col_name] for col_name in self.symbols]
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

        if self.verbose:
            print(f"PySR: Lowest level equation has {self.symbols[0]} = {eqn_rhs}")

        if self.return_print:
            return None, [self.symbols[0], eqn_rhs], "print"

        correct_equation_form = laws_helper.return_equation(eqn_rhs, self.symbols,
                                                            self.all_found_symbols,
                                                            self.verbose)

        return correct_equation_form.compute()
