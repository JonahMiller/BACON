import numpy as np
from sympy import Symbol, lambdify, simplify
from sympy.parsing.sympy_parser import parse_expr
from itertools import product

import warnings
warnings.filterwarnings('ignore')


eta = Symbol("eta")
nu = Symbol("nu")


class BACON_6:
    def __init__(self, initial_df, all_found_symbols,
                 expression=None, unknowns=None,
                 step=2, N_threshold=2):
        self.init_symbols = list(initial_df)
        self.data = [initial_df[col_name] for col_name in self.init_symbols]
        self.symbols = [eta, nu]
        self.all_found_symbols = all_found_symbols

        self.step = step
        self.N_threshold = N_threshold

        self.X = self.data[0]
        self.Y = self.data[1]

        self.parse_expression(expression, unknowns)

    def parse_expression(self, expression, unknowns):
        if not expression or not unknowns:
            expression = "w*X + x + y/(X+z)"
            unknowns = ["w", "x", "y", "z"]
        self.expr = parse_expr(expression)
        self.unknowns = [Symbol(u) for u in unknowns]
        self.n = len(unknowns)

    def generate_vars(self):
        vals = [-self.step, 0, self.step]
        return list(product(vals, repeat=self.n))

    def update_states(self):
        new_add_states = self.generate_vars()
        n_states = []
        for state in self.states:
            new_states = [state + nas for nas in new_add_states]
            n_states.extend(new_states)
        self.states = n_states

    def calculate_approx(self):
        self.best = []
        state_dict = {}
        self.min = 0
        for idx, vars in enumerate(self.states):
            Y_ = self.expr.subs(dict(zip(self.unknowns, vars)))
            if Symbol("X") in Y_.free_symbols:
                f_Y = lambdify([Symbol("X")], Y_)
                r = np.corrcoef(self.Y, f_Y(self.X))[0, 1]
            else:
                r = np.corrcoef(self.Y, [float(Y_)]*len(self.X))[0, 1]
            if r not in state_dict:
                state_dict[r] = np.array(list(vars))
                self.sort_threshold(idx, r)
        self.states = [state_dict[r] for r in self.best]
        self.step = self.step/2

    def sort_threshold(self, idx, r):
        if idx < self.N_threshold:
            self.best.append((r))
            self.min = min(self.best)
        else:
            if r > self.min:
                self.best.remove(self.min)
                self.best.append(r)
                self.min = min(self.best)

    def main(self):
        self.states = self.generate_vars()
        while self.step > 0.000001:
            self.calculate_approx()
            self.update_states()
        self.N_threshold = 1
        self.calculate_approx()
        self.output_variables()

    def output_variables(self):
        Y_ = self.expr.subs(dict(zip(self.unknowns, self.states[0])))
        if Symbol("X") in Y_.free_symbols:
            f_Y = lambdify([Symbol("X")], Y_)
            m, d = np.polyfit(f_Y(self.X), self.Y, 1)
        else:
            m, d = np.polyfit([float(Y_)]*len(self.X), self.Y, 1)
        print(f"Y = {simplify(parse_expr(f'{m}*({Y_}) + {d}'))}")

    def new_symbol(self):
        '''
        Draws new variables to use in the case of linear relationships. Starts with
        "a" and draws onwards in the alphabet.
        '''
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for sym in self.all_found_symbols:
            try:
                idx = letters.index(str(sym))
                letters = letters[idx + 1:]
            except ValueError:
                continue
        letter = Symbol(letters[0])
        return letter
