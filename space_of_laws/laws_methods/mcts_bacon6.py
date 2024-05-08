import numpy as np
from sympy import Symbol, Eq, lambdify, simplify
from sympy.parsing.sympy_parser import parse_expr
from itertools import product

from utils import losses as loss_helper

import warnings
warnings.filterwarnings('ignore')


extra_symbols = ["m", "n", "o", "p", "q", "r", "s", "t", "u", "v"]


class BACON_6:
    def __init__(self, initial_df, eqns,
                 step=2, N_threshold=1, verbose=True):
        self.initial_df = initial_df
        self.symbols = list(initial_df)
        self.data = [initial_df[col_name] for col_name in self.symbols]

        self.step = step
        self.N_threshold = N_threshold
        self.verbose = verbose

        self.parse_expression(eqns)
        print(self.expr)

    def parse_expression(self, eqns):
        new_eqns = []
        unknowns = []
        for idx in range(len(eqns)):
            eqn = eqns[idx]
            new_symb = Symbol(extra_symbols[idx])
            new_eqns.append(Eq(eqn.lhs, new_symb))
            unknowns.append(new_symb)

        new_eqn = loss_helper.simplify_eqns(self.initial_df, new_eqns,
                                            self.symbols[-1], unknowns).iterate_through_dummys()
        self.expr = simplify(new_eqn.lhs)
        self.unknowns = unknowns
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
            try:
                eta_ = self.expr.subs(dict(zip(self.unknowns, vars)))
                fs = list(eta_.free_symbols)
                key_var = self.initial_df.iloc[:, -1]

                if fs:
                    data = [self.initial_df[sym].to_numpy().flatten() for sym in fs]
                    f_eta = lambdify([*fs], eta_)
                    f_eta_data = f_eta(*data)

                    if not all(np.isfinite(f_eta_data)):
                        raise KeyError

                    r = np.corrcoef(f_eta_data, key_var)[0, 1]

                    if r not in state_dict:
                        state_dict[r] = np.array(list(vars))
                        self.sort_threshold(idx, r)
            except KeyError:
                continue
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

    def output_variables(self):
        results_dict = dict(zip(self.unknowns, self.states[0]))
        eta_ = self.expr.subs(results_dict)
        fs = list(eta_.free_symbols)
        key_var = self.initial_df.iloc[:, -1]
        data = [self.initial_df[sym].to_numpy().flatten() for sym in fs]
        f_eta = lambdify([*fs], eta_)
        m, c = np.polyfit(f_eta(*data), key_var, 1)

        if abs(c) < 0.00001:
            c = 0
        expr = parse_expr(f"{m*eta_} + {c}")

        if self.verbose:
            final_expr = simplify(expr)
            print(f"BACON 6: Expression determined is {self.symbols[-1]} = {final_expr}")

        return expr

    def run_iteration(self):
        self.states = self.generate_vars()
        while self.step > 0.000001:
            self.calculate_approx()
            self.update_states()
        self.N_threshold = 1
        self.calculate_approx()
        best_expr = self.output_variables()
        return Eq(best_expr, self.symbols[-1])
