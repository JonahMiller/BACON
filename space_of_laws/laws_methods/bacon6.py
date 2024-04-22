import numpy as np
from sympy import Symbol, lambdify, simplify
from sympy.parsing.sympy_parser import parse_expr
from itertools import product

from utils import laws_helper as laws_helper

import warnings
warnings.filterwarnings('ignore')


eta = Symbol("eta")
nu = Symbol("nu")


class BACON_6:
    def __init__(self, initial_df, all_found_symbols,
                 expression=None, unknowns=None,
                 step=2, N_threshold=1):
        print(initial_df)
        self.symbols = list(initial_df)
        self.data = [initial_df[col_name] for col_name in self.symbols]
        self.all_found_symbols = all_found_symbols

        self.step = step
        self.N_threshold = N_threshold

        self.eta = self.data[0]
        self.nu = self.data[1]

        self.parse_expression(expression, unknowns)

    def parse_expression(self, expression, unknowns):
        if not expression or not unknowns:
            expression = "x*nu + y/(nu + z)"
            unknowns = ["x", "y", "z"]
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
            eta_ = self.expr.subs(dict(zip(self.unknowns, vars)))
            fs = eta_.free_symbols

            if nu in fs and eta not in fs:
                f_eta = lambdify([nu], eta_)
                r = np.corrcoef(self.eta, f_eta(self.nu))[0, 1]
            elif nu not in fs and eta in fs:
                f_eta = lambdify([eta], eta_)
                r = np.corrcoef(self.eta, f_eta(self.eta))[0, 1]
            elif nu in fs and eta in fs:
                f_eta = lambdify([nu, eta], eta_)
                r = np.corrcoef(self.eta, f_eta(self.nu, self.eta))[0, 1]
            else:
                r = np.corrcoef(self.eta, [float(eta_)]*len(self.nu))[0, 1]

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

    def output_variables(self):
        results_dict = dict(zip(self.unknowns, self.states[0]))

        eta_ = self.expr.subs(results_dict)
        fs = eta_.free_symbols

        if nu in fs and eta not in fs:
            f_eta = lambdify([nu], eta_)
            m, d = np.polyfit(f_eta(self.nu), self.eta, 1)
        elif nu not in fs and eta in fs:
            f_eta = lambdify([eta], eta_)
            m, d = np.polyfit(f_eta(self.eta), self.eta, 1)
        elif nu in fs and eta in fs:
            f_eta = lambdify([nu, eta], eta_)
            m, d = np.polyfit(f_eta(self.nu, self.eta), self.eta, 1)
        else:
            m, d = np.polyfit([float(eta_)]*len(self.nu), self.eta, 1)

        if abs(d) < 0.00001:
            d = 0
        print(self.states[0])
        print(m, d, eta_)

        expr = parse_expr(f"{m}*{eta_} + {d}")
        print(f"{self.symbols[1]} = {simplify(expr.subs(nu, self.symbols[0]).subs(eta, self.symbols[1]))}")
        return expr

    def run_iteration(self):
        self.states = self.generate_vars()
        while self.step > 0.000001:
            self.calculate_approx()
            self.update_states()
        self.N_threshold = 1
        self.calculate_approx()
        eqn_rhs = self.output_variables()
        correct_equation_form = laws_helper.return_equation(eqn_rhs, self.symbols, self.all_found_symbols)
        return correct_equation_form.compute()
