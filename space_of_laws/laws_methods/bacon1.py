import numpy as np
from statistics import fmean
from sympy import Symbol, simplify
from scipy.stats import linregress as lr

from utils import laws_helper as laws_helper

import warnings
warnings.filterwarnings('ignore')


eta = Symbol("eta")
nu = Symbol("nu")
dummies = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']


class BACON_1:
    def __init__(self, initial_df, all_found_symbols,
                 epsilon=0.001, delta=0.1,
                 epsilon_scale=1.1, delta_scale=1.05,
                 c_val=0.02, verbose=False):
        self.initial_df = initial_df
        self.init_symbols = list(initial_df)
        self.all_found_symbols = all_found_symbols
        self.data = [initial_df[col_name] for col_name in self.init_symbols]
        self.symbols = [nu, eta]
        self.verbose = verbose
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.delta = delta
        self.delta_scale = delta_scale
        self.c_val = c_val
        print(initial_df)

    def bacon_iterations(self):
        '''
        Runs the BACON iterations and returns the constant variables with its value.
        If the program already has constants they're returned. A maximum complexity
        that can be found is induced by j = 3. If there is no update on the bacon
        iterations, the program returns the last data/symbol founds.
        '''
        self.lin_data = None
        self.update = ""

        j = 0
        while self.update != "constant" and j < 5:
            sy_start = len(self.symbols)
            self.bacon_instance(0, -1)
            if self.update == "product" or self.update == "division":
                self.bacon_instance(1, -2)
            elif self.update != "linear" and self.update != "constant":
                self.bacon_instance(1, -1)
            j += 1
            sy_end = len(self.symbols)
            if sy_start == sy_end or self.update == "linear":
                break

        if self.update != "constant" and self.update != "linear":
            if self.verbose:
                print("BACON 1: No relation found within acceptable parameters.")
                print("         Rerunning with increased epsilon and delta params.")
            new_eps = self.epsilon_scale*self.epsilon
            new_delta = self.delta_scale*self.delta
            return BACON_1(self.initial_df, self.all_found_symbols,
                           new_eps, new_delta, self.verbose).bacon_iterations()

        return self.data[-1], self.subs_expr(self.symbols[-1]), self.lin_data

    def bacon_instance(self, start, finish):
        '''
        A single bacon instance on the variables and data passed in. Follows the logic
        specified by Pat Langley's BACON.1.
        '''
        a, b = self.data[start], self.data[finish]
        a_, b_ = self.symbols[start], self.symbols[finish]

        m, c, r, p, std_err = lr(abs(a), abs(b))
        self.check_constant(b_, abs(b))

        if self.update != "constant":
            if 1 - abs(r) < self.epsilon and abs(c/fmean(b)) > self.c_val:
                # print(1-abs(r), abs(c/fmean(b)), self.subs_expr(a_), self.subs_expr(b_))
                self.linear(a_, b_, a, b)

            elif r > 0:
                sy = simplify(a_/b_)
                if self.new_term(sy):
                    self.division(a_, b_, a, b)

            elif r < 0:
                sy = simplify(a_*b_)
                if self.new_term(sy):
                    self.product(a_, b_, a, b)

    def new_term(self, symbol):
        '''
        Checks if the new term BACON.1 proposed has been tried already or is invalid.
        '''
        if symbol not in self.symbols \
           and simplify(1/symbol) not in self.symbols \
           and symbol != 1 \
           and len(symbol.free_symbols) != 1:
            return True
        else:
            self.update = ""
            return False

    def check_constant(self, symbol, data):
        '''
        Checks if the new term BACON.1 proposed is constant.
        '''
        if len(self.symbols) == 2:
            real_symb = self.subs_expr(symbol)
            if len(real_symb.free_symbols) == 1:
                if str(real_symb) not in dummies:
                    return

        M = fmean(data)
        if all(M*(1 - self.delta) < val < M*(1 + self.delta) for val in data):
            self.update = "constant"
            if self.verbose:
                print(f"BACON 1: {self.subs_expr(symbol)} is constant within our error")

    def linear(self, symbol_1, symbol_2, data_1, data_2):
        '''
        Calculates the terms involved in the found linear relationship.
        '''
        m, c = np.polyfit(data_1, data_2, 1)
        self.symbols.append(simplify(symbol_2 - m*symbol_1))
        self.data.append(data_2 - m*data_1)
        self.update = "linear"
        k = laws_helper.new_symbol(self.all_found_symbols)
        self.lin_data = ["linear", k, self.subs_expr(symbol_2 - k*symbol_1),
                         m, self.subs_expr(symbol_2), self.subs_expr(symbol_1)]
        if self.verbose:
            print(f"BACON 1: {self.subs_expr(symbol_2)} is linearly prop. to {self.subs_expr(symbol_1)}, we then see {self.subs_expr(self.symbols[-1])} is constant")  # noqa

    def product(self, symbol_1, symbol_2, data_1, data_2):
        '''
        Performs a product operation on the current terms.
        '''
        self.symbols.append(simplify(symbol_1*symbol_2))
        self.data.append(data_1*data_2)
        self.update = "product"
        if self.verbose:
            print(f"BACON 1: {self.subs_expr(symbol_1)} increases whilst {self.subs_expr(symbol_2)} decreases, considering new variable {self.subs_expr(simplify(symbol_1*symbol_2))}")  # noqa

    def division(self, symbol_1, symbol_2, data_1, data_2):
        '''
        Performs a division operation on the current terms.
        '''
        self.symbols.append(simplify(symbol_1/symbol_2))
        self.data.append(data_1/data_2)
        self.update = "division"
        if self.verbose:
            print(f"BACON 1: {self.subs_expr(symbol_1)} increases whilst {self.subs_expr(symbol_2)} increases, considering new variable {self.subs_expr(simplify(symbol_1/symbol_2))}")  # noqa

    def subs_expr(self, expr):
        e1 = expr.subs(nu, self.init_symbols[0])
        e2 = e1.subs(eta, self.init_symbols[1])
        return e2
