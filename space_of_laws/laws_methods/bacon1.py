import numpy as np
from statistics import fmean
from sympy import Symbol, simplify
from scipy.stats import linregress as lr

from utils import laws_helper as laws_helper

import warnings
warnings.filterwarnings('ignore')


eta = Symbol("eta")
nu = Symbol("nu")


class BACON_1:
    def __init__(self, initial_df, all_found_symbols,
                 epsilon=0.001, delta=0.1, verbose=False):
        self.init_symbols = list(initial_df)
        self.all_found_symbols = all_found_symbols
        self.data = [initial_df[col_name] for col_name in self.init_symbols]
        self.symbols = [eta, nu]
        self.verbose = verbose
        self.epsilon = epsilon
        self.delta = delta

    def bacon_iterations(self):
        '''
        Runs the BACON iterations and returns the constant variables with its value.
        If the program already has constants they're returned. A maximum complexity
        that can be found is induced by j = 5. If there is no update on the bacon
        iterations, the program returns the last data/symbol founds.
        '''
        self.lin_data = None
        init_d, init_sy, self.update = self.initial_constant()
        if self.update == "constant":
            return init_d, self.subs_expr(init_sy), self.update

        j = 0
        while self.update != "constant" and j < 6:
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
        return self.data[-1], self.subs_expr(self.symbols[-1]), self.lin_data

    def initial_constant(self):
        '''
        Checks if any variables are initialised as constant.
        '''
        M = fmean(self.data[0])
        if all(abs(M)*(1 - self.delta) < abs(v) < abs(M)*(1 + self.delta) for v in self.data[0]):
            return self.data[0], self.symbols[0], "constant"
        else:
            return self.data, self.symbols, ""

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
            if 1 - abs(r) < self.epsilon and abs(c) > 0.0000001:
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
        Checks if the new term BACON.1 proposed has been tried already.
        '''
        if symbol not in self.symbols and simplify(1/symbol) not in self.symbols and symbol != 1:
            return True
        else:
            self.update = ""
            return False

    def check_constant(self, symbol, data):
        '''
        Checks if the new term BACON.1 proposed is constant.
        '''
        M = fmean(data)
        if all(M*(1 - self.delta) < val < M*(1 + self.delta) for val in data):
            self.update = "constant"
            if self.verbose:
                print(f"BACON 1: {symbol} is constant within our error")

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
            print(f"BACON 1: {symbol_2} is linearly prop. to {symbol_1},")
            print(f"         we then see {self.symbols[-1]} is constant")

    def product(self, symbol_1, symbol_2, data_1, data_2):
        '''
        Performs a product operation on the current terms.
        '''
        self.symbols.append(simplify(symbol_1*symbol_2))
        self.data.append(data_1*data_2)
        self.update = "product"
        if self.verbose:
            print(f"BACON 1: {symbol_1} increases whilst {symbol_2} decreases,")
            print(f"         considering new variable {simplify(symbol_1*symbol_2)}")

    def division(self, symbol_1, symbol_2, data_1, data_2):
        '''
        Performs a division operation on the current terms.
        '''
        self.symbols.append(simplify(symbol_1/symbol_2))
        self.data.append(data_1/data_2)
        self.update = "division"
        if self.verbose:
            print(f"BACON 1: {symbol_1} increases whilst {symbol_2} increases,")
            print(f"         considering new variable {simplify(symbol_1/symbol_2)}")

    def subs_expr(self, expr):
        e1 = expr.subs(eta, self.init_symbols[0])
        e2 = e1.subs(nu, self.init_symbols[1])
        return e2
