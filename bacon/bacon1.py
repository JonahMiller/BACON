import numpy as np
from statistics import fmean
import sympy as sym
from sklearn.metrics import mean_absolute_error as mse
from scipy.stats import linregress as lr

import warnings
warnings.filterwarnings('ignore')


class BACON_1:
    def __init__(self, initial_df, bacon_1_info=False, lin_bound=0.0001, delta=0.01):
        self.symbols = list(initial_df)
        self.data = [initial_df[col_name] for col_name in self.symbols]
        self.info = bacon_1_info
        self.lin_bound = lin_bound
        self.delta = delta

    def new_symbol(self):
        '''
        Draws new variables to use in the case of linear relationships. Starts with
        "a" and draws onwards in the alphabet.
        '''
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        index = 0
        l = sym.Symbol(letters[index])
        used_symbols = sum(s for s in self.symbols).free_symbols
        while l in used_symbols and index < 25:
            index += 1
            l = sym.Symbol(letters[index])
        if index == 25:
            raise Exception
        return l

    def bacon_iterations(self):
        '''
        Runs the BACON iterations and returns the constant variables with its value. If the 
        program already has constants they're returned. A maximum complexity that can be found
        is induced by j = 10. If there is no update on the bacon iterations, the program returns 
        the last data/symbol founds.

        TODO: update how the fail returns in the latter case above.
        '''
        self.lin_data = ""
        init_d, init_sy, self.update = self.initial_constant()
        if self.update == "constant":
            return init_d, init_sy, self.update

        j = 0
        while self.update != "constant" and j < 10:
            sy_start = len(self.symbols)
            self.bacon_instance(0, -1)
            if self.update == "product" or self.update == "division":
                self.bacon_instance(1, -2)
            else:
                self.bacon_instance(1, -1)
            j += 1
            sy_end = len(self.symbols)
            if sy_start == sy_end:
                break
        return self.data[-1], self.symbols[-1], self.lin_data
    
    def initial_constant(self):
        '''
        Checks if any variables are initialised as constant.
        '''
        M_0 = fmean(self.data[0])
        M_1 = fmean(self.data[1])
        if all(M_0*(1 - self.delta) < abs(v) < M_0*(1 + self.delta) for v in self.data[0]):
            return self.data[0], self.symbols[0], "constant"
        elif all(M_1*(1 - self.delta) < abs(v) < M_1*(1 + self.delta) for v in self.data[1]):
            return self.data[1], self.symbols[1], "constant"
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
            if 1 - abs(r) < self.lin_bound and abs(c) > 0.0000001:
                sy = sym.simplify(b_ - m*a_)
                if self.new_term(sy):
                    self.check_linear(a_, b_, a, b, r)

            elif r > 0:
                sy = sym.simplify(a_/b_)
                if self.new_term(sy):
                    self.division(a_, b_, a, b)
        
            elif r < 0:
                sy = sym.simplify(a_*b_)
                if self.new_term(sy):
                    self.product(a_, b_, a, b)
    
    def new_term(self, symbol):
        '''
        Checks if the new term BACON.1 proposed has been tried already.
        '''
        if symbol not in self.symbols and sym.simplify(1/symbol) not in self.symbols and symbol != 1:
            return True
        else:
            self.update = ""
            return False 

    def check_constant(self, symbol, data):
        '''
        Checks if the new term BACON.1 proposed is constant.
        '''
        M = fmean(data)
        if all(M*(1 - self.delta) < l < M*(1 + self.delta) for l in data):
            self.update = "constant"
            if self.info:
                print(f"BACON 1: {symbol} is constant within our error")
    
    def check_linear(self, symbol_1, symbol_2, data_1, data_2, r):
        '''
        Checks if the new term BACON.1 proposed is linearly proportional
        with the other term in context.
        '''
        m, c = np.polyfit(data_1, data_2, 1)
        self.symbols.append(sym.simplify(symbol_2 - m*symbol_1))
        self.data.append(data_2 - m*data_1)
        self.update = "linear"
        k = self.new_symbol()
        self.lin_data = ["linear", k, symbol_2 - k*symbol_1, m]
        if self.info:
            print(f"BACON 1: {symbol_2} is linearly prop. to {symbol_1}, we then see {self.symbols[-1]} is constant")
    
    def product(self, symbol_1, symbol_2, data_1, data_2):
        '''
        Performs a product operation on the current terms.
        '''
        self.symbols.append(sym.simplify(symbol_1*symbol_2))
        self.data.append(data_1*data_2)
        self.update = "product"
        if self.info:
            print(f"BACON 1: {symbol_1} increases whilst {symbol_2} decreases, considering new variable {sym.simplify(symbol_1*symbol_2)}")
    
    def division(self, symbol_1, symbol_2, data_1, data_2):
        '''
        Performs a division operation on the current terms.
        '''
        self.symbols.append(sym.simplify(symbol_1/symbol_2))
        self.data.append(data_1/data_2)
        self.update = "division"
        if self.info:
            print(f"BACON 1: {symbol_1} increases whilst {symbol_2} increases, considering new variable {sym.simplify(symbol_1/symbol_2)}")
