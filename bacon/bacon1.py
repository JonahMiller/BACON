import numpy as np
from statistics import fmean
import sympy as sym
from sklearn.metrics import mean_absolute_error as mse

import warnings
warnings.filterwarnings('ignore')

class BACON_1:
    def __init__(self, data, symbols, info=False, mse_error=0.0001, delta=0.01, eps=0.0001):
        self.data = data
        self.symbols = symbols
        self.info = info
        self.mse_error = mse_error
        self.eps = eps
        self.delta = delta
        self.info = info

    def new_symbol(self):
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
        self.lin_data = ""
        init_d, init_sy, self.update = self.initial_constant()
        if self.update == "constant":
            return init_d, init_sy, self.update
        self.run_bacon(0, 1)
        j = 0
        while self.update != "constant" and j < 10:
            sy_start = len(self.symbols)
            self.run_bacon(0, -1)
            if self.update == "product" or self.update == "division":
                self.run_bacon(1, -2)
            else:
                self.run_bacon(1, -1)
            j += 1
            sy_end = len(self.symbols)
            # if sy_start == sy_end:
            #     break
        return self.data[-1], self.symbols[-1], self.lin_data
    
    def initial_constant(self):
        M_0 = fmean(self.data[0])
        M_1 = fmean(self.data[1])
        if all(M_0*(1 - self.delta) < abs(v) < M_0*(1 + self.delta) for v in self.data[0]):
            return self.data[0], self.symbols[0], "constant"
        elif all(M_1*(1 - self.delta) < abs(v) < M_1*(1 + self.delta) for v in self.data[1]):
            return self.data[1], self.symbols[1], "constant"
        else:
            return self.data, self.symbols, ""

    def run_bacon(self, start, finish):
        a, b = self.data[start], self.data[finish]
        a_, b_ = self.symbols[start], self.symbols[finish]

        m, c = np.polyfit(abs(a), abs(b), 1)
        # m, c = np.polyfit(a, b, 1)
        if abs(m) < self.eps:
            self.check_constant(b_, b)

        # elif mse(a*m + c, b) < self.mse_error and abs(c) > 0.0001:
        elif mse(abs(a)*m + c, abs(b)) < self.mse_error and abs(c) > 0.0001:
            sy = sym.simplify(b_ - m*a_)
            if self.new_term(sy):
                self.check_linear(a_, b_, a, b)

        elif m > self.eps:
            sy = sym.simplify(a_/b_)
            if self.new_term(sy):
                self.division(a_, b_, a, b)
    
        elif m < - self.eps:
            sy = sym.simplify(a_*b_)
            if self.new_term(sy):
                self.product(a_, b_, a, b)
    
    def new_term(self, symbol):
        if symbol not in self.symbols and sym.simplify(1/symbol) not in self.symbols and symbol != 1:
            return True
        else:
            self.update = ""
            return False 

    def check_constant(self, symbol, data):
        M = fmean(data)
        if all(M*(1 - self.delta) < l < M*(1 + self.delta) for l in data):
            self.update = "constant"
            if self.info:
                print(f"BACON 1: {symbol} is constant within our error")
    
    def check_linear(self, symbol_1, symbol_2, data_1, data_2):
        m, c = np.polyfit(data_1, data_2, 1)
        if mse(data_1*m + c, data_2)  < self.mse_error and abs(c) > 0.0001:
            self.symbols.append(sym.simplify(symbol_2 - m*symbol_1))
            self.data.append(data_2 - m*data_1)
            self.update = "linear"
            k = self.new_symbol()
            self.lin_data = ["linear", k, symbol_2 - k*symbol_1, m]
            if self.info:
                print(f"BACON 1: {symbol_2} is linearly prop. to {symbol_1}, we then see {self.symbols[-1]} is constant")
    
    def product(self, symbol_1, symbol_2, data_1, data_2):
        self.symbols.append(sym.simplify(symbol_1*symbol_2))
        self.data.append(data_1*data_2)
        self.update = "product"
        if self.info:
            print(f"BACON 1: {symbol_1} increases whilst {symbol_2} decreases, considering new variable {sym.simplify(symbol_1*symbol_2)}")
    
    def division(self, symbol_1, symbol_2, data_1, data_2):
        self.symbols.append(sym.simplify(symbol_1/symbol_2))
        self.data.append(data_1/data_2)
        self.update = "division"
        if self.info:
            print(f"BACON 1: {symbol_1} increases whilst {symbol_2} increases, considering new variable {sym.simplify(symbol_1/symbol_2)}")
