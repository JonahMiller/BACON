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
        self.info = False

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
        init_d, init_sy, init_up = self.initial_constant()
        if init_up == "constant":
            return init_d, init_sy, init_up
        d, sy, u, dt = self.run_bacon(0, 1, self.data, self.symbols, "")
        j = 0
        while u != "constant" and j < 300:
            sy_start = len(sy)
            d, sy, u, dt = self.run_bacon(0, -1, d, sy, dt)
            if u == "product" or u == "division":
                d, sy, u, dt = self.run_bacon(1, -2, d, sy, dt)
            else:
                d, sy, u, dt = self.run_bacon(1, -1, d, sy, dt)
            j += 1
            sy_end = len(sy)
            if sy_start == sy_end:
                break
        return d[-1], sy[-1], dt
    
    def initial_constant(self):
        M_0 = fmean(self.data[0])
        M_1 = fmean(self.data[1])
        if all(M_0*(1 - self.delta) < abs(v) < M_0*(1 + self.delta) for v in self.data[0]):
            return self.data[0], self.symbols[0], "constant"
        elif all(M_1*(1 - self.delta) < abs(v) < M_1*(1 + self.delta) for v in self.data[1]):
            return self.data[1], self.symbols[1], "constant"
        else:
            return self.data, self.symbols, "no relationship"

    def run_bacon(self, start, finish, data, symbols, previous_op):
        a, b = data[start], data[finish]
        a_, b_ = symbols[start], symbols[finish]
        update = "no relationship"

        m, c = np.polyfit(abs(a), abs(b), 1)

        if -self.eps < m < self.eps:
            M = fmean(b)
            if all(M*(1 - self.delta) < l < M*(1 + self.delta) for l in b):
                update = "constant"
                if self.info:
                    print(f"BACON 1: {b_} is constant within our error")

        elif mse(abs(a)*m + c, abs(b)) < self.mse_error and abs(c) > 0.0001:
            sy = self.symbols.append(sym.simplify(b_ - m*a_))
            data.append(b - m*a)
            update = "linear"
            k = self.new_symbol()
            previous_op = ["linear", k, b_ - k*a_, m]
            if self.info:
                print(f"BACON 1: {b_} is linearly prop. to {a_}, we then see {symbols[-1]} is constant")

        elif m > self.eps:
            sy = sym.simplify(a_/b_)
            if sy not in symbols and sym.simplify(1/sy) not in symbols and sy != 1:
                self.symbols.append(sym.simplify(a_/b_))
                data.append(a / b)
                update = "division"
                previous_op = update
                if self.info:
                    print(f"BACON 1: {a_} increases whilst {b_} also increases, considering new variable {sym.simplify(a_/b_)}")

        elif m < - self.eps:
            sy = sym.simplify(a_*b_)
            if sy not in symbols and sym.simplify(1/sy) not in symbols and sy != 1:
                self.symbols.append(sym.simplify(a_*b_))
                data.append(a * b)
                update = "product"
                previous_op = update
                if self.info:
                    print(f"BACON 1: {a_} increases whilst {b_} decreases, considering new variable {sym.simplify(a_*b_)}")
        
        return data, symbols, update, previous_op
