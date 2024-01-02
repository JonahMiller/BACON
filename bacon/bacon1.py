import numpy as np
import matplotlib.pyplot as plt
from statistics import fmean
import sympy as sym
import data as d
from sklearn.metrics import mean_absolute_error as mse


eps = 0.001
delta = 0.3
mse_error = 0.01


class BACON_1:
    def __init__(self, data, symbols, info, mse_error=0.01, delta=0.3, eps=0.001):
        self.data = data
        self.symbols = symbols
        self.info = info
        self.mse_error = mse_error
        self.eps = eps
        self.delta = delta

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
        d, sy, u, dt = self.run_bacon(0, 1, self.data, self.symbols, [])
        u = False
        j = 0
        while u != "constant" and j < 300:
            d, sy, u, dt = self.run_bacon(0, -1, d, sy, dt)
            if u == "yes":
                d, sy, u, dt = self.run_bacon(1, -2, d, sy, dt)
            elif u == "no":
                d, sy, u, dt = self.run_bacon(1, -1, d, sy, dt)
            j += 1
        return d[-1], sy[-1], dt

    def run_bacon(self, start, finish, data, symbols, previous_op):
        a, b = data[start], data[finish]
        a_, b_ = symbols[start], symbols[finish]
        update = "no successful variable"

        m, c = np.polyfit(a, b, 1)

        if -self.eps < m < self.eps:
            M = fmean(data[-1])
            if all(M*(1 - self.delta) < l < M*(1 + self.delta) for l in data[-1]):
                update = "constant"
                if self.info:
                    print(f"{symbols[-1]} is constant within our error")

        elif mse(a*m + c, b) < self.mse_error and abs(c) > 0.0001:
            sy = self.symbols.append(sym.simplify(b_ - m*a_))
            data.append(b - m*a)
            update = "linear"
            k = self.new_symbol()
            previous_op = ["linear", k, b_ - k*a_, m]
            if self.info:
                print(f"{b_} is linearly prop. to {a_}, we then see {symbols[-1]} is constant")

        elif m > self.eps:
            sy = sym.simplify(a_/b_)
            if sy not in symbols and sym.simplify(1/sy) not in symbols and sy != 1:
                self.symbols.append(sym.simplify(a_/b_))
                data.append(a / b)
                update = "division"
                previous_op = update
                if self.info:
                    print(f"{a_} increases whilst {b_} also increases, considering new variable {sym.simplify(a_/b_)}")

        elif m < - self.eps:
            sy = sym.simplify(a_*b_)
            if sy not in symbols and sym.simplify(1/sy) not in symbols and sy != 1:
                self.symbols.append(sym.simplify(a_*b_))
                data.append(a * b)
                update = "product"
                previous_op = update
                if self.info:
                    print(f"{a_} increases whilst {b_} decreases, considering new variable {sym.simplify(a_*b_)}")
        return data, symbols, update, previous_op


if __name__ == '__main__':
    initial_data, initial_symbols = d.mini_boyle()
    a = BACON_1(initial_data, initial_symbols, info=True)
    a.bacon_iterations()

