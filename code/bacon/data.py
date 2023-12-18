import numpy as np
import sympy as sym


def kepler():
    n = np.arange(1, 11)
    P = np.power(n, 3)
    D = np.power(n, 2)
    return [P, D], [sym.Symbol("P"), sym.Symbol("D")]

def boyle(noise=0):
    c = 3
    V = np.linspace(1, 32, 40)
    P = c/V + noise*np.random.normal(0, 1, 40)
    return [P, V], [sym.Symbol("P"), sym.Symbol("V")]

def boyle_real():
    V = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32])
    P = np.array([29.750, 19.125, 14.375, 9.5, 7.125, 5.625, 4.875, 4.25, 3.75, 3.375, 3, 2.625, 2.25, 2, 1.875, 1.75, 1.5, 1.375, 1.25])
    return [P, V], [sym.Symbol("P"), sym.Symbol("V")]

def ohm(noise=0):
    v = 2
    r = 3
    L = np.linspace(2, 130, 10)
    I = v/(r + L) + noise*np.random.normal(0, 1, 10)
    return [I, L], [sym.Symbol("I"), sym.Symbol("L")]

def ohm_real():
    L = np.array([2, 4, 6, 10, 18, 34, 66, 130])
    I = np.array([326.75, 300.75, 277.75, 238.25, 190.75, 134.50, 83.25, 48.50])
    return [I, L], [sym.Symbol("I"), sym.Symbol("L")]

def ideal_gas():
    M = np.array(9*[1] + 9*[2] + 9*[3])
    T = np.array(3*(3*[10] + 3*[20] + 3*[30]))
    P = np.array(3*(3*([10, 20, 30])))
    V = M*T/P
    # return [M, T, P, V], [sym.Symbol("M"), sym.Symbol("T"), sym.Symbol("P"), sym.Symbol("V")]
    return [P, M, T, V], [sym.Symbol("P"), sym.Symbol("M"), sym.Symbol("T"), sym.Symbol("V")]

def mini_boyle():
    P = np.array([1, 2, 3])
    V = np.array([2.354, 1.177, 0.785])
    return [P, V], [sym.Symbol("P"), sym.Symbol("V")]

