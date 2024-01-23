import numpy as np
import sympy as sym


def kepler(noise=0):
    n = np.arange(1, 11)
    P = np.power(n, 3)
    D = np.power(n, 2)
    return [P, D], [sym.Symbol("P"), sym.Symbol("D")]

def boyle_synthetic(noise=0):
    c = 3
    V = np.linspace(1, 32, 40)
    P = c/V + noise*np.random.normal(0, 1, 40)
    return [P, V], [sym.Symbol("P"), sym.Symbol("V")]

def boyle_real(noise=0):
    V = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32])
    P = np.array([29.750, 19.125, 14.375, 9.5, 7.125, 5.625, 4.875, 4.25, 3.75, 3.375, 3, 2.625, 2.25, 2, 1.875, 1.75, 1.5, 1.375, 1.25])
    return [P, V], [sym.Symbol("P"), sym.Symbol("V")]

def ohm_synthetic(noise=0):
    v = 2
    r = 3
    L = np.linspace(2, 130, 10)
    I = v/(r + L) + noise*np.random.normal(0, 1, 10)
    return [I, L], [sym.Symbol("I"), sym.Symbol("L")]

def ohm_real(noise=0):
    L = np.array([2, 4, 6, 10, 18, 34, 66, 130])
    I = np.array([326.75, 300.75, 277.75, 238.25, 190.75, 134.50, 83.25, 48.50])
    return [I, L], [sym.Symbol("I"), sym.Symbol("L")]

def ideal_gas(noise=0):
    a = 2
    b = 3
    M = np.array(9*[1] + 9*[2] + 9*[3])
    T = np.array(3*(3*[10] + 3*[20] + 3*[30]))
    P = np.array(3*(3*([10, 20, 30])))
    V = (a*M*T + b*M)/P + noise*np.random.normal(0, 1, 27)
    return [M, T, P, V], [sym.Symbol("M"), sym.Symbol("T"), sym.Symbol("P"), sym.Symbol("V")]

def ohm_large(noise=0):
    v = 2
    r = 3
    T = np.array(9*[100] + 9*[120] + 9*[140])
    D = np.array(3*(3*[1] + 3*[2] + 3*[3]))
    L = np.array(3*(3*([5, 10, 15])))
    I = np.array(T*D**2/(v*(L + r)))
    return [T, D, L, I], [sym.Symbol("T"), sym.Symbol("D"), sym.Symbol("L"), sym.Symbol("I")]

def black(noise=0):
    c_1 = 0.31
    c_2 = 2.54
    M_1 = np.array(27*[1] + 27*[2] + 27*[3])
    M_2 = np.array(3*(9*[1] + 9*[2] + 9*[3]))
    T_1 = np.array(9*(3*[50] + 3*[60] + 3*[70]))
    T_2 = np.array(27*([50, 60, 70]))
    T_f = np.array((c_1*M_1*T_1 + c_2*M_2*T_2)/(c_1*M_1 + c_2*M_2))
    return [M_1, M_2, T_1, T_2, T_f], \
           [sym.Symbol("M_1"), sym.Symbol("M_2"), sym.Symbol("T_1"), sym.Symbol("T_2"), sym.Symbol("T_f")]

def allowed_data():
    data = {
            "kepler": kepler,
            "boyle_synthetic": boyle_synthetic,
            "boyle_real": boyle_real,
            "ohm_synthetic": ohm_synthetic,
            "ohm_real": ohm_real,
            "ohm_large": ohm_large,
            "ideal_gas": ideal_gas,
            "black": black
           }
    return data