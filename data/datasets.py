import numpy as np
import sympy as sym

# from data.ode_systems import SIR


# FUNCTIONS FOR BACON.1

def kepler(noise=0):
    n = np.arange(1, 11)
    P = np.power(n, 3)
    D = np.power(n, 2)
    return [P, D], [sym.Symbol("P"), sym.Symbol("D")]


def boyle(noise=0):
    V = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32])
    P = np.array([29.750, 19.125, 14.375, 9.5, 7.125, 5.625, 4.875, 4.25, 3.75,
                  3.375, 3, 2.625, 2.25, 2, 1.875, 1.75, 1.5, 1.375, 1.25])
    return [P, V], [sym.Symbol("P"), sym.Symbol("V")]


# FUNCTIONS FOR BACON.3/5:

def ideal(noise=0):
    a = 2
    b = 3
    M = np.array(9*[1] + 9*[2] + 9*[3])
    T = np.array(3*(3*[10] + 3*[20] + 3*[30]))
    P = np.array(3*(3*([10, 20, 30])))
    V = (a*M*T + b*M)/P + noise*np.random.normal(0, 1, 27)
    return [M, T, P, V], [sym.Symbol("M"), sym.Symbol("T"), sym.Symbol("P"), sym.Symbol("V")]


def ohm(noise=0):
    v = 2
    r = 3
    T = np.array(9*[100] + 9*[120] + 9*[140])
    D = np.array(3*(3*[1] + 3*[2] + 3*[3]))
    L = np.array(3*(3*([5, 10, 15])))
    I = np.array(T*D**2/(v*(L + r)))
    return [T, D, L, I], [sym.Symbol("T"), sym.Symbol("D"), sym.Symbol("L"), sym.Symbol("I")]


def black(noise=0):
    c_1 = 4.2
    c_2 = 1.81
    M_1 = np.array(27*[1] + 27*[2] + 27*[3])
    M_2 = np.array(3*(9*[1] + 9*[2] + 9*[3]))
    T_1 = np.array(9*(3*[50] + 3*[60] + 3*[70]))
    T_2 = np.array(27*([50, 60, 70]))
    T_f = np.array((c_1*M_1*T_1 + c_2*M_2*T_2)/(c_1*M_1 + c_2*M_2))
    return [M_1, M_2, T_1, T_2, T_f], \
           [sym.Symbol("M_1"), sym.Symbol("M_2"), sym.Symbol("T_1"), sym.Symbol("T_2"), sym.Symbol("T_f")]


# def sir(noise=0):
#     beta = 1e-8
#     v = 0.02
#     S_0 = 1e7
#     I_0 = 1000
#     R_0 = 0
#     S, I, R = SIR(beta, v, S_0, I_0, R_0, 0, 600, 0.004)
#     return [S, I, R], [sym.Symbol("S"), sym.Symbol("I"), sym.Symbol("R")]


def allowed_data():
    data = {
            "kepler": kepler,
            "boyle": boyle,
            "ohm": ohm,
            "ideal": ideal,
            "black": black
            # "sir": sir
           }
    return data
