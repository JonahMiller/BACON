import numpy as np
from sympy import Symbol
from math import comb, factorial
from scipy import integrate


def basic(noise=0):
    M = np.array(9*[1] + 9*[2] + 9*[3])
    T = np.array(3*(3*[10] + 3*[20] + 3*[30]))
    P = np.array(3*(3*([10, 20, 30])))
    V = 3*M*T/P + noise*np.random.normal(0, 1, 27)
    return [M, T, P, V], [Symbol("M"), Symbol("T"), Symbol("P"), Symbol("V")]


def ideal(noise=0):
    a = 2
    b = 3
    M = np.array(9*[1] + 9*[2] + 9*[3])
    T = np.array(3*(3*[10] + 3*[20] + 3*[30]))
    P = np.array(3*(3*([10, 20, 30])))
    V = (a*M*T + b*M)/P + noise*np.random.normal(0, 1, 27)
    return [M, T, P, V], [Symbol("M"), Symbol("T"), Symbol("P"), Symbol("V")]


def ohm(noise=0):
    v = 2
    r = 3
    T = np.array(9*[100] + 9*[120] + 9*[140])
    D = np.array(3*(3*[1] + 3*[2] + 3*[3]))
    L = np.array(3*(3*([5, 10, 15])))
    I = np.array(T*D**2/(v*(L + r))) + noise*np.random.normal(0, 1, 27)  # noqa
    return [T, D, L, I], [Symbol("T"), Symbol("D"), Symbol("L"), Symbol("I")]


def black(noise=0):
    c_1 = 4
    c_2 = 2
    M_1 = np.array(27*[1] + 27*[2] + 27*[3])
    M_2 = np.array(3*(9*[1] + 9*[2] + 9*[3]))
    T_1 = np.array(9*(3*[50] + 3*[60] + 3*[70]))
    T_2 = np.array(27*([50, 60, 70]))
    T_f = np.array((c_1*M_1*T_1 + c_2*M_2*T_2)/(c_1*M_1 + c_2*M_2)) + noise*np.random.normal(0, 1, 81)
    return [M_1, M_2, T_1, T_2, T_f], \
           [Symbol("M_1"), Symbol("M_2"), Symbol("T_1"), Symbol("T_2"), Symbol("T_f")]


def kepler(noise=0):
    n = np.arange(1, 11)
    P = np.power(n, 3)
    D = np.power(n, 2) + noise*np.random.normal(0, 1, 10)
    return [P, D], [Symbol("P"), Symbol("D")]


def boyle(noise=0):
    V = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32])
    P = np.array([29.750, 19.125, 14.375, 9.5, 7.125, 5.625, 4.875, 4.25, 3.75,
                  3.375, 3, 2.625, 2.25, 2, 1.875, 1.75, 1.5, 1.375, 1.25])
    return [P, V], [Symbol("P"), Symbol("V")]


def birthday(noise=0):

    def Q_func(n):
        q = 0
        for k in range(1, n + 1):
            q += comb(n, n - k)*factorial(k)/n**k
        return q

    N = np.array([i for i in range(2, 10)])
    Q = np.vectorize(Q_func)(N) + noise*np.random.normal(0, 1, 8)
    return [N, N*Q], [Symbol("N"), Symbol("N")*Symbol("Q")]


# https://scientific-python.readthedocs.io/en/latest/notebooks_rst/3_Ordinary_Differential_Equations/02_Examples/Lotka_Volterra_model.html
def lotka_volterra(noise=0):
    alpha, beta, gamma, delta, x_0, y_0 = 1, 0.1, 0.2, 0.5, 10, 10
    Nt, tmax = 1000, 30

    def derivative(X, t, alpha, beta, delta, gamma):
        x, y = X
        dotx = x * (alpha - beta * y)
        doty = y * (-delta + gamma * x)
        return np.array([dotx, doty])

    t = np.linspace(0., tmax, Nt)
    X_0 = [x_0, y_0]
    res = integrate.odeint(derivative, X_0, t, args=(alpha, beta, delta, gamma))
    X, Y = res.T
    X = [X[idx] for idx in range(len(X)) if idx % 100 == 0]
    Y = [Y[idx] for idx in range(len(Y)) if idx % 100 == 0]
    Y = Y + noise*np.random.normal(0, 1, 10)
    return [X, Y], [Symbol("X"), Symbol("Y")]


def allowed_data():
    data = {
            "basic": basic,
            "ohm": ohm,
            "ideal": ideal,
            "black": black,
            "kepler": kepler,
            "boyle": boyle,
            "birthday": birthday,
            "lv": lotka_volterra
           }
    return data
