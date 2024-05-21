import numpy as np
from sympy import Symbol
from math import comb, factorial
from scipy import integrate


np.random.seed(2)


def ideal(noise=0):
    M = np.array(9*[1] + 9*[2] + 9*[3])
    T = np.array(3*(3*[10] + 3*[20] + 3*[30]))
    P = np.array(3*(3*([1000, 2000, 3000])))
    V = M*(T + 273)/P
    if noise:
        V += np.random.normal(0, noise*abs(V))
    return [M, T, P, V], [Symbol("M", real=True), Symbol("T", real=True),
                          Symbol("P", real=True), Symbol("V", real=True)]


def ohm(noise=0):
    v = 2
    r = -3
    T = np.array(9*[100] + 9*[120] + 9*[140])
    D = np.array(3*(3*[0.01] + 3*[0.02] + 3*[0.03]))
    L = np.array(3*(3*([0.5, 1, 1.5])))
    I = np.array(T*D**2/(v*(L - r)))  # noqa
    if noise:
        I += np.random.normal(0, noise*abs(I))
    return [T, D, L, I], [Symbol("T", real=True), Symbol("D", real=True),
                          Symbol("L", real=True), Symbol("I", real=True)]


def black(noise=0):
    c_1 = 1
    c_2 = 1
    M_1 = np.array(27*[1] + 27*[2] + 27*[3])
    M_2 = np.array(3*(9*[1] + 9*[2] + 9*[3]))
    T_1 = np.array(9*(3*[50] + 3*[60] + 3*[70]))
    T_2 = np.array(27*([50, 60, 70]))
    T_f = np.array((c_1*M_1*T_1 + c_2*M_2*T_2)/(c_1*M_1 + c_2*M_2))
    if noise:
        T_f += np.random.normal(0, noise*abs(T_f))
    return [M_1, M_2, T_1, T_2, T_f], \
           [Symbol("M_1", real=True), Symbol("M_2", real=True),
            Symbol("T_1", real=True), Symbol("T_2", real=True),
            Symbol("T_f", real=True)]


def synthetic(noise=0):
    X = np.arange(1, 6, dtype="float64")
    Y = 3/(X+2)
    if noise:
        Y += np.random.normal(0, noise*abs(Y))
    return [X, Y], [Symbol("X"), Symbol("Y")]


def kepler(noise=0):
    n = np.arange(1, 7, dtype="float64")
    P = np.power(n, 3)
    D = np.power(n, 2)
    if noise:
        D += np.random.normal(0, noise*abs(D))
    return [P, D], [Symbol("P"), Symbol("D")]


def birthday(noise=0):

    def Q_func(n):
        q = 0
        for k in range(1, n + 1):
            q += comb(n, n - k)*factorial(k)/n**k
        return q

    N = np.array([i for i in range(2, 10)])
    Q = np.vectorize(Q_func)(N)
    if noise:
        Q += np.random.normal(0, noise*abs(Q))
    return [N, N*Q], [Symbol("N_"), Symbol("Q*N_")]


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
    if noise:
        Y += np.random.normal(0, noise*abs(Y))
    return [X, Y], [Symbol("X"), Symbol("Y")]


def allowed_data():
    data = {
            "ohm": ohm,
            "ideal": ideal,
            "black": black,
            "kepler": kepler,
            "birthday": birthday,
            "lv": lotka_volterra,
            "synthetic": synthetic,
           }
    return data
