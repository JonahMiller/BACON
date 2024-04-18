import numpy as np
import sympy as sym
from math import comb, factorial
from scipy import integrate


def kepler(noise=0):
    n = np.arange(1, 11)
    P = np.power(n, 3)
    D = np.power(n, 2) + noise*np.random.normal(0, 1, 11)
    return [P, D], [sym.Symbol("P"), sym.Symbol("D")]


def boyle(noise=0):
    V = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32])
    P = np.array([29.750, 19.125, 14.375, 9.5, 7.125, 5.625, 4.875, 4.25, 3.75,
                  3.375, 3, 2.625, 2.25, 2, 1.875, 1.75, 1.5, 1.375, 1.25])
    return [P, V], [sym.Symbol("P"), sym.Symbol("V")]


def birthday(noise=0):
    def general_n(n):
        for p in range(1, n + 1):
            if 1 - comb(n, n - p)*factorial(p)/n**p >= 0.5:
                return p
        return 0
    N = np.array([2, 5, 10, 23, 50, 79, 365])
    P = np.vectorize(general_n)(N) + noise*np.random.normal(0, 1, 7)
    return [N, P], [sym.Symbol("N"), sym.Symbol("P")]


# https://scientific-python.readthedocs.io/en/latest/notebooks_rst/3_Ordinary_Differential_Equations/02_Examples/Lotka_Volterra_model.html
def lotka_volterra(alpha, beta, gamma, delta, x_0, y_0, Nt=1000, tmax=30., noise=0):
    def derivative(X, t, alpha, beta, delta, gamma):
        x, y = X
        dotx = x * (alpha - beta * y)
        doty = y * (-delta + gamma * x)
        return np.array([dotx, doty])
    t = np.linspace(0., tmax, Nt)
    X_0 = [x_0, y_0]
    res = integrate.odeint(derivative, X_0, t, args=(alpha, beta, delta, gamma))
    X, Y = res.T
    Y = Y + noise*np.random.normal(0, 1, Nt)
    return [X, Y], [sym.Symbol("X"), sym.Symbol("Y")]
