import numpy as np
from sympy import Symbol
import pandas as pd
from math import comb, factorial
from scipy import integrate

import sys
sys.path.append("..")
from bacon6 import BACON_6  # noqa: E402


def kepler(noise=0):
    n = np.arange(1, 11)
    P = np.power(n, 3)
    D = np.power(n, 2) + noise*np.random.normal(0, 1, 10)
    return [P, D], [Symbol("P"), Symbol("D")]


def ideal(noise=0):
    a = 2
    P = np.array([10, 20, 30])
    V = a/P + noise*np.random.normal(0, 1, 3)
    return [P, V], [Symbol("P"), Symbol("V")]


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
    return [N, N*Q], [Symbol("N"), Symbol("Q")]


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
    return [X, Y*X], [Symbol("X"), Symbol("Y")]


def run_bacon_6(function, expression=None, unknowns=None,
                noise=0, steps=2, N_threshold=4):
    init_data, init_symb = function(noise)
    initial_df = pd.DataFrame({v: d for v, d in zip(init_symb, init_data)})
    bacon = BACON_6(initial_df, init_symb,
                    expression=expression, unknowns=unknowns,
                    step=steps, N_threshold=N_threshold)
    bacon.main()


if __name__ == "__main__":
    run_bacon_6(birthday,
                expression="j*nu**(3/2) + k*nu + l*nu**(1/2)",
                unknowns=["j", "k", "l"])

    run_bacon_6(boyle,
                expression="j/nu",
                unknowns=["j"])

    run_bacon_6(kepler,
                expression="j*nu**(2/3)",
                unknowns=["j"])

    run_bacon_6(ideal)
    run_bacon_6(ideal,
                noise=0.01)
    run_bacon_6(ideal,
                noise=0.1)
