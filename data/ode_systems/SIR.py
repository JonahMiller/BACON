import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def derivative(X, t, beta, v):
    S, I, R = X
    N = sum(X)
    dS_dt = - beta*I*S
    dI_dt = beta*I*S - v*I
    dR_dt = v*I
    return np.array([dS_dt, dI_dt, dR_dt])


# From https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html
def rungekutta4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y


if __name__ == "__main__":
    beta = 1e-8
    v = 0.02
    S_0 = 1e7
    I_0 = 1000
    R_0 = 0

    time = np.arange(0, 600, 0.0004)
    sol = rungekutta4(derivative, [S_0, I_0, R_0], time, args=(beta, v))

    plt.plot(time, sol[:, 0], "black", label="S")
    plt.plot(time, sol[:, 1], "red", label="I")
    plt.plot(time, sol[:, 2], "blue", label="R")
    plt.legend(loc="best")
    plt.ylabel("Population")
    plt.xlabel("Time")
    plt.grid()
    plt.show()
