import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arviz as az
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

from scipy.optimize import least_squares
from scipy.integrate import odeint

import warnings
warnings.filterwarnings("ignore")


def derivative_SIR(X, t, logbeta, v):
    S, I, R = X
    beta = 10**logbeta
    dS_dt = - beta*I*S
    dI_dt = beta*I*S - v*I
    dR_dt = v*I
    return np.array([dS_dt, dI_dt, dR_dt])


# From https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html
def rungekutta4(func, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = func(y[i], t[i], *args)
        k2 = func(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = func(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = func(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y


def SIR(beta, v, S_0, I_0, R_0, init_time, end_time, steps):
    time = np.arange(init_time, end_time, steps)
    sol = odeint(derivative_SIR, [S_0, I_0, R_0], time, args=(beta, v))
    return sol[:, 0], sol[:, 1], sol[:, 2]


def plot_model(ax, x_y, time, alpha=1, lw=2, title="SIR model",):
    ax.plot(time, x_y[:, 0], color='black', alpha=alpha, lw=lw, label="S (Model)")
    ax.plot(time, x_y[:, 1], color='red', alpha=alpha, lw=lw, label="I (Model)")
    ax.plot(time, x_y[:, 2], color='blue', alpha=alpha, lw=lw, label="R (Model)")

    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    ax.grid()
    return ax


def noisy_SIR(S, I, R, index, noise):
    S_data, I_data, R_data = [], [], []
    for i in index:
        S_data.append(S[i] + noise*np.random.normal(0, 1))
        I_data.append(I[i])
        R_data.append(R[i])
    return S_data, I_data, R_data


# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return odeint(func=derivative_SIR, y0=theta[-3:], t=[50, 70, 100, 120, 150, 180, 250], args=(*theta[:2],))


# https://www.pymc.io/projects/examples/en/latest/ode_models/ODE_Lotka_Volterra_multiple_ways.html
class inference_SIR:
    def __init__(self, S, I, R, t):
        self.S = S
        self.I = I
        self.R = R
        self.time = np.arange(0, 300, 1)
        self.df = pd.DataFrame(dict(
            step=t,
            S=self.S,
            I=self.I,
            R=self.R
        ))

    def init_params(self):
        logbeta, v, S_0, I_0, R_0 = -8, 0.02, 1e7, 1000, 0
        self.theta = np.array([logbeta, v, S_0, I_0, R_0])

    def init_ode_plot(self):
        self.init_params()
        x_y = odeint(func=derivative_SIR, y0=self.theta[-3:], t=self.time, args=(*self.theta[:2], ))
        _, ax = plt.subplots()
        self.plot_data(ax, lw=0)
        plot_model(ax, x_y, self.time, title="LV model")

    def ode_model_resid(self, theta):
        return (
            self.df[["S", "I", "R"]] - odeint(func=derivative_SIR, y0=theta[-3:],
                                                   t=self.df.step, args=(*theta[:2], ))
        ).values.flatten()

    def least_squares_pred(self):
        results = least_squares(self.ode_model_resid, x0=self.theta)
        self.ls_theta = results.x
        self.return_thetas()
        self.ls_theta = self.theta
        x_y = odeint(func=derivative_SIR, y0=self.ls_theta[-3:], t=self.time, args=(*self.ls_theta[:2], ))
        fig, ax = plt.subplots()
        self.plot_data(ax, lw=0)
        plot_model(ax, x_y, self.time, title="Least squares SIR model")

    def plot_data(self, ax, lw=2, title="SIR model data"):
        ax.plot(self.df.step, self.S, color='black', lw=lw, marker="+", markersize=12, label="S (Data)")
        ax.plot(self.df.step, self.I, color='red', lw=lw, marker="+", markersize=12, label="I (Data)")
        ax.plot(self.df.step, self.R, color='blue', lw=lw, marker="+", markersize=12, label="R (Data)")
        ax.set_xlim([0, 300])
        ax.set_xlabel("Step", fontsize=14)
        ax.set_ylabel("Population size", fontsize=14)
        ax.set_title(title, fontsize=16)
        return ax

    def return_thetas(self):
        print([t for t in self.theta])
        print([t for t in self.ls_theta])

    def infer(self):
        az.style.use("arviz-whitegrid")
        theta = self.ls_theta
        with pm.Model() as model:
            # Priors
            logbeta = pm.TruncatedNormal("logbeta", mu=theta[0], sigma=0.1, lower=-8.5, upper=-7.5, initval=theta[0])
            # v = pm.TruncatedNormal("v", mu=theta[1], sigma=0.01, lower=0, upper=0.4, initval=theta[1])
            S_0 = pm.LogNormal("S_0", mu=np.log(theta[2]), sigma=0.01, initval=np.log(theta[2]))
            # I_0 = pm.TruncatedNormal("I_0", mu=theta[3], sigma=1000, lower=0, upper=2000, initval=theta[3])
            # R_0 = pm.TruncatedNormal("R_0", mu=theta[4], sigma=100, lower=-10, upper=1000, initval=theta[4])

            sigma = pm.HalfNormal("sigma", 5)
            # Ode solution function
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([logbeta, v, S_0, I_0, R_0])
                )
            # Likelihood
            pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=self.df[["S", "I", "R"]].values)

        vars_list = list(model.values_to_rvs.keys())[:-1]
        sampler = "DEMetropolisZ"
        tune = draws = 2000
        with model:
            trace_DEMZ = pm.sample(step=[pm.DEMetropolis(vars_list)], tune=tune, draws=draws)
        trace = trace_DEMZ
        az.summary(trace)
        az.plot_trace(trace, kind="rank_bars")
        plt.suptitle(f"Trace Plot {sampler}")
        az.style.use("default")
        fig, ax = plt.subplots()
        self.plot_inference(ax, trace,
                            title=f"Data and Inference Model Runs\n{sampler} Sampler")

    def plot_model_trace(self, ax, trace_df, row_idx, lw=1, alpha=0.2):
        # cols = ["logbeta", "v", "S_0", "I_0", "R_0"]
        cols = ["logbeta", "S_0"]
        row = trace_df.iloc[row_idx, :][cols].values
        theta = [row[0], v, row[1], I_0, R_0]
        x_y = odeint(func=derivative_SIR, y0=theta[-3:], t=self.time, args=(*theta[:2],))
        plot_model(ax, x_y, time=self.time, lw=lw, alpha=alpha)

    def plot_inference(self, ax, trace, num_samples=25,
                       title="Inference Model", plot_model_kwargs=dict(lw=1, alpha=0.2)):
        trace_df = az.extract(trace, num_samples=num_samples).to_dataframe()
        self.plot_data(ax, lw=0)
        for row_idx in range(num_samples):
            self.plot_model_trace(ax, trace_df, row_idx, **plot_model_kwargs)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:6], labels[:6], loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(title, fontsize=16)


if __name__ == "__main__":
    logbeta = -8
    v = 0.02
    S_0 = 1e7
    I_0 = 1000
    R_0 = 0

    time = np.arange(0, 300, 1)
    sol = odeint(derivative_SIR, [S_0, I_0, R_0], time, args=(logbeta, v))

    times = [50, 70, 100, 120, 150, 180, 250]

    S_data, I_data, R_data = noisy_SIR(S=sol[:, 0], I=sol[:, 1], R=sol[:, 2],
                                       index=times,
                                       noise=100000)

    inf = inference_SIR(S=S_data, I=I_data, R=R_data, t=times)
    inf.init_ode_plot()
    inf.least_squares_pred()
    inf.infer()
