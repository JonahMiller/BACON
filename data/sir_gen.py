import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from statistics import fmean, stdev


def rbf_kernel(x1, x2, lengthScale, varSigma):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    K = varSigma*np.exp(-np.power(d, 2)/lengthScale)
    return K


def gen_meshgrid():
    beta = (np.linspace(-8.5, -7.5, 10))
    S_0 = (np.linspace(6.75, 7.25, 10))
    nu = np.linspace(0.15, 0.25, 10)
    X1, X2, X3 = np.meshgrid(beta, S_0, nu)
    X = np.stack((X1.flatten(), X2.flatten(), X3.flatten()), axis=-1)
    return X


def gen_data(samples, beta_noise=0, S_0_noise=0, nu_noise=0):
    beta = np.random.normal(-8, beta_noise, samples)
    S_0 = np.random.normal(7, S_0_noise, samples)
    nu = np.random.normal(0.2, nu_noise, samples)
    R_0 = beta*S_0/nu
    X = np.stack((beta, S_0, nu), axis=-1)
    return X, R_0


def gp_prediction(x_data, y_data, x_grid, lengthScale, varSigma):
    k_starX = rbf_kernel(x_grid, x_data, lengthScale, varSigma)
    k_xx = rbf_kernel(x_data, None, lengthScale, varSigma)
    k_starstar = rbf_kernel(x_grid, None, lengthScale, varSigma)
    mu = k_starX.dot(np.linalg.inv(k_xx)).dot(y_data)
    var = k_starstar - (k_starX).dot(np.linalg.inv(k_xx)).dot(k_starX.T)
    return mu, var, x_grid


if __name__ == "__main__":
    X_train = gen_meshgrid()
    X_real, y_real = gen_data(5, beta_noise=0.1, S_0_noise=0.1, nu_noise=0.1)
    mu_star, var_star, x_star = gp_prediction(X_real, y_real, X_train, 0.5, 0.1)

    best_std = 10000
    best_mean = 0
    best_mu = 0
    best_var = 0

    for _ in range(100):
        f_star = np.random.multivariate_normal(mu_star, var_star, 1)

        mean, std = fmean(f_star.flatten()), stdev(f_star.flatten())

        if std < best_std:
            best_std = std
            best_mean = mean
            best_mu = mu_star
            best_var = var_star

    print(best_mean, best_std)
    print(best_mu)
    print(best_var)
