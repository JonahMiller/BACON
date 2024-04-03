import numpy as np
import jax as jnp
from scipy.spatial.distance import cdist
import GPy

import pandas as pd
import data.datasets as data
# import datasets as data


def rbf_kernel(x1, x2, length, var):
    if x2 is None:
        return var*np.exp(-np.power(cdist(x1, x1), 2)/length)
    else:
        return var*np.exp(-np.power(cdist(x1, x2), 2)/length)


def white_kernel(x1, x2, var):
    if x2 is None:
        return var*np.eye(x1.shape[0])
    else:
        return np.zeros((x1.shape[0], x2.shape[0]))


def squared_exponential(x1, x2, var, length, noise_var):
    if x2 is None:
        return var*jnp.exp(-cdist(x1, x1, metric='sqeuclidean')/length**2) \
               + (1/noise_var)*jnp.eye(x1.shape[0])
    else:
        return var*jnp.exp(-cdist(x1, x2, metric='sqeuclidean')/length**2)


class gp_pred:
    def __init__(self, x_train, x_real, y_real, length, var):
        self.x_grid = x_train
        self.x_data = x_real
        self.y_data = y_real
        self.kernel = rbf_kernel
        self.length = length
        self.var = var

    def gp_prediction(self, kernel):
        k_starX = kernel(self.x_grid, self.x_data, self.length, self.var)
        k_xx = kernel(self.x_data, None, self.length, self.var)
        k_starstar = kernel(self.x_grid, None, self.length, self.var)
        mu = k_starX.dot(np.linalg.inv(k_xx)).dot(self.y_data)
        var = k_starstar - (k_starX).dot(np.linalg.inv(k_xx)).dot(k_starX.T)
        return mu, var

    def gp_prediction_with_noise(self, x1, y1, xstar, noise, kernel, **args):
        k_starX = kernel(xstar, x1, **args)
        k_xx = kernel(x1, None, **args)
        k_starstar = kernel(xstar, None, **args)
        mu = k_starX.dot(np.linalg.inv(k_xx + noise*np.identity(len(x1)))).dot(y1)
        var = k_starstar - (k_starX).dot(np.linalg.inv(k_xx + noise*np.identity(len(x1)))).dot(k_starX.T)
        return mu, var, xstar

    def main(self):
        mu, var = self.gp_prediction(self.kernel)
        f_ = np.random.multivariate_normal(mu, var, 100)
        mean = [np.mean(f) for f in f_.T]
        return mean


def denoise(X, y):

    k = GPy.kern.RBF(input_dim=3, variance=1000, lengthscale=10) + GPy.kern.White(input_dim=3, variance=0.001)
    m = GPy.models.GPRegression(X, y, k)
    m.constrain_positive('')  # '' is a regex matching all parameter names
    m.optimize()

    return X, m.predict(X)


class ranking:
    def __init__(self, init_df, kernel=None):
        self.init_df = init_df
        init_dim = len(self.init_df.columns) - 1
        if not kernel:
            self.kernel = GPy.kern.RBF(input_dim=init_dim, variance=1000, lengthscale=10) + \
                          GPy.kern.White(input_dim=init_dim, variance=0.001)
        else:
            self.kernel = kernel

    def sort_df(self):
        X = np.array(self.init_df.iloc[:, :-1])
        y = np.array(self.init_df.iloc[:, -1])
        y = y.reshape((len(self.init_df.index), 1))
        return X, y

    def gp_denoiser(self):
        X, y = self.sort_df()
        m = GPy.models.GPRegression(X, y, self.kernel)
        m.constrain_positive('')  # '' is a regex matching all parameter names
        m.optimize()
        print(m[0]/m[3])
        return X, y, m.predict(X)

    def rank_new_df(self):
        X, y, pred = self.gp_denoiser()
        # print(self.init_df)
        # seq = sorted((y - pred[0]))
        # index = np.array([seq.index(v) + 1 for v in (y - pred[0])])
        # ranked_frame = pd.DataFrame({"rank": index})
        # last_col = list(self.init_df.columns.values)[-1]
        # df = self.init_df.iloc[:, :-1].join(pd.DataFrame({last_col: pred[0].flatten()}))
        # return df, ranked_frame


if __name__ == "__main__":
    data_func = data.allowed_data()["ideal"]
    init_data, init_symb = data_func(noise=0)
    initial_df = pd.DataFrame({v: d for v, d in zip(init_symb, init_data)})

    r = ranking(initial_df)
    df, ranked_frame = r.rank_new_df()
    # print(df)
    # print(ranked_frame)
