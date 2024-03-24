import numpy as np
import jax as jnp
from scipy.spatial.distance import cdist


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

    def main(self):
        mu, var = self.gp_prediction(self.kernel)
        f_ = np.random.multivariate_normal(mu, var, 100)
        mean = [np.mean(f) for f in f_.T]
        return mean


class gp_denoise:
    def __init__(self, x_real, y_real, length, var):
        self.x_data = x_real
        self.y_data = y_real
        self.length = length
        self.var = var

    def gp_prediction(self, kernel):
        k_starX = kernel(self.x_data, self.x_data, self.length, self.var)
        k_xx = kernel(self.x_data, None, self.length, self.var)
        k_starstar = kernel(self.x_data, None, self.length, self.var)
        mu = k_starX.dot(np.linalg.inv(k_xx)).dot(self.y_data)
        var = k_starstar - (k_starX).dot(np.linalg.inv(k_xx)).dot(k_starX.T)
        return mu, var

    def main(self):
        mu, var = self.gp_prediction()
        f_ = np.random.multivariate_normal(mu, var, 100)
        mean = [np.mean(f) for f in f_.T]
        return mean


class ranking:
    def __init__(self, df, length, var):
        pass
