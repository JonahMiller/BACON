import numpy as np
from scipy.spatial.distance import cdist


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


class gp_pred:
    def __init__(self, x_train, x_real, y_real, length, var):
        self.x_grid = x_train
        self.x_data = x_real
        self.y_data = y_real
        self.kernel = self.rbf_kernel
        self.length = length
        self.var = var

    def rbf_kernel(self, x1, x2):
        if x2 is None:
            d = cdist(x1, x1)
        else:
            d = cdist(x1, x2)
        K = self.var*np.exp(-np.power(d, 2)/self.length)
        return K

    def gp_prediction(self, kernel):
        k_starX = kernel(self.x_grid, self.x_data)
        k_xx = kernel(self.x_data, None)
        k_starstar = kernel(self.x_grid, None)
        mu = k_starX.dot(np.linalg.inv(k_xx)).dot(self.y_data)
        var = k_starstar - (k_starX).dot(np.linalg.inv(k_xx)).dot(k_starX.T)
        return mu, var

    def main(self):
        mu, var = self.gp_prediction(self.kernel)
        f_ = np.random.multivariate_normal(mu, var, 100)
        mean = [np.mean(f) for f in f_.T]
        return mean


if __name__ == "__main__":
    X_train = gen_meshgrid()
    X_real, y_real = gen_data(5, beta_noise=0.2, S_0_noise=0.2, nu_noise=0.002)
    gp = gp_pred(X_train, X_real, y_real, 1, 0.1)
    ave_vals = gp.main()

    for a, b in zip(ave_vals, X_train):
        print(a, b)
