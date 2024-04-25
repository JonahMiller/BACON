import numpy as np
import GPy


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

    def signal_noise_ratio(self):
        X, y = self.sort_df()
        m = GPy.models.GPRegression(X, y, self.kernel)
        m.constrain_positive('')  # '' is a regex matching all parameter names
        m.optimize()

        if m[0]/m[3] > 1e100:
            return 1e100
        else:
            return m[0]/m[3]
