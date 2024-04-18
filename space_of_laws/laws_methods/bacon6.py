import numpy as np
from sympy import Symbol

import warnings
warnings.filterwarnings('ignore')


eta = Symbol("eta")
nu = Symbol("nu")


class BACON_6:
    def __init__(self, initial_df, step_size=1, N_threshold=2):
        self.init_symbols = list(initial_df)
        self.data = [initial_df[col_name] for col_name in self.init_symbols]
        self.symbols = [eta, nu]
        self.step_size = step_size
        self.N_threshold = N_threshold

        self.X = self.data[0]
        self.Y = self.data[1]

    def generate_initial_vars(self):
        vals = [-1, 0, 1]
        self.states = [np.array([x, y]) for x in vals for y in vals]

    def update_states(self):
        vals = [-self.step_size, 0, self.step_size]
        new_add_states = [np.array([x, y]) for x in vals for y in vals]
        n_states = []
        for state in self.states:
            new_states = [state + nas for nas in new_add_states]
            n_states.extend(new_states)
        self.states = n_states

    def calculate_approx(self):
        self.best = []
        state_dict = {}
        self.min = 0
        for idx, (a, b) in enumerate(self.states):
            Y_ = a*self.X**2 + b*self.X
            r = np.corrcoef(self.Y, Y_)[0, 1]
            if r not in state_dict:
                state_dict[r] = np.array([a, b])
                self.sort_threshold(idx, r)
        self.states = [state_dict[r] for r in self.best]
        self.step_size = self.step_size/2

    def sort_threshold(self, idx, r):
        if idx < self.N_threshold:
            self.best.append((r))
            self.min = min(self.best)
        else:
            if r > self.min:
                self.best.remove(self.min)
                self.best.append(r)
                self.min = min(self.best)

    def main(self):
        self.generate_initial_vars()
        for _ in range(10):
            self.calculate_approx()
            self.update_states()

        self.N_threshold = 1
        self.calculate_approx()
        print(self.states)
        self.output_variables()

    def output_variables(self):
        a, b = self.states[0]
        m, c = np.polyfit(a*self.X**2 + b*self.X, self.Y, 1)
        print(f"Y = {m}({a}X**2 + {b}X) + {c}")


if __name__ == "__main__":
    X = np.array([1, 3, 6, 10, 15])
    Y = 3*X**2 + 2*X + 1 + 0.1*np.random.normal(0, 1, 5)
    # Y = 3*X**2 + 2*X + 1
    bacon = BACON_6(X, Y)
    bacon.main()
