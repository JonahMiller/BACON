import datasets
from gp import gp_pred

import pandas as pd
import numpy as np
from itertools import product

import sys
sys.path.append("..")
from bacon.bacon5 import BACON_5


class fix_df:
    """
    Uses a GP to identify and fill missing points in a dataset.
    BACON needs minimum 3 points for each variable to be permuted wich each 
    3 points of all other variables.
    """
    def __init__(self, df, length, var):
        self.df = df
        self.indep_df = df.iloc[:, :-1]
        self.length = length
        self.var = var

    def identify_all_points(self):
        """
        Identifies all variables to find, and how many points are currently in
        existance.
        """
        vars = self.df.columns.tolist()[:-1]
        self.vars_dict = {var: self.df[var].unique() for var in vars}

        self.points = np.empty((0, len(vars)))
        self.values = np.array([])
        self.grid_points = np.empty((0, len(vars)))

    def identify_missing(self):
        """
        Creates new dataframe filled with missing points.
        """
        keys, values = zip(*self.vars_dict.items())
        full_perms = [v for v in product(*values)]
        for perm in full_perms:
            if (self.indep_df == perm).all(1).any():
                idx = self.indep_df.loc[(self.indep_df == perm).all(axis=1)].index[0]
                self.values = np.append(self.values, self.df.iloc[:, -1].loc[[idx]])
                self.points = np.vstack([self.points, perm])
            else:
                self.grid_points = np.vstack([self.grid_points, perm])

    def gp_predictions(self):
        """
        Creates predictions for the missing points based on what points exist (and
        their associated values).
        """
        gp = gp_pred(self.grid_points, self.points, self.values, self.length, self.var)
        self.best_approx = gp.main()

    def remake_df(self):
        """
        Adds the newly found values back to the dataframe.
        """
        vars = self.df.columns.tolist()
        new_vals = {vars[-1]: self.best_approx}
        for idx, var in enumerate(vars[:-1]):
            vals = [grid_point[idx] for grid_point in self.grid_points]
            new_vals[var] = vals
        new_df = pd.DataFrame(new_vals)
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def sort_df(self):
        """
        Rebuilds the dataframe in the correct format to be used by BACON.
        """
        cols = len(self.df.columns)
        df_dicts = {cols: [self.df]}
        for i in range(cols, 1, -1):
            smaller_dfs = []
            for df in df_dicts[i]:
                for k in np.sort(df[df.columns[cols - i]].unique()):
                    smaller_dfs.append(df[df[df.columns[cols - i]] == k])
            df_dicts[i - 1] = smaller_dfs
        smallest_dfs = df_dicts[min(df_dicts)]

        final_df = pd.DataFrame()
        for df in smallest_dfs:
            ndf = pd.DataFrame(df)
            final_df = pd.concat([final_df, ndf], ignore_index=True)
        self.df = final_df

    def fix_df_process(self):
        """
        Runs the program in the correct order.
        """
        self.identify_all_points()
        self.identify_missing()
        self.gp_predictions()
        self.remake_df()
        self.sort_df()
        return self.df


if __name__ == "__main__":
    data_func = datasets.allowed_data()["ideal"]
    var, data = data_func(noise=0)
    df1 = pd.DataFrame({v: d for v, d in zip(data, var)})

    sample = df1.sample(n=20)

    fix = fix_df(sample, 20000, 10)
    df = fix.fix_df_process()

    # print(df)
    # print(df1)

    bacon = BACON_5(df, bacon_5_info=True)
    bacon.bacon_iterations()
