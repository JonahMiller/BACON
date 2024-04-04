import pandas as pd
from statistics import fmean
from sympy import Eq

from space_of_laws.bacon1 import BACON_1
from utils import df_helper as df_helper
from utils.gp import ranking


def run_bacon_1(df, col_1, col_2, delta, epsilon, verbose=False):
    """
    Runs an instance of BACON.1 on the specified columns
    col_1 and col_2 in the specified dataframe df.
    """
    if verbose:
        unused_df = df.iloc[:, :-2]
        col_names = unused_df.columns.tolist()
        col_ave = [unused_df.loc[:, name].mean() for name in col_names]
        if len(col_names) != 0:
            print(f"BACON 1: Running BACON 1 on variables [{col_1}, {col_2}] and")
            print(f"         unused variables {col_names} set as {col_ave}.")
        else:
            print(f"BACON 1: Running BACON 1 on variables [{col_1}, {col_2}]")
    bacon_1_instance = BACON_1(df[[col_1, col_2]],
                               epsilon, delta,
                               bacon_1_info=verbose)
    return bacon_1_instance.bacon_iterations()


class layer:
    """
    BACON.3 can be thought of a layer-by-layer running of BACON.1 with
    previous variable fixes. This class runs each layer instance.
    """
    def __init__(self, df, delta, epsilon, bacon_1_info=False):
        self.df = df
        self.delta = delta
        self.epsilon = epsilon
        self.bacon_1_info = bacon_1_info

    def find_exprs(self):
        """
        Runs the BACON.1 iterations over the dataframe found above.
        if there is a list returned it means a linear relationship was found
        as the linear relationship value can be a variable it then creates two
        instances, one with the y-intercept and the other with the gradient of
        the linear relationship. It returnse the new columns found.
        """
        exprs_found = {}
        lin_relns = {}

        s_dfs = df_helper.deconstruct_df(self.df)

        for df in s_dfs:
            # Perform Bacon.1 on last 2 columns in system
            data, symb, lin = run_bacon_1(df, df.columns[-1], df.columns[-2],
                                          self.delta, self.epsilon,
                                          verbose=self.bacon_1_info)

            if isinstance(lin, list):
                symb = lin[2]

            if symb in exprs_found:
                exprs_found[symb] += 1
            else:
                exprs_found[symb] = 1
                if isinstance(lin, list):
                    lin_relns[symb] = [lin[1], lin[2], lin[4], lin[5]]

        return exprs_found, lin_relns

    def construct_dfs(self):
        new_dfs = {}
        exprs, lin_relns = self.find_exprs()

        for expr in exprs:
            new_dfs[expr] = []

            if expr in lin_relns:
                df = df_helper.update_df_with_multiple_expr(lin_relns[expr][2],
                                                            lin_relns[expr][3],
                                                            self.df)
                s_dfs = df_helper.deconstruct_df(df)

                new_dummy_col, new_expr_col = pd.DataFrame(), pd.DataFrame()

                for s_df in s_dfs:
                    dummy_col, expr_col = df_helper.linear_relns(s_df,
                                                                 lin_relns[expr][0],
                                                                 lin_relns[expr][1])
                    new_dummy_col = pd.concat([new_dummy_col, dummy_col])
                    new_expr_col = pd.concat([new_expr_col, expr_col])

                n_df1 = df.iloc[:, :-2].join(new_dummy_col)
                n_df2 = df.iloc[:, :-2].join(new_expr_col)

                new_dfs[expr] = [n_df1, n_df2]

            else:
                n_df = df_helper.update_df_with_single_expr(expr, self.df)
                new_dfs[expr] = [n_df]

        return new_dfs

    @staticmethod
    def rank_exprs(exprs_df_dict):
        best_ratio = 0
        for expr, dfs in exprs_df_dict.items():
            if len(dfs) == 2:
                s_n_ratio = min(ranking(dfs[0]).signal_noise_ratio(),
                                ranking(dfs[1]).signal_noise_ratio())
            else:
                s_n_ratio = ranking(dfs[0]).signal_noise_ratio()
            if s_n_ratio > best_ratio:
                best_ratio = s_n_ratio
                best_expr = expr
            print(expr, s_n_ratio)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return best_expr

    def run_single_iteration(self):
        expr_df_dict = self.construct_dfs()
        expr = layer.rank_exprs(expr_df_dict)
        return expr_df_dict[expr]


class RANKING_FORWARD:
    """
    Manages the layers of the dataframe, including the potentially new layers found
    when linear relationships are found. Then it runs BACON.1 on the those two columns.
    """
    def __init__(self, initial_df, delta=0.15, epsilon=0.001,
                 bacon_1_info=False, bacon_3_info=False):
        self.initial_df = initial_df
        self.dfs = [initial_df]
        self.delta = delta
        self.epsilon = epsilon
        self.bacon_1_info = bacon_1_info
        self.bacon_3_info = bacon_3_info
        self.eqns = []

    def bacon_iterations(self):
        """
        Manages the iterations over all the layers in a for loop until each dataframe
        only has two columns left.
        """
        while self.not_last_iteration():
            new_dfs = []

            self.dfs, self.eqns = df_helper.check_const_col(self.dfs, self.eqns,
                                                            self.delta, logging=False)

            for df in self.dfs:
                bacon_layer_in_context = layer(df, self.delta, self.epsilon, self.bacon_1_info)
                new_df = bacon_layer_in_context.run_single_iteration()
                new_dfs.extend(new_df)

                if self.bacon_3_info:
                    var1, var2 = df.columns[-1], df.columns[-2]
                    unused_df = df.iloc[:, :-2]
                    col_names = unused_df.columns.tolist()

                    print(f"BACON 3: Running BACON 1 on variables [{var1}, {var2}] and")
                    print(f"         keeping constant unused variables {col_names}")
                    print(f"         displayed fix variables {[df.columns[-1] for df in new_df]}.")

            self.dfs = new_dfs

            self.print_dfs()
            print(self.eqns)

            self.epsilon = self.epsilon*0.1
            self.delta += 0.02

        constants = []
        for df in self.dfs:
            # When only 2 columns left do simple Bacon 1

            if self.bacon_3_info:
                print(f"BACON 3: Running BACON 1 on final variables [{df.columns[0]}, {df.columns[1]}]")

            results = run_bacon_1(df, df.columns[0], df.columns[1],
                                  self.delta, self.epsilon,
                                  verbose=self.bacon_1_info)

            if self.bacon_3_info:
                print(f"BACON 3: {results[1]} is constant at {fmean(results[0])}")
            constants.append(results[1])
            self.eqns.append(Eq(results[1], fmean(results[0])))

        df_helper.score(self.initial_df, self.eqns)

    def not_last_iteration(self):
        for df in self.dfs:
            if len(df.columns) > 2:
                return True
        return False

    def print_dfs(self):
        for df in self.dfs:
            print(df)
