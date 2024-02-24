import pandas as pd
import numpy as np
from statistics import fmean
from sympy import Eq, lambdify

from bacon.bacon1 import BACON_1
import bacon.losses as bl


def run_bacon_1(df, col_1, col_2, verbose=False):
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
    bacon_1_instance = BACON_1(df[[col_1, col_2]], bacon_1_info=verbose)
    return bacon_1_instance.bacon_iterations()


class BACON_5_dummy:
    """
    When a dummy variable is found, it needs to be traced up the chain to find its
    relationship with previous independent variables before being replaced below.
    """
    def __init__(self, df, dummy_var, bacon_1_info=False, bacon_5_info=False):
        self.df = df
        self.dummy_var = dummy_var
        self.bacon_1_info = bacon_1_info
        self.bacon_5_info = bacon_5_info

    def run_bacon_5_from_dummy(self):
        """
        Runs BACON_5 up the dummy chain to solve the variable
        """
        bacon_5_instance = BACON_5(self.df, bacon_1_info=self.bacon_1_info, bacon_5_info=self.bacon_5_info)
        bacon_5_instance.bacon_iterations()

    def return_dummy_relation(self):
        ...


class BACON_5:
    """
    BACON.5 assumes symmetry in the relationship to narrow down the processing time.
    It allows additional benefits such as noise resistance and machine learning based
    pruning.
    """
    def __init__(self, initial_df, bacon_1_info=False, bacon_5_info=False):
        self.initial_df = initial_df
        self.dfs = [initial_df]
        self.delta = 0.01
        self.bacon_1_info = bacon_1_info
        self.bacon_5_info = bacon_5_info
        self.eqns = []

    def get_smaller_df(self, df):
        column_vars = df.columns.tolist()[:-2]
        for var in column_vars:
            df = df.loc[df[var] == min(df[var].unique())]
        return df.drop_duplicates(df.columns.tolist()[:-1])

    def new_df_col(self, eqn):
        vars = eqn.free_symbols
        f = lambdify([tuple(vars)], eqn)
        new_col = np.array([(f(tuple(val))) for val in self.initial_df[list(vars)].to_numpy().tolist()])
        return pd.DataFrame({eqn: new_col})

    def solve_for_dummy(self):
        ...

    def bacon_iterations(self):
        """
        Manages the iterations over all the layers in a for loop until each dataframe
        only has two columns left.
        """
        while self.not_last_iteration():
            new_dfs = []
            self.check_const_col()
            for df in self.dfs:
                small_df = self.get_smaller_df(df)

                results = run_bacon_1(small_df, small_df.columns[-1], small_df.columns[-2],
                                      verbose=self.bacon_1_info)

                # Special check for linear relationship added to dataframe
                if isinstance(results[2], list):
                    dummy_instance = BACON_5_dummy(df, results[2][1])
                    dummy_val = dummy_instance.return_dummy_relation()
                    new_symbol = self.solve_for_dummy(results[2][2], dummy_val)
                else:
                    # Save results as new column for dataframe with correct indecies
                    new_symbol = results[1]

                new_col = self.new_df_col(new_symbol)
                df = df.iloc[:, :-2].join(new_col)
                new_dfs.append(df)

            self.dfs = new_dfs

        constants = []
        for df in self.dfs:
            df = df.drop_duplicates(df.columns.tolist()[:-1])
            # When only 2 columns left do simple Bacon 1

            if self.bacon_5_info:
                print(f"BACON 5: Running BACON 1 on final variables [{df.columns[0]}, {df.columns[1]}]")

            results = run_bacon_1(df, df.columns[0], df.columns[1], verbose=self.bacon_1_info)

            if self.bacon_5_info:
                print(f"BACON 5: {results[1]} is constant at {fmean(results[0])}")
            constants.append(results[1])
            self.eqns.append(Eq(results[1], fmean(results[0])))

        self.bacon_losses()

    def bacon_losses(self):
        const_eqns = self.eqns
        key_var = self.initial_df.columns[-1]

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("The constant equations found are:")
        for eqn in const_eqns:
            print(f"{eqn.rhs} = {eqn.lhs}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        eqn = bl.simplify_eqns(self.initial_df, const_eqns, key_var).iterate_through_dummys()
        loss = bl.loss_calc(self.initial_df, eqn).loss()
        print(f"Final form is {eqn.rhs} = {eqn.lhs} with loss {loss}.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def not_last_iteration(self):
        for df in self.dfs:
            if len(df.columns) > 2:
                return True
        return False

    def print_dfs(self):
        for df in self.dfs:
            print(df)

    def check_const_col(self):
        """
        Checks if there are fixed variables in the columns, these may be from the linearity
        relationship or just being found when initialised. Should protect against data being
        put in different orders.
        """
        for i, df in enumerate(list(self.dfs)):
            temp_dict = df.to_dict("list")
            for idx, val in temp_dict.items():
                mean = fmean(val)
                M = abs(mean)
                if all(M*(1 - self.delta) < abs(v) < M*(1 + self.delta) for v in val):
                    if self.bacon_5_info:
                        print(f"BACON 5: {idx} is constant at {mean}")
                    self.eqns.append(Eq(idx, mean))
                    del self.dfs[i]
