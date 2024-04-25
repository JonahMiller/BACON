import pandas as pd
import numpy as np
from statistics import fmean
from sympy import Eq

from space_of_laws.laws_methods.bacon1 import BACON_1
from utils import df_helper as df_helper


np.random.seed(8)


def run_bacon_1(df, col_1, col_2, all_found_symbols,
                verbose=False, delta=0.1, epsilon=0.001):
    """
    Runs an instance of BACON.1 on the specified columns
    col_1 and col_2 in the specified dataframe df.
    """
    if verbose:
        unused_df = df.iloc[:, :-2]
        col_names = unused_df.columns.tolist()
        col_ave = [unused_df.loc[:, name].mean() for name in col_names]
        if len(col_names) != 0:
            print(f"Laws manager: Running BACON 1 on variables [{col_1}, {col_2}] and")
            print(f"              unused variables {col_names} set as {col_ave}.")
        else:
            print(f"Laws manager: Running BACON 1 on variables [{col_1}, {col_2}]")
    bacon_1_instance = BACON_1(df[[col_1, col_2]], all_found_symbols,
                               epsilon, delta,
                               verbose=verbose)
    return bacon_1_instance.bacon_iterations()


class BACON_5:
    """
    BACON.5 assumes symmetry in the relationship to narrow down the processing time.
    It allows additional benefits such as noise resistance and machine learning based
    pruning.
    """
    def __init__(self, initial_df,
                 epsilon=0.001, delta=0.01,
                 bacon_1_info=False, verbose=False):
        self.initial_df = initial_df
        self.epsilon = epsilon
        self.delta = delta
        self.bacon_1_info = bacon_1_info
        self.verbose = verbose
        self.eqns = []
        self.found_exprs = []
        self.iteration_level = 1
        self.symbols = list(sum(sym for sym in list(initial_df)).free_symbols)

    @staticmethod
    def initial_choosing(df):
        n_cols = len(df.columns)
        index_list = []
        for i in range(n_cols, 2, -1):
            unique_vals = df[df.columns[n_cols - i]].unique()
            best_val = np.random.choice(unique_vals)
            other_vals = np.delete(unique_vals, np.argwhere(unique_vals == best_val))
            for val in reversed(other_vals):
                row = df[df[df.columns[n_cols - i]] == val].sample(n=1)
                row_index = list(row.index.values)
                index_list.extend(row_index)
            df = df[df[df.columns[n_cols - i]] == best_val]
        last_indecies = list(df.index.values)
        index_list.extend(reversed(last_indecies))
        return list(reversed(index_list))

    @staticmethod
    def generate_backup_df(df, df_idx, iteration_level):
        n_cols = len(df.columns)
        col_names = list(df)[:n_cols - iteration_level]
        sym = ["=="]*(n_cols - iteration_level - 1) + ["!="]
        b_idx = []
        for idx in df_idx:
            vals = df.iloc[idx, :n_cols - iteration_level].values.tolist()
            query = " & ".join(f'{i} {j} {repr(k)}' for i, j, k in zip(col_names, sym, vals))
            b_idxs = df.query(query).index.values
            b_idx.append(np.random.choice(b_idxs))
        backup_df = pd.DataFrame(df, index=b_idx)
        return backup_df

    def dataframe_manager(self):
        '''
        Chooses which dataframe values to use for BACON testing.
        '''
        indecies = BACON_5.initial_choosing(self.initial_df)
        init_df = pd.DataFrame(self.initial_df, index=indecies)
        self.dfs = [init_df]

    def get_smaller_df(self, df):
        """
        Get dataframe subset to get rule for components.
        """
        column_vars = df.columns.tolist()[:-2]
        for var in column_vars:
            df = df.loc[df[var] == min(df[var].unique())]
        return df.drop_duplicates(df.columns.tolist()[:-1])

    def update_backup_df(self, backup_df, found_exprs, df_count):
        exprs = [expr for expr in found_exprs if expr[-1] == df_count]
        if exprs:
            for expr in exprs:
                if len(expr) == 3:
                    backup_df = df_helper.update_df_with_single_expr(expr[0], backup_df)
                elif len(expr) == 6:
                    backup_backup_df = BACON_5.generate_backup_df(self.initial_df,
                                                                  backup_df.index.values,
                                                                  expr[4])
                    backup_backup_df = df_helper.update_df_with_multiple_expr(expr[2], expr[3], backup_backup_df)
                    backup_df = df_helper.update_df_with_multiple_expr(expr[2], expr[3], backup_df)
                    dummy_col, expr_col = df_helper.lin_reln_2_df(backup_df, backup_backup_df,
                                                                      expr[0], expr[1])

                    b_df1 = backup_df.iloc[:, :-2].join(dummy_col)
                    b_df2 = backup_df.iloc[:, :-2].join(expr_col)

                    if df_count == 0:
                        backup_df = b_df1
                    elif df_count == 1:
                        backup_df = b_df2
                    else:
                        raise Exception

        return backup_df

    def bacon_iterations(self):
        """
        Manages the iterations over all the layers in a for loop until each dataframe
        only has two columns left.
        """
        self.dataframe_manager()
        while self.not_last_iteration():
            new_dfs = []
            df_helper.check_const_col(self.dfs, self.eqns, self.delta, self.verbose)

            df_count = 0
            for df in self.dfs:

                # small_df = self.get_smaller_df(df)
                small_df = df.iloc[:3, :]
                indecies = small_df.index.values

                results = run_bacon_1(small_df, small_df.columns[-1], small_df.columns[-2], self.symbols,
                                      epsilon=self.epsilon, delta=self.delta, verbose=self.bacon_1_info)

                # Special check for linear relationship added to dataframe
                if isinstance(results[2], list):
                    self.symbols.append(results[2][1])
                    small_dummy_df = pd.DataFrame({results[2][1]: results[2][3]}, index=indecies)
                    small_expr_df = pd.DataFrame({results[2][2]: results[0]})

                    extra_vals_df = df.drop(index=indecies)

                    backup_df = BACON_5.generate_backup_df(self.initial_df,
                                                           extra_vals_df.index.values,
                                                           self.iteration_level)
                    backup_df = self.update_backup_df(backup_df, self.found_exprs, df_count)

                    extra_vals_df = df_helper.update_df_with_multiple_expr(results[2][4], results[2][5], extra_vals_df)
                    backup_df = df_helper.update_df_with_multiple_expr(results[2][4], results[2][5], backup_df)

                    dummy_col, expr_col = df_helper.lin_reln_2_df(extra_vals_df, backup_df,
                                                                  results[2][1], results[2][2])

                    n_df1 = df.iloc[:, :-2].join(pd.concat((small_dummy_df, dummy_col)))
                    n_df2 = df.iloc[:, :-2].join(pd.concat((small_expr_df, expr_col)))

                    new_dfs.append(n_df1.drop(index=indecies[1:]))
                    new_dfs.append(n_df2.drop(index=indecies[1:]))

                    self.found_exprs.append([results[2][1], results[2][2],
                                             results[2][4], results[2][5],
                                             self.iteration_level,
                                             df_count])

                else:
                    # Save results as new column for dataframe with correct indecies
                    new_expression = results[1]
                    df = df_helper.update_df_with_single_expr(new_expression, df)
                    self.found_exprs.append([new_expression,
                                             self.iteration_level,
                                             df_count])

                    new_dfs.append(df.drop(index=indecies[1:]))
                df_count += 1

            self.iteration_level += 1
            self.dfs = new_dfs

        constants = []
        for df in self.dfs:
            # When only 2 columns left do simple Bacon 1

            if self.verbose:
                print(f"BACON 5: Running BACON 1 on final variables [{df.columns[0]}, {df.columns[1]}]")

            results = run_bacon_1(df, df.columns[0], df.columns[1], self.symbols,
                                  epsilon=self.epsilon, delta=self.delta, verbose=self.bacon_1_info)

            if self.verbose:
                print(f"BACON 5: {results[1]} is constant at {fmean(results[0])}")
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
