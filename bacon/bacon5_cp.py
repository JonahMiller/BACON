import pandas as pd
import numpy as np
from statistics import fmean
from sympy import Eq, lambdify,  Symbol, simplify
from itertools import islice

from bacon.bacon1 import BACON_1
import bacon.losses as bl


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


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


class BACON_5:
    """
    BACON.5 assumes symmetry in the relationship to narrow down the processing time.
    It allows additional benefits such as noise resistance and machine learning based
    pruning.
    """
    def __init__(self, initial_df, bacon_1_info=False, bacon_5_info=False):
        self.initial_df = initial_df
        self.delta = 0.01
        self.bacon_1_info = bacon_1_info
        self.bacon_5_info = bacon_5_info
        self.eqns = []
        self.found_exprs = []
        self.iteration_level = 1

    def dataframe_manager(self):
        '''
        Chooses which dataframe values to use for BACON testing.

        TODO: Use maching learning here.
        '''
        init_df = pd.DataFrame(self.initial_df, index=[0, 1, 2, 4, 8, 13, 26, 40, 80])
        # init_df = pd.DataFrame(self.initial_df, index=[0, 1, 2, 4, 8, 13, 26])
        self.dfs = [init_df]

    def get_smaller_df(self, df):
        """
        Get dataframe subset to get rule for components.
        """
        column_vars = df.columns.tolist()[:-2]
        for var in column_vars:
            df = df.loc[df[var] == min(df[var].unique())]
        return df.drop_duplicates(df.columns.tolist()[:-1])

    def new_df_col(self, expr, current_df):
        """
        Creates a new dataframe column based on expression found by smaller df.
        Notation fixes to allow expressions to be substituted.
        """
        # Simplify column names deterministically for sympy to detect equivalent equations
        current_df.columns = [*current_df.columns[:-1], simplify(current_df.columns.tolist()[-1])]

        indices = current_df.index.values
        for col_name in current_df.columns.tolist():
            if len(col_name.free_symbols) > 1:
                temp_df = current_df.rename(columns={col_name: Symbol("z")})
                new_expr = expr.subs(col_name, Symbol("z"))
            else:
                new_expr = expr
                temp_df = current_df

        vars = temp_df.columns.tolist()
        f = lambdify([tuple(vars)], new_expr)
        new_col = np.array([(f(tuple(val))) for val in temp_df[list(vars)].to_numpy().tolist()])
        return pd.DataFrame({expr: new_col}, index=indices)

    def update_df_with_single_expr(self, expression, df):
        """
        Removes last 2 columns of df and replace with expression replacing
        these 2 columns.
        """
        new_col = self.new_df_col(expression, df)
        df = df.iloc[:, :-2].join(new_col)
        return df

    def update_df_with_multiple_expr(self, expression1, expression2, df):
        new_col_1 = self.new_df_col(expression1, df)
        new_col_2 = df.loc[:, [expression2]]
        new_cols = new_col_1.join(new_col_2)
        df = df.iloc[:, :-2].join(new_cols)
        return df

    def generate_backup_df(self, df_idx, iteration_level):
        indecies = []
        init_idx = len(self.initial_df.index.values)

        groups = list(chunk(range(init_idx), 3))

        if iteration_level == 1:
            for idx in df_idx:
                for group in groups:
                    if idx in group:
                        lgroup = list(group)
                        lgroup.remove(idx)
                        indecies.append(min(lgroup))

        elif iteration_level == 2:
            groups = list(chunk(groups, 3))
            for idx in df_idx:
                for group in groups:
                    if idx in list(sum(group, ())):
                        for grou in group:
                            if idx in grou:
                                lgroup = list(group)
                                lgroup.remove(grou)
                                indecies.append(min(list(sum(lgroup, ()))))

        elif iteration_level == 3:
            groups = list(chunk(groups, 9))
            for idx in df_idx:
                for group in groups:
                    if idx in list(sum(group, ())):
                        for grou in group:
                            if idx in grou:
                                lgroup = list(group)
                                lgroup.remove(grou)
                                indecies.append(min(list(sum(lgroup, ()))))

        backup_df = pd.DataFrame(self.initial_df, index=indecies)
        return backup_df

    def update_backup_df(self, backup_df, found_exprs, df_count):
        exprs = [expr for expr in found_exprs if expr[-1] == df_count]
        if exprs:
            for expr in exprs:
                if len(expr) == 3:
                    backup_df = self.update_df_with_single_expr(expr[0], backup_df)
                elif len(expr) == 6:
                    backup_backup_df = self.generate_backup_df(backup_df.index.values, expr[4])
                    backup_backup_df = self.update_df_with_multiple_expr(expr[2], expr[3], backup_backup_df)
                    backup_df = self.update_df_with_multiple_expr(expr[2], expr[3], backup_df)
                    dummy_col, expr_col = self.get_linear_values(backup_df, backup_backup_df,
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
            self.check_const_col()

            df_count = 0
            for df in self.dfs:

                small_df = self.get_smaller_df(df)
                indecies = small_df.index.values

                results = run_bacon_1(small_df, small_df.columns[-1], small_df.columns[-2],
                                      verbose=self.bacon_1_info)

                # Special check for linear relationship added to dataframe
                if isinstance(results[2], list):
                    small_dummy_df = pd.DataFrame({results[2][1]: results[2][3]}, index=indecies)
                    small_expr_df = pd.DataFrame({results[2][2]: results[0]})

                    extra_vals_df = df.drop(index=indecies)

                    backup_df = self.generate_backup_df(extra_vals_df.index.values, self.iteration_level)
                    backup_df = self.update_backup_df(backup_df, self.found_exprs, df_count)

                    extra_vals_df = self.update_df_with_multiple_expr(results[2][4], results[2][5], extra_vals_df)
                    backup_df = self.update_df_with_multiple_expr(results[2][4], results[2][5], backup_df)

                    dummy_col, expr_col = self.get_linear_values(extra_vals_df, backup_df,
                                                                 results[2][1], results[2][2])

                    n_df1 = df.iloc[:, :-2].join(pd.concat((small_dummy_df, dummy_col)))
                    # b_df1 = backup_df.iloc[:, :-2].join(dummy_col.set_index(backup_df.index.copy()))

                    n_df2 = df.iloc[:, :-2].join(pd.concat((small_expr_df, expr_col)))
                    # b_df2 = backup_df.iloc[:, :-2].join(expr_col.set_index(backup_df.index.copy()))

                    new_dfs.append(n_df1.drop(index=indecies[1:]))
                    new_dfs.append(n_df2.drop(index=indecies[1:]))

                    self.found_exprs.append([results[2][1], results[2][2],
                                             results[2][4], results[2][5],
                                             self.iteration_level,
                                             df_count])

                else:
                    # Save results as new column for dataframe with correct indecies
                    new_expression = results[1]
                    df = self.update_df_with_single_expr(new_expression, df)
                    self.found_exprs.append([new_expression,
                                             self.iteration_level,
                                             df_count])

                    # Update backup dfs
                    # backup_df = self.update_df_with_single_expr(new_expression, backup_df)

                    new_dfs.append(df.drop(index=indecies[1:]))
                df_count += 1

            self.iteration_level += 1
            self.dfs = new_dfs

        constants = []
        for df in self.dfs:
            # When only 2 columns left do simple Bacon 1

            if self.bacon_5_info:
                print(f"BACON 5: Running BACON 1 on final variables [{df.columns[0]}, {df.columns[1]}]")

            results = run_bacon_1(df, df.columns[0], df.columns[1], verbose=self.bacon_1_info)

            if self.bacon_5_info:
                print(f"BACON 5: {results[1]} is constant at {fmean(results[0])}")
            constants.append(results[1])
            self.eqns.append(Eq(results[1], fmean(results[0])))

        self.bacon_losses()

    def get_linear_values(self, df, backup_df, dummy_sym, expr_sym):
        """
        Uses a backup dataframe to find values for linear relationships.
        """
        indecies = df.index.values

        data1 = [[v1, v2] for v1, v2 in zip(df.iloc[:, -1].values.tolist(),
                                            backup_df.iloc[:, -1].values.tolist())]
        data2 = [[v1, v2] for v1, v2 in zip(df.iloc[:, -2].values.tolist(),
                                            backup_df.iloc[:, -2].values.tolist())]

        expr_data, dummy_data = [], []
        for x, y in zip(data1, data2):
            m, c = np.polyfit(x, y, 1)
            dummy_data.append(m)
            expr_data.append(c)
        return pd.DataFrame({dummy_sym: dummy_data}, index=indecies), \
            pd.DataFrame({expr_sym: expr_data}, index=indecies)

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
