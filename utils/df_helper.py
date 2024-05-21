import pandas as pd
import numpy as np
from statistics import fmean
from sympy import Symbol, Eq, lambdify, simplify, factor

import utils.losses as loss_helper
from space_of_laws.laws_methods.bacon7_law import BACON_7_law


def new_eqn(mini_expr, temp_val, full_expr):
    """
    Disgusting solution to detect equivalent expressions using Sympy.
    """
    new_expr = full_expr.subs(mini_expr,
                              temp_val).subs(1/mini_expr,
                                             1/temp_val).subs(mini_expr**2,
                                                              temp_val**2).subs(1/mini_expr**2,
                                                                                1/temp_val**2)
    if temp_val in new_expr.free_symbols:
        return new_expr
    else:
        new_expr = simplify(full_expr).subs(mini_expr,
                                            temp_val).subs(1/mini_expr,
                                                           1/temp_val).subs(mini_expr**2,
                                                                            temp_val**2).subs(1/mini_expr**2,
                                                                                              1/temp_val**2)
        if temp_val in new_expr.free_symbols:
            return new_expr
        else:
            new_expr = full_expr.subs(simplify(mini_expr),
                                      temp_val).subs(1/simplify(mini_expr),
                                                     1/temp_val).subs(simplify(mini_expr**2),
                                                                      temp_val**2).subs(1/simplify(mini_expr**2),
                                                                                        1/temp_val**2)
            if temp_val in new_expr.free_symbols:
                return new_expr
            else:
                new_expr = simplify(full_expr).subs(simplify(mini_expr),
                                                    temp_val).subs(1/simplify(mini_expr),
                                                                   1/temp_val).subs(simplify(mini_expr**2),
                                                                                    temp_val**2).subs(1/simplify(mini_expr**2),  # noqa
                                                                                                      1/temp_val**2)
                if temp_val in new_expr.free_symbols:
                    return new_expr
                else:
                    return full_expr


def new_df_col(expr, current_df):
    """
    Creates a new dataframe column based on expression found by smaller df.
    Notation fixes to allow expressions to be substituted.
    """
    penul_col_name = current_df.columns.tolist()[-2]
    last_col_name = current_df.columns.tolist()[-1]
    indices = current_df.index.values

    expr2 = new_eqn(penul_col_name, Symbol("d_0"), expr)
    expr3 = new_eqn(last_col_name, Symbol("d_1"), expr2)

    temp_df = current_df.rename(columns={penul_col_name: Symbol("d_0"),
                                         last_col_name: Symbol("d_1")})
    vars = temp_df.columns.tolist()
    f = lambdify([tuple(vars)], expr3)
    new_col = np.array([(f(tuple(val))) for val in temp_df[list(vars)].to_numpy().tolist()])
    return pd.DataFrame({simplify(expr): new_col}, index=indices)


def update_df_single_expr(expression, df):
    """
    Removes last 2 columns of df and replace with expression replacing
    these 2 columns.
    """
    new_col = new_df_col(expression, df)
    df = df.iloc[:, :-2].join(new_col)
    return df


def update_df_multiple_expr(expression1, expression2, df):
    """
    Removes last 2 columns of df and replace with expressions replacing
    these 2 columns.
    """
    col_names = df.columns.tolist()
    dep_var = col_names[-1]

    if expression1 in col_names and expression2 in col_names:
        if expression2 == dep_var:
            expr_idx1 = col_names.index(expression1)
            new_col1 = df.iloc[:, [expr_idx1]]
            expr_idx2 = col_names.index(expression2)
            new_col2 = df.iloc[:, [expr_idx2]]
        else:
            expr_idx1 = col_names.index(expression1)
            new_col2 = df.iloc[:, [expr_idx1]]
            expr_idx2 = col_names.index(expression2)
            new_col1 = df.iloc[:, [expr_idx2]]

    elif expression1 in col_names and expression2 not in col_names:
        expr_idx1 = col_names.index(expression1)
        new_col1 = df.iloc[:, [expr_idx1]]
        new_col2 = new_df_col(expression2, df)

    elif expression1 not in col_names and expression2 in col_names:
        expr_idx1 = col_names.index(expression2)
        new_col1 = df.iloc[:, [expr_idx1]]
        new_col2 = new_df_col(expression1, df)

    else:
        new_col1 = new_df_col(expression1, df)
        new_col2 = new_df_col(expression2, df)

    new_cols = new_col1.join(new_col2)
    df = df.iloc[:, :-2].join(new_cols)
    return df


def check_const_col(dfs, eqns, delta, logging):
    """
    Checks if the final column (of newly found variable) is fixed.
    """
    idx_delete = []
    for i, df in enumerate(list(dfs)):
        col = df.iloc[:, -1].values
        mean = fmean(col)
        M = abs(mean)

        if len([v for v in col
                if abs(v) < M*(1 - delta) or abs(v) > M*(1 + delta)]) \
                <= np.ceil(delta*len(col)/2):
            if logging:
                print(f"SCORE : {df.columns.tolist()[-1]} is constant at {mean}")
            eqns.append(Eq(df.columns.tolist()[-1], mean))
            idx_delete.append(i)
    for idx in sorted(idx_delete, reverse=True):
        del dfs[idx]
    return dfs, eqns


def deconstruct_df(df):
    """
    Creates the lowest level of dataframe to run BACON.1 on.
    Eg. if a dataframe of variables ABCD is fed in, it creates a
    dataframe for each combination of fixed A, B with variable C, D to
    find local patterns for C, D.
    """
    n_cols = len(df.columns)
    df_dicts = {n_cols: [df]}

    for i in range(n_cols, 2, -1):
        smaller_dfs = []
        for df in df_dicts[i]:
            for k in df[df.columns[n_cols - i]].unique():
                smaller_dfs.append(df[df[df.columns[n_cols - i]] == k])
        df_dicts[i - 1] = smaller_dfs
    smallest_dfs = df_dicts[min(df_dicts)]
    return smallest_dfs


def deconstruct_deconstructed_df(df):
    """
    Gets dataframe that shares all the same values in the
    penultimate column of a deconstructed df.
    """
    if len(df.index) == 3:
        return df
    else:
        col_0 = df.columns[-2]
        unique_vals = df[col_0].unique()
        small_dfs = []
        for val in unique_vals:
            mini_df = df.loc[df[col_0] == val]
            small_dfs.append(mini_df)
        return small_dfs


def average_df(df):
    if len(df.index) == 3:
        return df
    else:
        col_0, col_1 = df.columns[-2], df.columns[-1]
        unique_vals = df[col_0].unique()
        return_df = pd.DataFrame()
        for val in unique_vals:
            mini_df = df.loc[df[col_0] == val]
            n_df = mini_df.iloc[[0], :-1]
            n_df[col_1] = np.mean(mini_df[mini_df.columns[-1]])
            return_df = pd.concat((return_df, n_df))
        return return_df


def linear_relns(df, dummy_sym, expr_sym):
    indecies = df.index.values

    data1 = df.iloc[:, -1].values.tolist()
    data2 = df.iloc[:, -2].values.tolist()

    expr_data, dummy_data = [], []
    m, c = np.polyfit(data2, data1, 1)

    for d1, d2 in zip(data1, data2):
        dummy_data.append((d1 - c)/d2)
        expr_data.append(d1 - m*d2)

    return pd.DataFrame({dummy_sym: dummy_data}, index=indecies), \
        pd.DataFrame({expr_sym: expr_data}, index=indecies)


def lin_reln_2_df(df, backup_df, dummy_sym, expr_sym):
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


def not_last_iteration(dfs):
    for df in dfs:
        if len(df.columns) > 2:
            return True
    return False


def print_dfs(dfs):
    for df in dfs:
        print(df)


def score(init_df, eqns):
    key_var = init_df.columns[-1]
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("The constant equations found are:")
    for eqn in eqns:
        print(f"{eqn.rhs} = {eqn.lhs}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    eqn = loss_helper.simplify_eqns(init_df, eqns, key_var).iterate_through_dummys()
    loss = loss_helper.loss_calc(init_df, eqn).loss()
    print(f"Final form is {eqn.rhs} = {factor(eqn.lhs)} with loss {loss}.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def score_bacon_7(init_df, eqns):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("The constant equations found are:")
    for eqn in eqns:
        print(f"{eqn.rhs} = {eqn.lhs}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    eqn = BACON_7_law(init_df, eqns).run_iteration()
    loss = loss_helper.loss_calc(init_df, eqn).loss()
    print(f"Final form is {eqn.rhs} = {factor(eqn.lhs)} with loss {loss}.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# https://stackoverflow.com/a/13821695
def timeout(func, args=(), kwargs=None, timeout_duration=1, default=None):
    kwargs = kwargs or {}
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        result = default
    finally:
        signal.alarm(0)

    return result
