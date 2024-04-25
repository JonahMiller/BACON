import pandas as pd
import numpy as np
from statistics import fmean
from sympy import Symbol, Eq, lambdify, simplify, factor

import utils.losses as loss_helper


def new_df_col(expr, current_df):
    """
    Creates a new dataframe column based on expression found by smaller df.
    Notation fixes to allow expressions to be substituted.
    """
    # Simplify column names deterministically for sympy to detect equivalent equations
    current_df.columns = [*current_df.columns[:-1], simplify(current_df.columns.tolist()[-1])]
    indices = current_df.index.values
    last_col_name = current_df.columns.tolist()[-1]
    temp_df = current_df.rename(columns={last_col_name: Symbol("d_0")})
    expr2 = simplify(expr).subs(last_col_name, Symbol("d_0"))
    expr3 = expr2.subs(1/last_col_name, 1/Symbol("d_0"))
    vars = temp_df.columns.tolist()
    f = lambdify([tuple(vars)], expr3)
    new_col = np.array([(f(tuple(val))) for val in temp_df[list(vars)].to_numpy().tolist()])
    return pd.DataFrame({expr: new_col}, index=indices)


def update_df_with_single_expr(expression, df):
    """
    Removes last 2 columns of df and replace with expression replacing
    these 2 columns.
    """
    new_col = new_df_col(expression, df)
    df = df.iloc[:, :-2].join(new_col)
    return df


def update_df_with_multiple_expr(expression1, expression2, df):
    """
    Removes last 2 columns of df and replace with expressions replacing
    these 2 columns.
    """
    print(expression1, expression2)
    print(df)
    try:
        new_col_1 = new_df_col(expression1, df)
        new_col_2 = df.loc[:, [expression2]]
    except KeyError:
        new_col_2 = new_df_col(expression2, df)
        new_col_1 = df.loc[:, [expression1]]

    new_cols = new_col_1.join(new_col_2)
    df = df.iloc[:, :-2].join(new_cols)
    print(df)
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

        # if all(M*(1 - delta) < abs(v) < M*(1 + delta) for v in df.iloc[:, -1].values):
        if len([v for v in col
                if abs(v) < M*(1 - delta) or abs(v) > M*(1 + delta)]) \
                <= np.ceil(delta*len(col)):
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


def average_df(df):
    if len(df.index) == 3:
        return df
    else:
        col_0, col_1 = df.columns[-2], df.columns[-1]
        unique_vals = df[col_0].unique()
        return_df = pd.DataFrame()
        for val in unique_vals:
            mini_df = df.loc[df[col_0] == val]
            n_df = mini_df.iloc[:, :-1]
            n_df[col_1] = np.mean(mini_df[mini_df.columns[-1]])
            return_df = pd.concat((return_df, n_df))
        return return_df


def linear_relns(df, dummy_sym, expr_sym):
    indecies = df.index.values

    data1 = df.iloc[:, -1].values.tolist()
    data2 = df.iloc[:, -2].values.tolist()

    expr_data, dummy_data = [], []

    # Is the below correct?
    for idx in range(len(data1)):
        d1_1 = data1[:idx + 1] + data1[idx + 2:]
        d2_1 = data2[:idx + 1] + data2[idx + 2:]

        d1_2 = data1[:idx - 1] + data1[idx:]
        d2_2 = data2[:idx - 1] + data2[idx:]

        m1, c1 = np.polyfit(d1_1, d2_1, 1)
        m2, c2 = np.polyfit(d1_2, d2_2, 1)
        dummy_data.append((m1 + m2)/2)
        expr_data.append((c1 + c2)/2)

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
