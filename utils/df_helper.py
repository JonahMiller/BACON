import pandas as pd
import numpy as np
from statistics import fmean
from sympy import Eq, lambdify,  Symbol, simplify

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
    expr2 = expr.subs(last_col_name, Symbol("d_0"))
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
    try:
        new_col_1 = new_df_col(expression1, df)
        new_col_2 = df.loc[:, [expression2]]
    except KeyError:
        new_col_2 = new_df_col(expression2, df)
        new_col_1 = df.loc[:, [expression1]]

    new_cols = new_col_1.join(new_col_2)
    df = df.iloc[:, :-2].join(new_cols)
    return df


def check_const_col(dfs, eqns, delta, logging):
    """
    Checks if the final column (of newly found variable) is fixed.
    """
    idx_delete = []
    for i, df in enumerate(list(dfs)):

        col = df.iloc[:, -1].values
        col = col.reshape(-1, 3).mean(axis=1)
        mean = fmean(col)
        M = abs(mean)
        # print(mean, M, M*(1-delta), M*(1+delta), delta)
        # print(df.columns.tolist()[-1], df.iloc[:, -1].values)
        if all(M*(1 - delta) < abs(v) < M*(1 + delta) for v in df.iloc[:, -1].values):

            # if len([v for v in col
            #         if abs(v) < M*(1 - delta) or abs(v) > M*(1 + delta)]) \
            #         <= np.ceil(delta*len(col)):

            if logging:
                print(f"BACON 5: {df.columns.tolist()[-1]} is constant at {mean}")
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


def average_small_df(df):
    rows = len(df.index)
    if rows != 3:
        var_name = df.columns.tolist()[-1]
        mean, idx = [], []
        for i in range(3):
            row_vals = df.iloc[i*int(rows/3):(i+1)*int(rows/3), -1].values.tolist()
            mean.append(fmean(row_vals))
            idx.append(i*int(rows/3))
        n_df = df.iloc[idx, :-1]
        var_df = pd.DataFrame({var_name: mean}, index=n_df.index)
        return n_df.join(var_df)
    return df


def linear_relns(df, dummy_sym, expr_sym):
    # print(df)
    indecies = df.index.values

    data1 = df.iloc[:, -1].values.tolist()
    data2 = df.iloc[:, -2].values.tolist()

    expr_data, dummy_data = [], []

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


def score(init_df, eqns):
    key_var = init_df.columns[-1]

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("The constant equations found are:")
    for eqn in eqns:
        print(f"{eqn.rhs} = {eqn.lhs}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    eqn = loss_helper.simplify_eqns(init_df, eqns, key_var).iterate_through_dummys()
    loss = loss_helper.loss_calc(init_df, eqn).loss()
    print(f"Final form is {eqn.rhs} = {eqn.lhs} with loss {loss}.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
