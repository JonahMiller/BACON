from space_of_laws.laws_methods.bacon1 import BACON_1
from space_of_laws.laws_methods.low_level_pysr import main


def bacon_1(df, col_1, col_2, verbose=False, delta=0.1, epsilon=0.001):
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
    bacon_1_instance = BACON_1(df[[col_1, col_2]],
                               epsilon, delta,
                               bacon_1_info=verbose)
    return bacon_1_instance.bacon_iterations()


def pysr(df, col_1, col_2, verbose=False):
    if verbose:
        unused_df = df.iloc[:, :-2]
        col_names = unused_df.columns.tolist()
        col_ave = [unused_df.loc[:, name].mean() for name in col_names]
        if len(col_names) != 0:
            print(f"Laws manager: Running PySR on variables [{col_1}, {col_2}] and")
            print(f"              unused variables {col_names} set as {col_ave}.")
        else:
            print(f"Laws manager: Running PySR on variables [{col_1}, {col_2}]")
    return main(df[[col_1, col_2]])


def laws_main(laws_type, kwarg_dict):
    if kwarg_dict:
        if laws_type == "bacon.1":
            return lambda df, col_1, col_2: bacon_1(df, col_1, col_2, **kwarg_dict)
        elif laws_type == "pysr":
            return lambda df, col_1, col_2: pysr(df, col_1, col_2, **kwarg_dict)

    else:
        if laws_type == "bacon.1":
            return bacon_1
        elif laws_type == "pysr":
            return pysr
