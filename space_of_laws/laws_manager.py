from space_of_laws.laws_methods.bacon1 import BACON_1
from space_of_laws.laws_methods.bacon6 import BACON_6
from space_of_laws.laws_methods.low_level_pysr import PYSR


def bacon_1(df, col_1, col_2, all_found_symbols,
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


def bacon_6(df, col_1, col_2, all_found_symbols,
            verbose=False, expression=None, unknowns=None,
            step=32, N_threshold=3, return_print=False):
    """
    Runs an instance of BACON.6 on the specified columns
    col_1 and col_2 in the specified dataframe df.
    """
    if verbose:
        unused_df = df.iloc[:, :-2]
        col_names = unused_df.columns.tolist()
        col_ave = [unused_df.loc[:, name].mean() for name in col_names]
        if len(col_names) != 0:
            print(f"Laws manager: Running BACON 6 on variables [{col_1}, {col_2}] and")
            print(f"              unused variables {col_names} set as {col_ave}.")
        else:
            print(f"Laws manager: Running BACON 6 on variables [{col_1}, {col_2}]")
    bacon_6_instance = BACON_6(df[[col_1, col_2]], all_found_symbols,
                               expression=expression, unknowns=unknowns,
                               step=step, N_threshold=N_threshold,
                               verbose=verbose, return_print=return_print)
    return bacon_6_instance.run_iteration()


def pysr(df, col_1, col_2, all_found_symbols, verbose=False, return_print=False):
    if verbose:
        unused_df = df.iloc[:, :-2]
        col_names = unused_df.columns.tolist()
        col_ave = [unused_df.loc[:, name].mean() for name in col_names]
        if len(col_names) != 0:
            print(f"Laws manager: Running PySR on variables [{col_1}, {col_2}] and")
            print(f"              unused variables {col_names} set as {col_ave}.")
        else:
            print(f"Laws manager: Running PySR on variables [{col_1}, {col_2}]")
    pysr_instance = PYSR(df[[col_1, col_2]], all_found_symbols,
                         verbose=verbose, return_print=return_print)
    return pysr_instance.run_iteration()


def laws_main(laws_type, kwargs):
    if kwargs:
        if laws_type == "bacon.1":
            return lambda df, col_1, col_2, afs: bacon_1(df, col_1, col_2, afs, **kwargs)
        elif laws_type == "bacon.6":
            return lambda df, col_1, col_2, afs: bacon_6(df, col_1, col_2, afs, **kwargs)
        elif laws_type == "pysr":
            return lambda df, col_1, col_2, afs: pysr(df, col_1, col_2, afs, **kwargs)

    else:
        if laws_type == "bacon.1":
            return bacon_1
        elif laws_type == "bacon.6":
            return bacon_6
        elif laws_type == "pysr":
            return pysr
