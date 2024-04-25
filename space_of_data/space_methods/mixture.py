from statistics import fmean
from sympy import Eq
import sys

from utils import df_helper as df_helper
from space_of_data.layer_methods.gp_ranking import ranking_layer
from space_of_data.layer_methods.popularity import popular_layer


class mixture:
    """
    Manages the layers of the dataframe, including the potentially new layers found
    when linear relationships are found. Then it runs BACON.1 on the those two columns.
    """
    def __init__(self, initial_df, laws_method, delta=0.2, verbose=False):
        self.initial_df = initial_df
        self.dfs = [initial_df]
        self.laws_method = laws_method
        self.verbose = verbose
        self.symbols = list(sum(sym for sym in list(initial_df)).free_symbols)
        self.delta = delta
        self.eqns = []

    def run_iterations(self):
        """
        Manages the iterations over all the layers in a for loop until each dataframe
        only has two columns left.
        """
        while self.not_final_3():
            new_dfs = []

            self.dfs, self.eqns = df_helper.check_const_col(self.dfs, self.eqns,
                                                            self.delta, self.verbose)

            for df in self.dfs:
                layer_in_context = popular_layer(df, self.laws_method,
                                                 self.symbols, True)
                new_df, self.symbols = layer_in_context.run_single_iteration()
                new_dfs.extend(new_df)

                if self.verbose:
                    var1, var2 = df.columns[-1], df.columns[-2]
                    unused_df = df.iloc[:, :-2]
                    col_names = unused_df.columns.tolist()

                    print(f"Data space: Calculating laws on variables [{var1}, {var2}] and")
                    print(f"            keeping constant unused variables {col_names}")
                    print(f"            displayed fix variables {[df.columns[-1] for df in new_df]}.")

            self.dfs = new_dfs

        while self.not_last_iteration():
            new_dfs = []

            self.dfs, self.eqns = df_helper.check_const_col(self.dfs, self.eqns,
                                                            self.delta, self.verbose)

            for df in self.dfs:
                layer_in_context = ranking_layer(df, self.laws_method,
                                                 self.symbols, True)
                new_df, self.symbols = layer_in_context.run_single_iteration()
                new_dfs.extend(new_df)

                if self.verbose:
                    var1, var2 = df.columns[-1], df.columns[-2]
                    unused_df = df.iloc[:, :-2]
                    col_names = unused_df.columns.tolist()

                    print(f"Data space: Calculating laws on variables [{var1}, {var2}] and")
                    print(f"            keeping constant unused variables {col_names}")
                    print(f"            displayed fix variables {[df.columns[-1] for df in new_df]}.")

            self.dfs = new_dfs

        constants = []
        for df in self.dfs:
            # When only 2 columns left do final calculation

            if self.verbose:
                print(f"Data space: Calculating laws on final variables [{df.columns[0]}, {df.columns[1]}]")

            ave_df = df_helper.average_df(df)
            print(ave_df)
            results = self.laws_method(ave_df, ave_df.columns[1], ave_df.columns[0], self.symbols)

            if results[2] == "print":
                print(f"Data space: {results[1][0]} = {results[1][1]}")
                sys.exit()

            if not results[1]:
                raise Exception("Data space: No satisfactory result found at last level")
            for r in results:
                print(r)
            if self.verbose:
                print(f"Data space: {results[1]} is constant at {fmean(results[0])}")
            constants.append(results[1])

            self.eqns.append(Eq(results[1], fmean(results[0])))

        df_helper.score(self.initial_df, self.eqns)

    def not_final_3(self):
        for df in self.dfs:
            if len(df.columns) > 3:
                return True
        return False

    def not_last_iteration(self):
        for df in self.dfs:
            if len(df.columns) > 2:
                return True
        return False
