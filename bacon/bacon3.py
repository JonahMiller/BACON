import pandas as pd
from statistics import fmean
from sympy import Eq

from bacon.bacon1 import BACON_1


def run_bacon_1(df, col_1, col_2, verbose=False):
    """
    Runs an instance of BACON.1 on the specified columns 
    col_1 and col_2 in the specified dataframe df.
    """
    var1, var2 = col_1, col_2
    data1, data2 = df[var1].values, df[var2].values
    if verbose:
        unused_df = df.iloc[:, :-2]
        col_names = unused_df.columns.tolist()
        col_ave = [unused_df.loc[:, name].mean() for name in col_names]
        if len(col_names) != 0:
            print(f"BACON 1: Running BACON 1 on variables [{var1}, {var2}] and") 
            print(f"         unused variables {col_names} set as {col_ave}.")
        else:
            print(f"BACON 1: Running BACON 1 on variables [{var1}, {var2}]")
    bacon_1_instance = BACON_1([data1, data2], [var1, var2], info=verbose)
    return bacon_1_instance.bacon_iterations()


class BACON_3_layer:
    """
    BACON.3 can be thought of a layer-by-layer running of BACON.1 with
    previous variable fixes. This class runs each layer instance.
    """
    def __init__(self, df, bacon_1_info=False):
        self.df = df
        self.print_df_to_file()
        self.n_cols = len(df.columns)
        self.broken_dfs = []
        self.df_dicts = {self.n_cols: [df]}
        self.bacon_1_info = bacon_1_info

    def print_df_to_file(self):
        self.df.to_csv('df.txt', sep='\t', index=False)

    def break_down_df(self):
        """
        Creates the lowest level of dataframe to run BACON.1 on.
        Eg. if a dataframe of variables ABCD is fed in, it creates a
        dataframe for each combination of fixed A, B with variable C, D to
        find local patterns for C, D.
        """
        for i in range(self.n_cols, 2, -1):
            smaller_dfs = []
            for df in self.df_dicts[i]:
                for k in df[df.columns[self.n_cols - i]].unique():
                    smaller_dfs.append(df[df[df.columns[self.n_cols - i]] == k])
            self.df_dicts[i - 1] = smaller_dfs
        self.smallest_dfs = self.df_dicts[min(self.df_dicts)]
    
    def iterate_over_df(self):
        """
        Runs the BACON.1 iterations over the dataframe found above. 
        if there is a list returned it means a linear relationship was found
        as the linear relationship value can be a variable it then creates two
        instances, one with the y-intercept and the other with the gradient of
        the linear relationship. It returnse the new columns found.
        """
        new_cols = pd.DataFrame()
        for df in self.smallest_dfs:
            indecies = df.index.values
        
            # Perform Bacon.1 on last 2 columns in system
            results = run_bacon_1(df, df.columns[-1], df.columns[-2], verbose=self.bacon_1_info)

            # Special check for linear relationship added to dataframe
            if isinstance(results[2], list):
                current_data = pd.DataFrame({results[2][2]: results[0], 
                                            results[2][1]: results[2][3]}, index=indecies)
                new_cols = pd.concat([new_cols, current_data])
            else:
                # Save results as new column for dataframe with correct indecies
                current_data = pd.DataFrame({results[1]: results[0]}, index=indecies)
                new_cols = pd.concat([new_cols, current_data])
        return new_cols
    
    def construct_updated_df(self, new_cols):
        """
        Reconstructs the dataframe with the new columns, ie. the column
        for C, D becomes a column for f(C, D) for f the relationship found
        """
        new_dfs = []
        if len(new_cols.columns) == 1:
            # Replace the last columns of the dataframe into the new column
            # if relationship found between last two elements proportional
            df = self.df.iloc[:, :-2].join(new_cols)
            new_dfs.append(df)
        elif len(new_cols.columns) == 2:
            # Create multiple new dataframes with the found last two columns
            # if the relationship is linear
            for cols in new_cols:
                df = self.df.iloc[:, :-2].join(new_cols[cols])
                new_dfs.append(df)
        return new_dfs

    def run_single_iteration(self):
        """
        Runs the entire class and returns new dataframes for BACON.3 to act on.
        """
        self.break_down_df()
        new_cols = self.iterate_over_df()
        df_list = self.construct_updated_df(new_cols)
        return df_list


class BACON_3:
    """
    Manages the layers of the dataframe, including the potentially new layers found
    when linear relationships are found. Then it runs BACON.1 on the those two columns. 
    """
    def __init__(self, data, variables, bacon_1_info=False, bacon_3_info=False):
        self.initial_df = pd.DataFrame({v: d for v, d in zip(variables, data)})
        self.dfs = [self.initial_df]
        self.delta = 0.01
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
            self.check_const_col()
            for df in self.dfs:

                bacon_layer_in_context = BACON_3_layer(df, self.bacon_1_info)
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

        constants = []
        for df in self.dfs:
            # When only 2 columns left do simple Bacon 1

            if self.bacon_3_info:
                print(f"BACON 3: Running BACON 1 on final variables [{df.columns[0]}, {df.columns[1]}]")

            results = run_bacon_1(df, df.columns[0], df.columns[1], verbose=self.bacon_1_info)
            print(f"BACON 3: {results[1]} is constant at {fmean(results[0])}")
            constants.append(results[1])
            self.eqns.append(Eq(results[1], fmean(results[0])))

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
                    print(f"BACON 3: {idx} is constant at {mean}")
                    self.eqns.append(Eq(idx, mean))
                    del self.dfs[i]
