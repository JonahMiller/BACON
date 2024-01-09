import pandas as pd
from statistics import fmean

from bacon.bacon1 import BACON_1


def run_bacon_1(df, col_1, col_2, verbose=False):
    var1, var2 = col_1, col_2
    data1, data2 = df[var1].values, df[var2].values
    bacon_1_instance = BACON_1([data1, data2], [var1, var2], info=verbose)
    return bacon_1_instance.bacon_iterations()


class BACON_3_layer:
    def __init__(self, df):
        self.df = df
        self.n_cols = len(df.columns)
        self.broken_dfs = []
        self.df_dicts = {self.n_cols: [df]}

    def break_down_df(self):
        for i in range(self.n_cols, 2, -1):
            smaller_dfs = []
            for df in self.df_dicts[i]:
                for k in df[df.columns[self.n_cols - i]].unique():
                    smaller_dfs.append(df[df[df.columns[self.n_cols - i]] == k])
            self.df_dicts[i - 1] = smaller_dfs
        self.smallest_dfs = self.df_dicts[min(self.df_dicts)]
    
    def iterate_over_df(self):
        new_cols = pd.DataFrame()
        for df in self.smallest_dfs:
            indecies = df.index.values
        
            # Perform Bacon.1 on last 2 columns in system
            results = run_bacon_1(df, df.columns[-1], df.columns[-2])

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
        self.break_down_df()
        new_cols = self.iterate_over_df()
        df_list = self.construct_updated_df(new_cols)
        return df_list


class BACON_3:
    def __init__(self, data, variables, info=False):
        self.initial_df = pd.DataFrame({v: d for v, d in zip(variables, data)})
        self.dfs = [self.initial_df]
        self.delta = 0.01

    def bacon_iterations(self):
        while self.not_last_iteration():
            new_dfs = []
            self.check_const_col()
            for df in self.dfs:
                bacon_layer_in_context = BACON_3_layer(df)
                new_df = bacon_layer_in_context.run_single_iteration()
                new_dfs.extend(new_df)
            self.dfs = new_dfs

        constants = []
        for df in self.dfs:
            # When only 2 columns left do simple Bacon 1
            results = run_bacon_1(df, df.columns[0], df.columns[1], verbose=True)
            print(f"BACON 3: {results[1]} is constant at {fmean(results[0])}") 
            constants.append(results[1])

    
    def not_last_iteration(self):
        for df in self.dfs:
            if len(df.columns) > 2:
                return True
        return False
    
    def print_dfs(self):
        for df in self.dfs:
            print(df)

    def check_const_col(self):
        for i, df in enumerate(list(self.dfs)):
            temp_dict = df.to_dict("list")
            for idx, val in temp_dict.items():
                mean = fmean(val)
                M = abs(mean)
                if all(M*(1 - self.delta) < abs(v) < M*(1 + self.delta) for v in val):
                    print(f"BACON 3: {idx} is constant at {mean}")
                    del self.dfs[i]
