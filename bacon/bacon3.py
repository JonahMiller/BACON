import pandas as pd
from statistics import fmean

from bacon.bacon1 import BACON_1


class BACON_3:
    def __init__(self, data, variables, info):
        self.df = pd.DataFrame({v: d for v, d in zip(variables, data)})
        self.dfs = [self.df]
        self.info = info

    def iterate_over_df(self, df_subset, unused_var):
        new_col = pd.DataFrame()
        # Split dataframe into smaller based on uniqueness in first column

        # if self.info:
        #     print(f"BACON 3: Considering variables "
        #           f"{unused_var}, "
        #           f"the relationship {new_col.columns[0]} is fixed\n")

        for i in df_subset[df_subset.columns[0]].unique():

            df = df_subset[df_subset[df_subset.columns[0]] == i]
            indecies = df.index.values
            
            # Perform Bacon.1 on last 2 columns in system
            var1, var2 = df.columns[1], df.columns[2]
            data1, data2 = df[var1].values, df[var2].values
            bacon_1_instance = BACON_1([data1, data2], [var1, var2])
            results = bacon_1_instance.bacon_iterations()

            # Special check for linear relationship added to dataframe
            if isinstance(results[2], list):
                current_data = pd.DataFrame({results[2][2]: results[0], 
                                             results[2][1]: results[2][3]}, index=indecies)
                new_col = pd.concat([new_col, current_data])
            else:
                # Save results as new column for dataframe with correct indecies
                current_data = pd.DataFrame({results[1]: results[0]}, index=indecies)
                new_col = pd.concat([new_col, current_data])

        if self.info:
            print(f"BACON 3: Within a locally constant value for "
                  f"{unused_var}, "
                  f"the relationship {new_col.columns[0]} is fixed")

        return new_col
    
    def check_df_columns_greater_than_2(self):
        for df in self.dfs:
            if len(df.columns) > 2:
                return True
        return False
    
    def print_dfs(self):
        for df in self.dfs:
            print(df)

    def bacon_iterations(self):
        # Checks if any dataframes still need to be minimised
        while self.check_df_columns_greater_than_2():
            new_dfs = []
            for df in self.dfs:
                unused_var = [df.columns[i] for i in range(len(df.columns) - 2)]
                if len(self.df.columns) > 2:
                    # Take last 3 columns of data frame
                    small_df = self.iterate_over_df(df.iloc[:, -3:], unused_var)
                    if len(small_df.columns) == 1:
                        # Replace the last columns of the dataframe into the new column
                        # if relationship found between last two elements proportional
                        df = df.iloc[:, :-2].join(small_df)
                        new_dfs.append(df)
                    elif len(small_df.columns) == 2:
                        # Create multiple new dataframes with the found last two columns
                        # if the relationship is linear
                        for cols in small_df:
                            ndf = df.iloc[:, :-2].join(small_df[cols])
                            new_dfs.append(ndf)
            self.dfs = new_dfs

        constants = []
        for df in self.dfs:
            # When only 2 columns left do simple Bacon 1
            var1, var2 = df.columns[0], df.columns[1]
            data1, data2 = df[var1].values, df[var2].values
            bacon_1_instance = BACON_1([data1, data2], [var1, var2])
            r = bacon_1_instance.bacon_iterations()
            print(f"BACON 3: {r[1]} is constant at {fmean(r[0])}") 
            constants.append(r[1])

