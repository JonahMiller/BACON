import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import data as d
import pandas as pd
from bacon1 import BACON_1
from statistics import fmean


class BACON_3:
    def __init__(self, data, variables):
        self.df = pd.DataFrame({v: d for v, d in zip(variables, data)})
        self.dfs = [self.df]
        self.var = variables
        self.num_var = len(self.var)

    def iterate_over_df(self, df_subset):
        new_col = pd.DataFrame()
        # Split dataframe into smaller based on uniqueness in first column
        for i in df_subset[df_subset.columns[0]].unique():
            df = df_subset[df_subset[df_subset.columns[0]] == i]
            indecies = df.index.values
            
            # Perform Bacon.1 on last 2 columns in system
            var1, var2 = df.columns[1], df.columns[2]
            data1, data2 = df[var1].values, df[var2].values
            bacon_1_instance = BACON_1([data1, data2], [var1, var2])
            results = bacon_1_instance.main()

            if isinstance(results[2], list):
                current_data = pd.DataFrame({results[2][2]: results[0], 
                                             results[2][1]: results[2][3]}, index=indecies)
                new_col = pd.concat([new_col, current_data])
                self.var.append(results[2][1])
            else:
                # Save results as new column for dataframe with correct indecies
                current_data = pd.DataFrame({results[1]: results[0]}, index=indecies)
                new_col = pd.concat([new_col, current_data])

        print(f"Within a locally constant value for "
              f"{[self.df.columns[i] for i in range(self.num_var - 2)]}, "
              f"the relationship {new_col.columns[0]} is fixed")
        return new_col
    
    def check_df_columns_greater_than_2(self):
        for df in self.dfs:
            if len(df.columns) > 2:
                return True
        return False


    def bacon_3_iterations(self):
        # Checks if any dataframes still need to be minimised
        while self.check_df_columns_greater_than_2():
            new_dfs = []
            for df in self.dfs:
                if len(self.df.columns) > 2:
                    # Take last 3 columns of data frame to turn into 2 columns
                    small_df = self.iterate_over_df(df.iloc[:, -3:])
                    if len(small_df.columns) == 1:
                        # Replace the last columns of the dataframe into the new column
                        df = df.iloc[:, :-2].join(small_df)
                        new_dfs.append(df)
                    elif len(small_df.columns) == 2:
                        for cols in small_df:
                            ndf = df.iloc[:, :-2].join(small_df[cols])
                            new_dfs.append(ndf)
            self.dfs = new_dfs
        for df in self.dfs:
            # When only 2 columns left do simple Bacon 1
            var1, var2 = df.columns[0], df.columns[1]
            data1, data2 = df[var1].values, df[var2].values
            bacon_1_instance = BACON_1([data1, data2], [var1, var2])
            r = bacon_1_instance.main(info=True)
            print(f"{r[1]} is constant at {fmean(r[0])}") 
            



if __name__ == '__main__':
    initial_data, initial_symbols = d.ideal_gas()
    b = BACON_3(initial_data, initial_symbols)
    b.bacon_3_iterations()
