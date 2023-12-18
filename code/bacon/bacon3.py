import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import data as d
import pandas as pd
from bacon1 import BACON_1


class BACON_3:
    def __init__(self, data, variables):
        self.df = pd.DataFrame({v: d for v, d in zip(variables, data)})
        self.var = variables
        self.num_var = len(variables)

    def iterate_over_df(self, df_subset):
        new_constant = []
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
            
            # Save results as new column for dataframe with correct indecies
            new_constant.append(results[1])
            current_data = pd.DataFrame({results[1]: results[0]}, index=indecies)
            new_col = pd.concat([new_col, current_data])
        return new_col

    def bacon_3_iterations(self):
        while self.num_var > 2:
            # Take last 3 columns of data frame to turn into 2 columns
            small_df = self.iterate_over_df(self.df.iloc[:, -3:])

            # Replace the 3 columns of the dataframe into the 2 columns
            self.df = self.df.iloc[:, :-2].join(small_df)
            self.num_var -= 1
    
        # When only 2 columns left do simple Bacon 1
        var1, var2 = self.df.columns[0], self.df.columns[1]
        data1, data2 = self.df[var1].values, self.df[var2].values
        bacon_1_instance = BACON_1([data1, data2], [var1, var2])
        print(f"{bacon_1_instance.main()[1]} is constant") 



if __name__ == '__main__':
    initial_data, initial_symbols = d.ideal_gas()
    b = BACON_3(initial_data, initial_symbols)
    b.bacon_3_iterations()
