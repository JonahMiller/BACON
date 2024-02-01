import pandas as pd
import numpy as np
from itertools import product

from bacon.bacon3 import BACON_3
import bacon.losses as bl


class BACON_4:
    """
    Finds nominative values in inputted data. Splits the dataframe up and runs BACON.3 
    on the numerical components.
    """
    def __init__(self, initial_df, bacon_1_info=False, bacon_3_info=False, bacon_4_info=False):
        self.initial_df = initial_df
        self.dfs = []
        self.bacon_1_info = bacon_1_info
        self.bacon_3_info = bacon_3_info
        self.bacon_4_info = bacon_4_info
        self.eqns = []

    def find_string_col(self):
        """
        Returns the names of columns that have nominative values in them.
        """
        self.cols_with_strings = []
        for col in self.initial_df:
            if self.initial_df[col].dtype == np.object_:
                self.cols_with_strings.append(col)

    def split_df_in_cols(self):
        """
        Creates all the dataframes of combinations of unique nominative values based
        on the list calculated in the above function.
        """
        unique_strings = []
        self.smaller_dfs = []
        for col in self.cols_with_strings:
            unique_strings.append(self.initial_df[col].unique())
        n = len(self.cols_with_strings)
        for it in product(*unique_strings):
            smaller_df = self.initial_df
            for i in range(n):
                smaller_df = smaller_df.loc[smaller_df[self.cols_with_strings[i]] == it[i]]
            self.smaller_dfs.append(smaller_df)

    def bacon_iterations(self):
        """
        Run BACON 3 iteration over each combination of nominative values and specify
        what nominative values these are.
        """
        self.find_string_col()
        self.split_df_in_cols()

        for df in self.smaller_dfs:

            if self.bacon_4_info:
                print(f"BACON 4: Running BACON 3 iteratively on nominative values.")
                print(f"         Currently running with {self.cols_with_strings} equal")
                print(f"         to {[df[col].iloc[0] for col in self.cols_with_strings]}")

            df = df.drop(columns=self.cols_with_strings)

            bacon = BACON_3(df, 
                            bacon_1_info=self.bacon_1_info,
                            bacon_3_info=self.bacon_3_info)
        
            bacon.bacon_iterations()
