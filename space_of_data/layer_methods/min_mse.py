import pandas as pd
from statistics import fmean
from sklearn.metrics import mean_squared_error as mse

from utils import df_helper as df_helper


class min_mse_layer:
    def __init__(self, df, laws_method, symbols, verbose=False):
        self.df = df
        self.laws_method = laws_method
        self.symbols = symbols
        self.verbose = verbose

    def find_exprs(self):
        self.exprs_found = {}
        self.lin_relns = {}
        invalid_returns = 0

        s_dfs = df_helper.deconstruct_df(self.df)
        for df in s_dfs:
            ave_df = df_helper.average_df(df)
            data, symb, lin = self.laws_method(ave_df, ave_df.columns[-2],
                                               ave_df.columns[-1], self.symbols)

            if isinstance(lin, list):
                symb = lin[2]

            if symb:
                if symb in self.exprs_found:
                    self.exprs_found[symb] += 1
                else:
                    self.exprs_found[symb] = 1
                    if isinstance(lin, list):
                        self.lin_relns[symb] = [lin[1], lin[2], lin[4], lin[5]]
            else:
                invalid_returns += 1

        if self.verbose:
            print(f"Ranking layer: Expressions found are {self.exprs_found}")

        if invalid_returns == len(s_dfs):
            # raise Exception("No relationships found compatible with this program")
            return "Invalid"

        return "Continue"

    def construct_dfs(self):
        self.exprs_dict = {}

        for expr in self.exprs_found:
            self.exprs_dict[expr] = []

            if expr in self.lin_relns:
                df = df_helper.update_df_multiple_expr(self.lin_relns[expr][2],
                                                       self.lin_relns[expr][3],
                                                       self.df)
                s_dfs = df_helper.deconstruct_df(df)

                new_dummy_col, new_expr_col = pd.DataFrame(), pd.DataFrame()
                for s_df in s_dfs:
                    dummy_col, expr_col = df_helper.linear_relns(s_df,
                                                                 self.lin_relns[expr][0],
                                                                 self.lin_relns[expr][1])
                    new_dummy_col = pd.concat([new_dummy_col, dummy_col])
                    new_expr_col = pd.concat([new_expr_col, expr_col])

                n_df1 = df.iloc[:, :-2].join(new_dummy_col)
                n_df2 = df.iloc[:, :-2].join(new_expr_col)

                self.exprs_dict[expr] = [n_df1, n_df2]

            else:
                n_df = df_helper.update_df_single_expr(expr, self.df)
                self.exprs_dict[expr] = [n_df]

        return self.exprs_dict

    @staticmethod
    def calc_mse(df):
        average_mse = 0
        n = len(df.iloc[:, -2].unique())
        rows = len(df.index)
        for i in range(int(rows/n)):
            col = df.iloc[i*n: (i+1)*n, -1]
            mse_score = mse(col, len(col)*[fmean(col)])
            average_mse += mse_score
        return average_mse

    def rank_exprs(self):
        best_ave_mse = 1e100
        len_best_expr = 1

        if len(self.exprs_dict) > 1:
            if self.verbose:
                print("Ranking layer: Iteratively ranking the found expressions:")

            for expr, dfs in self.exprs_dict.items():
                if len(dfs) == 2:
                    average_mse = (min_mse_layer.calc_mse(dfs[0])
                                   + min_mse_layer.calc_mse(dfs[1]))/2
                else:
                    average_mse = min_mse_layer.calc_mse(dfs[0])
                if average_mse < best_ave_mse:
                    best_ave_mse = average_mse
                    best_expr = expr
                    len_best_expr = len(dfs)
                if self.verbose:
                    print(f"               {expr} has average mse {average_mse}")
        else:
            best_expr = list(self.exprs_dict.keys())[0]
            len_best_expr = len(self.exprs_dict[best_expr])

        if len_best_expr == 2:
            self.symbols.append(self.lin_relns[best_expr][0])
        return best_expr

    def run_single_iteration(self):
        return_status = self.find_exprs()
        if return_status == "Invalid":
            return None, None

        self.construct_dfs()
        expr = self.rank_exprs()
        return self.exprs_dict[expr], self.symbols
