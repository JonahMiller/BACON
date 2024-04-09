import pandas as pd

from utils import df_helper as df_helper
from utils.gp import ranking


class ranking_layer:
    def __init__(self, df, laws_method, verbose=False):
        self.df = df
        self.laws_method = laws_method
        self.verbose = verbose

    def find_exprs(self):
        exprs_found = {}
        lin_relns = {}

        s_dfs = df_helper.deconstruct_df(self.df)

        invalid_returns = 0

        for df in s_dfs:
            ave_df = df_helper.average_small_df(df)
            data, symb, lin = self.laws_method(ave_df, ave_df.columns[-1], ave_df.columns[-2])

            if isinstance(lin, list):
                symb = lin[2]

            if symb:
                if symb in exprs_found:
                    exprs_found[symb] += 1
                else:
                    exprs_found[symb] = 1
                    if isinstance(lin, list):
                        lin_relns[symb] = [lin[1], lin[2], lin[4], lin[5]]

            else:
                invalid_returns += 1

        if invalid_returns == len(s_dfs):
            raise Exception("No relationships found compatible with this program")

        return exprs_found, lin_relns

    def construct_dfs(self):
        new_dfs = {}
        exprs, lin_relns = self.find_exprs()

        for expr in exprs:
            new_dfs[expr] = []

            if expr in lin_relns:
                df = df_helper.update_df_with_multiple_expr(lin_relns[expr][2],
                                                            lin_relns[expr][3],
                                                            self.df)
                s_dfs = df_helper.deconstruct_df(df)

                new_dummy_col, new_expr_col = pd.DataFrame(), pd.DataFrame()

                for s_df in s_dfs:
                    dummy_col, expr_col = df_helper.linear_relns(s_df,
                                                                 lin_relns[expr][0],
                                                                 lin_relns[expr][1])
                    new_dummy_col = pd.concat([new_dummy_col, dummy_col])
                    new_expr_col = pd.concat([new_expr_col, expr_col])

                n_df1 = df.iloc[:, :-2].join(new_dummy_col)
                n_df2 = df.iloc[:, :-2].join(new_expr_col)

                new_dfs[expr] = [n_df1, n_df2]

            else:
                n_df = df_helper.update_df_with_single_expr(expr, self.df)
                new_dfs[expr] = [n_df]

        return new_dfs

    @staticmethod
    def rank_exprs(exprs_df_dict):
        print("+++++++++++++++++++++++++++++++")
        best_ratio = 0
        for expr, dfs in exprs_df_dict.items():
            if len(dfs) == 2:
                s_n_ratio = min(ranking(dfs[0]).signal_noise_ratio(),
                                ranking(dfs[1]).signal_noise_ratio())
            else:
                s_n_ratio = ranking(dfs[0]).signal_noise_ratio()
            if s_n_ratio > best_ratio:
                best_ratio = s_n_ratio
                best_expr = expr
            print(expr, s_n_ratio)
        print("+++++++++++++++++++++++++++++++")
        return best_expr

    def run_single_iteration(self):
        expr_df_dict = self.construct_dfs()
        expr = ranking_layer.rank_exprs(expr_df_dict)
        return expr_df_dict[expr]
