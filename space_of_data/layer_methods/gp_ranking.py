import pandas as pd

from utils import df_helper as df_helper
from utils.gp import ranking


class ranking_layer:
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
            data, symb, lin = self.laws_method(ave_df, ave_df.columns[-1],
                                               ave_df.columns[-2], self.symbols)

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
            raise Exception("No relationships found compatible with this program")

    def construct_dfs(self):
        self.exprs_dict = {}

        for expr in self.exprs_found:
            self.exprs_dict[expr] = []

            if expr in self.lin_relns:
                df = df_helper.update_df_with_multiple_expr(self.lin_relns[expr][2],
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
                n_df = df_helper.update_df_with_single_expr(expr, self.df)
                self.exprs_dict[expr] = [n_df]

        return self.exprs_dict

    def rank_exprs(self):
        best_ratio = 0
        len_best_expr = 1
        if self.verbose:
            print("Ranking layer: Iteratively ranking the found expressions:")

        if len(self.exprs_dict) > 1:
            for expr, dfs in self.exprs_dict.items():
                if len(dfs) == 2:
                    s_n_ratio = min(ranking(dfs[0]).signal_noise_ratio(),
                                    ranking(dfs[1]).signal_noise_ratio())
                else:
                    s_n_ratio = ranking(dfs[0]).signal_noise_ratio()
                if s_n_ratio > best_ratio:
                    best_ratio = s_n_ratio
                    best_expr = expr
                    len_best_expr = len(dfs)
                if self.verbose:
                    print(f"               {expr} has score {s_n_ratio}")
        else:
            best_expr = list(self.exprs_dict.keys())[0]

        if len_best_expr == 2:
            self.symbols.append(self.lin_relns[best_expr][0])
        return best_expr

    def run_single_iteration(self):
        self.find_exprs()
        self.construct_dfs()
        expr = self.rank_exprs()
        return self.exprs_dict[expr], self.symbols
