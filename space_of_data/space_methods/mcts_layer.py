import pandas as pd
from statistics import fmean
from sklearn.metrics import mean_squared_error as mse

from utils import df_helper as df_helper


class layer:
    def __init__(self, df, laws_method, symbols, ranking_method, verbose=False):
        self.df = df
        self.laws_method = laws_method
        self.symbols = symbols
        self.verbose = verbose

        self.ranking_method = ranking_method

    def find_exprs(self):
        self.exprs_found = {}
        self.exprs_idx = {}
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
                    self.exprs_idx[symb].extend(list(df.index.values))
                else:
                    self.exprs_found[symb] = 1
                    self.exprs_idx[symb] = list(df.index.values)

                    if isinstance(lin, list):
                        self.lin_relns[symb] = [lin[1], lin[2], lin[4], lin[5]]
            else:
                invalid_returns += 1

        if self.verbose:
            print(f"Ranking layer: Expressions found are {self.exprs_found}")
            # print(f"Indexes for expression are {self.exprs_idx}")

        if invalid_returns == len(s_dfs):
            return "Invalid"

        return "Continue"

    def rank_popular(self):
        # best_expr = max(self.exprs_found, key=self.exprs_found.get)
        most_pop_count = max(self.exprs_found.values())
        self.exprs_found = {k: v for k, v in self.exprs_found.items() if v == most_pop_count}

        if len(self.exprs_found) == 1:
            best_expr = list(self.exprs_found.keys())[0]

            if best_expr in self.lin_relns:
                lin_reln = self.lin_relns[best_expr]
            else:
                lin_reln = None

            if lin_reln:
                self.symbols.append(self.lin_relns[best_expr][0])
            return best_expr

        else:
            self.ranking_method = "weight_mses"

    @staticmethod
    def calc_mse(df):
        average_mse = 0
        s_dfs = df_helper.deconstruct_df(df)
        for sdf in s_dfs:
            s_df2 = df_helper.deconstruct_deconstructed_df(sdf)
            for sdf2 in s_df2:
                col = sdf2.iloc[:, -1]
                max_col = max(col)
                new_col = (1/max_col) * col
                mse_score = mse(new_col, len(new_col)*[fmean(new_col)])
                average_mse += mse_score
        return average_mse

    def user_input(self):
        best_ave_mse = 1e100
        len_best_expr = 1

        if len(self.exprs_dict) > 1:
            if self.verbose:
                print("Ranking layer: Iteratively ranking the found expressions:")

            for expr, dfs in self.exprs_dict.items():
                if len(dfs) == 2:
                    average_mse = min(layer.calc_mse(dfs[0]),
                                      layer.calc_mse(dfs[1]))
                else:
                    average_mse = layer.calc_mse(dfs[0])
                if average_mse < best_ave_mse:
                    best_ave_mse = average_mse
                    best_expr = expr
                    len_best_expr = len(dfs)
                if self.verbose:
                    print(f"               {expr} has average mse {average_mse}")
            keys = list(self.exprs_dict.keys())
            idx = int(input(f"Select expression index from {list(zip(keys, range(0, len(keys))))}:" + '\n'))
            best_expr = list(self.exprs_dict)[idx]
            len_best_expr = len(self.exprs_dict[best_expr])
        else:
            best_expr = list(self.exprs_dict.keys())[0]
            len_best_expr = len(self.exprs_dict[best_expr])

        if len_best_expr == 2:
            self.symbols.append(self.lin_relns[best_expr][0])
        return best_expr

    def rank_min_mse(self):
        best_ave_mse = 1e100
        len_best_expr = 1

        if len(self.exprs_dict) > 1:
            if self.verbose:
                print("Ranking layer: Iteratively ranking the found expressions:")

            for expr, dfs in self.exprs_dict.items():
                if len(dfs) == 2:
                    average_mse = min(layer.calc_mse(dfs[0]),
                                      layer.calc_mse(dfs[1]))
                else:
                    average_mse = layer.calc_mse(dfs[0])
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

    def get_relations(self):
        return_status = self.find_exprs()
        if return_status == "Invalid":
            if self.verbose:
                print("Ranking layer: Proceeding with invalid return")
            return None, None

        return self.exprs_found, self.lin_relns


class construct_dfs:
    def __init__(self, df, expr, lin_relns=False, verbose=True):
        self.df = df
        self.verbose = verbose

        self.expr = expr
        self.lin_relns = lin_relns

    def construct_dfs(self):
        if self.lin_relns:
            df = df_helper.update_df_multiple_expr(self.lin_relns[2],
                                                   self.lin_relns[3],
                                                   self.df)
            s_dfs = df_helper.deconstruct_df(df)

            new_dummy_col, new_expr_col = pd.DataFrame(), pd.DataFrame()
            for s_df in s_dfs:
                dummy_col, expr_col = df_helper.linear_relns(s_df,
                                                             self.lin_relns[0],
                                                             self.lin_relns[1])
                new_dummy_col = pd.concat([new_dummy_col, dummy_col])
                new_expr_col = pd.concat([new_expr_col, expr_col])

            n_df1 = df.iloc[:, :-2].join(new_dummy_col)
            n_df2 = df.iloc[:, :-2].join(new_expr_col)

            self.dfs = [n_df1, n_df2]

        else:
            n_df = df_helper.update_df_single_expr(self.expr, self.df)
            self.dfs = [n_df]

        return self.dfs
