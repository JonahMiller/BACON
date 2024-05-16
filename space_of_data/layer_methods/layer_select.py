import pandas as pd
from statistics import fmean
import random
from sklearn.metrics import mean_squared_error as mse

from utils import df_helper as df_helper
from utils.gp import gp


class layer:
    def __init__(self, df, laws_method, symbols, ranking_method, verbose=True):
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

    def rank_gp(self):
        best_ratio = 0
        len_best_expr = 1
        if self.verbose:
            print("Ranking layer: Iteratively scoring the found by GP signal/noise ratio:")

        if len(self.exprs_dict) > 1:
            for expr, dfs in self.exprs_dict.items():
                if len(dfs) == 2:
                    s_n_ratio = min(gp(dfs[0]).signal_noise_ratio(),
                                    gp(dfs[1]).signal_noise_ratio())
                else:
                    s_n_ratio = gp(dfs[0]).signal_noise_ratio()
                if s_n_ratio > best_ratio:
                    best_ratio = s_n_ratio
                    best_expr = expr
                    len_best_expr = len(dfs)
                if self.verbose:
                    print(f"               {expr} has score {s_n_ratio}")
        else:
            best_expr = list(self.exprs_dict.keys())[0]
            len_best_expr = len(self.exprs_dict[best_expr])

        if len_best_expr == 2:
            self.symbols.append(self.lin_relns[best_expr][0])
        return best_expr

    def rank_popularity(self):
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

    def rank_bacon_3(self):
        if len(self.exprs_found) != 1:
            raise Exception("Ranking layer: No relationships found compatible with BACON.3")
        else:
            best_expr = list(self.exprs_found.keys())[0]

        if best_expr in self.lin_relns:
            lin_reln = self.lin_relns[best_expr]
        else:
            lin_reln = None

        if lin_reln:
            self.symbols.append(self.lin_relns[best_expr][0])

        self.exprs_found = {best_expr: self.exprs_found[best_expr]}
        return best_expr

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

    @staticmethod
    def count_satisfy(df):
        satisfy = 0
        s_dfs = df_helper.deconstruct_df(df)
        for sdf in s_dfs:
            s_df2 = df_helper.deconstruct_deconstructed_df(sdf)
            for sdf2 in s_df2:
                col = sdf2.iloc[:, -1]
                min_col = min(col)
                if min_col < 0:
                    col += min_col

                M = fmean(col)
                if all(M*(0.97) < val < M*(1.03) for val in col):
                    satisfy += 1

        return satisfy

    def rank_satisfy(self):
        best_satisfy = 0
        len_best_expr = 1

        if len(self.exprs_dict) > 1:
            if self.verbose:
                print("Ranking layer: Iteratively ranking the found expressions:")

            for expr, dfs in self.exprs_dict.items():
                if len(dfs) == 2:
                    satisfy = max(layer.count_satisfy(dfs[0]),
                                  layer.count_satisfy(dfs[1]))
                else:
                    satisfy = layer.count_satisfy(dfs[0])
                if satisfy > best_satisfy:
                    best_satisfy = satisfy
                    best_expr = expr
                    len_best_expr = len(dfs)
                if self.verbose:
                    print(f"               {expr} has equality in {satisfy}")
        else:
            best_expr = list(self.exprs_dict.keys())[0]
            len_best_expr = len(self.exprs_dict[best_expr])

        if len_best_expr == 2:
            self.symbols.append(self.lin_relns[best_expr][0])
        return best_expr

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

    def random_choice(self):
        if len(self.exprs_dict) > 1:
            if self.verbose:
                print("Ranking layer: Iteratively ranking the found expressions:")

            best_expr, dfs = random.choice(list(self.exprs_dict.items()))
            len_best_expr = len(dfs)

        else:
            best_expr = list(self.exprs_dict.keys())[0]
            len_best_expr = len(self.exprs_dict[best_expr])

        if len_best_expr == 2:
            self.symbols.append(self.lin_relns[best_expr][0])
        return best_expr

    def rank_propto_mse(self):
        mse_dict = {}
        weightings = []

        if len(self.exprs_dict) > 1:
            if self.verbose:
                print("Ranking layer: Iteratively ranking the found expressions:")

            for expr, dfs in self.exprs_dict.items():
                if len(dfs) == 2:
                    average_mse = min(layer.calc_mse(dfs[0]),
                                      layer.calc_mse(dfs[1]))
                else:
                    average_mse = layer.calc_mse(dfs[0])
                mse_dict[average_mse] = [expr, len(dfs)]

                if self.verbose:
                    print(f"               {expr} has average mse {average_mse}")

            found_mses = list(mse_dict.keys())

            if len(found_mses) > 1:
                mse_sum = sum(found_mses)

                for m in found_mses:
                    weightings.append(1 - m/mse_sum)

                weight_sum = sum(weightings)
                norm_weight = [100*float(w)/weight_sum for w in weightings]
                m = random.choices(found_mses, weights=norm_weight, k=1)[0]

                best_expr, len_best_expr = mse_dict[m]
            else:
                best_expr, len_best_expr = mse_dict[found_mses[0]]

        else:
            best_expr = list(self.exprs_dict.keys())[0]
            len_best_expr = len(self.exprs_dict[best_expr])

        if len_best_expr == 2:
            self.symbols.append(self.lin_relns[best_expr][0])
        return best_expr

    def run_single_iteration(self):
        return_status = self.find_exprs()
        if return_status == "Invalid":
            if self.verbose:
                print("Ranking layer: Proceeding with invalid return")
            return None, None

        if self.ranking_method == "bacon.3":
            expr = self.rank_bacon_3()
        elif self.ranking_method == "popularity":
            expr = self.rank_popularity()

        self.construct_dfs()

        if self.ranking_method == "gp_ranking":
            expr = self.rank_gp()
        elif self.ranking_method == "min_mse":
            expr = self.rank_min_mse()
        elif self.ranking_method == "weight_mses":
            expr = self.rank_propto_mse()
        elif self.ranking_method == "satisfy_equality":
            expr = self.rank_satisfy()
        elif self.ranking_method == "random":
            expr = self.random_choice()
        elif self.ranking_method == "user_input":
            expr = self.user_input()

        if self.verbose:
            print(f"Ranking layer: Proceeding with {expr}")

        return self.exprs_dict[expr], self.symbols
