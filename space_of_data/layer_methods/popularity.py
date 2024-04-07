import pandas as pd

from utils import df_helper as df_helper


class popular_layer:
    def __init__(self, df, laws_method, verbose=False):
        self.df = df
        self.laws_method = laws_method
        self.verbose = verbose

    def find_exprs(self):
        exprs_found = {}
        lin_relns = {}

        s_dfs = df_helper.deconstruct_df(self.df)

        for df in s_dfs:
            data, symb, lin = self.laws_method(df, df.columns[-1], df.columns[-2])

            if isinstance(lin, list):
                symb = lin[2]

            if symb in exprs_found:
                exprs_found[symb] += 1
            else:
                exprs_found[symb] = 1
                if isinstance(lin, list):
                    lin_relns[symb] = [lin[1], lin[2], lin[4], lin[5]]

        best_expr = max(exprs_found, key=exprs_found.get)

        if best_expr in lin_relns:
            lin_reln = lin_relns[best_expr]
        else:
            lin_reln = None

        return best_expr, lin_reln

    def construct_dfs(self):
        new_dfs = {}
        expr, lin_reln = self.find_exprs()

        new_dfs = []

        if lin_reln:
            df = df_helper.update_df_with_multiple_expr(lin_reln[2],
                                                        lin_reln[3],
                                                        self.df)
            s_dfs = df_helper.deconstruct_df(df)

            new_dummy_col, new_expr_col = pd.DataFrame(), pd.DataFrame()

            for s_df in s_dfs:
                dummy_col, expr_col = df_helper.linear_relns(s_df,
                                                             lin_reln[0],
                                                             lin_reln[1])
                new_dummy_col = pd.concat([new_dummy_col, dummy_col])
                new_expr_col = pd.concat([new_expr_col, expr_col])

            n_df1 = df.iloc[:, :-2].join(new_dummy_col)
            n_df2 = df.iloc[:, :-2].join(new_expr_col)

            new_dfs = [n_df1, n_df2]

        else:
            n_df = df_helper.update_df_with_single_expr(expr, self.df)
            new_dfs = [n_df]

        return new_dfs

    def run_single_iteration(self):
        return self.construct_dfs()