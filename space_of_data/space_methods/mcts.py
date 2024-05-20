# https://ai-boson.github.io/mcts/
import numpy as np
from statistics import fmean
from sympy import Eq, factor, Symbol
from itertools import product
from collections import defaultdict

from utils import df_helper
from utils import laws_helper
from utils import losses as loss_helper
from space_of_laws.laws_methods.bacon1 import BACON_1
from space_of_data.space_methods.mcts_layer import layer, construct_dfs


def bacon_1(df, col_1, col_2, all_found_symbols,
            epsilon, delta, verbose=False):
    """
    Runs an instance of BACON.1 on the specified columns
    col_1 and col_2 in the specified dataframe df.
    """
    if verbose:
        unused_df = df.iloc[:, :-2]
        col_names = unused_df.columns.tolist()
        col_ave = [unused_df.loc[:, name].mean() for name in col_names]
        if len(col_names) != 0:
            print(f"Laws manager: Running BACON 1 on variables [{col_1}, {col_2}] and")
            print(f"              unused variables {col_names} set as {col_ave}.")
        else:
            print(f"Laws manager: Running BACON 1 on variables [{col_1}, {col_2}]")
    bacon_1_instance = BACON_1(df[[col_1, col_2]], all_found_symbols,
                               epsilon, delta,
                               verbose=verbose)
    return bacon_1_instance.bacon_iterations()


class MonteCarloTreeSearchNode():
    def __init__(self, initial_df, state, parent=None, parent_action=None):
        self.initial_df = initial_df
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(float)
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

    def untried_actions(self):
        self._untried_actions = get_legal_actions(self.state)
        return self._untried_actions

    def q(self):
        score = 0
        for key, val in self._results.items():
            score += key*val
        return score

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = move(self.state, action)
        child_node = MonteCarloTreeSearchNode(self.initial_df, next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return is_game_over(self.state)

    def rollout(self):
        current_rollout_state = self.state
        while not is_game_over(current_rollout_state):
            possible_moves = get_legal_actions(current_rollout_state)
            action = self.rollout_policy(possible_moves)
            new_state = move(current_rollout_state, action)
            current_rollout_state = new_state
        return game_result(current_rollout_state, self.initial_df)

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=6):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt(np.log(self.n()) / c.n()) for c in self.children]
        if c_param == 0:
            print(choices_weights)
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 200
        for idx in range(simulation_no):
            if idx % 20 == 0:
                print(f"SIMULATION NUMBER {idx}")
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        print("GETTING BEST CHILD STATES")
        print([c.state for c in self.children])
        print([c._results for c in self.children])
        return self.best_child(c_param=0.).state, df_dict[str(self.best_child(c_param=0.).state)]


def get_legal_actions(var_list):
    # eps_del_list = [(0.02, 0.04), (0.02, 0.08), (0.05, 0.04), (0.05, 0.08)]
    eps_del_list = [(0.02, 0.04)]
    dict_entry = df_dict[str(var_list)]
    if "dfs" not in dict_entry:
        return []
    dfs = dict_entry["dfs"]
    symbols = list(sum(sym for sym in list(var_list)).free_symbols)
    df_len = len(dfs[0].columns)
    if df_len == 2:
        return eps_del_list

    exprs = {idx: [] for idx in range(len(dfs))}
    for idx, df in enumerate(dfs):
        if df_len > 2:
            expr_list = []
            lin_relns = {}
            for eps, delt in eps_del_list:
                layer_in_context = layer(df, lambda df, col_1, col_2, afs:
                                         bacon_1(df, col_1, col_2, afs,
                                                 epsilon=eps, delta=delt),
                                         symbols, "mcts")
                exprs_found, lin_reln = layer_in_context.get_relations()
                lin_relns = lin_relns | lin_reln
                expr_list.extend(list(exprs_found.keys()))

            expr_list_ = list(set(expr_list))

            exprs_list = [[expr, lin_relns[expr]] if expr in lin_relns
                          else expr for expr in expr_list_]

            exprs[idx] = exprs_list

    actions = list(product(*exprs.values()))

    new_actions = []
    if len(dfs) > 1:
        for action in actions:
            new_action = []
            linear_vars = []
            state = ""
            for expr in action:
                if isinstance(expr, list):
                    symb = expr[1][0]
                    if symb == Symbol("e"):
                        state == "illegal"
                        break
                    if symb in linear_vars:
                        new_symb = laws_helper.new_symbol(linear_vars)
                        main_expr = expr[0].subs(symb, new_symb)
                        new_expr = [main_expr,
                                    [new_symb, main_expr, expr[1][2], expr[1][3]]]
                        new_action.append(new_expr)
                    else:
                        linear_vars.append(symb)
                        new_action.append(expr)
                else:
                    new_action.append(expr)
            if state != "illegal":
                new_actions.append(new_action)
    else:
        new_actions = actions

    return new_actions


def is_game_over(var_list):
    if "final_eqns" in df_dict[str(var_list)]:
        return True
    return False


def move(var_list, action):
    dict_entry = df_dict[str(var_list)]
    dfs = dict_entry["dfs"]
    if "eqns" in dict_entry:
        eqns = dict_entry["eqns"]
    else:
        eqns = []
    new_dfs, final_eqns = [], []
    symbols = list(sum(sym for sym in list(var_list)).free_symbols)
    df_len = len(dfs[0].columns)
    for idx, df in enumerate(dfs):
        if df_len > 2:
            if isinstance(action[idx], list):
                df_constructor = construct_dfs(df, expr=action[idx][0],
                                               lin_relns=action[idx][1])
            else:
                df_constructor = construct_dfs(df, expr=action[idx])
            new_df = df_constructor.construct_dfs()
            new_dfs.extend(new_df)

        elif df_len == 2:
            ave_df = df_helper.average_df(df)
            results = bacon_1(ave_df, ave_df.columns[0], ave_df.columns[1], symbols,
                              epsilon=action[0], delta=action[1])
            final_eqns.append(Eq(results[1], fmean(results[0])))

    if df_len > 2:
        dfs, eqns = df_helper.check_const_col(new_dfs, eqns, delta=0.1, logging=False)
        var_list = [eqn.lhs for eqn in eqns] + \
                   [df.columns.tolist()[-1] for df in dfs] + \
                   [df_len - 1]
        if len(eqns) != 0 and len(dfs) != 0:
            df_dict[str(var_list)] = {"dfs": dfs, "eqns": eqns}
        elif len(eqns) == 0 and len(dfs) != 0:
            df_dict[str(var_list)] = {"dfs": dfs}
        elif len(eqns) != 0 and len(dfs) == 0:
            df_dict[str(var_list)] = {"final_eqns": eqns}
        return var_list
    else:
        final_eqns = eqns + final_eqns
        var_list = [eqn.lhs for eqn in final_eqns] + \
                   [1]
        df_dict[str(var_list)] = {"final_eqns": final_eqns}
        return var_list


def game_result(var_list, initial_df):
    dict_entry = df_dict[str(var_list)]
    var = initial_df.columns.to_list()[-1]
    if "score" in dict_entry:
        return dict_entry["score"]

    eqns = dict_entry["final_eqns"]

    try:
        if len(eqns) > 0:
            eqn = df_helper.timeout(loss_helper.simplify_eqns(initial_df,
                                                              eqns,
                                                              var).iterate_through_dummys,
                                    timeout_duration=3, default="fail")
        else:
            eqn = df_helper.timeout(loss_helper.simplify_eqns(initial_df,
                                                              eqns,
                                                              var).iterate_through_dummys,
                                    timeout_duration=3, default="fail")

        if eqn == "fail":
            print("equations not combining in good time")
            df_dict[str(var_list)] = {"final_eqns": eqns, "score": -3}
            return -4

        loss = loss_helper.loss_calc(initial_df, eqn).loss()

        if isinstance(loss, complex):
            print("score is complex - likely solvable by hand")
            df_dict[str(var_list)] = {"final_eqns": eqns, "score": -2, "eqn_form": eqn}
            return -2

        if loss < 10:
            score = 2
        elif loss < 14:
            score = 0.8
        elif loss < 16:
            score = 0.5
        elif loss < 20:
            score = 0.2
        elif loss < 30:
            score = 0.1
        else:
            score = 0

        num_var = len(eqn.free_symbols)
        if num_var < len(initial_df.columns) - 1:
            score -= 1.5*(len(initial_df.columns) - num_var)
        df_dict[str(var_list)] = {"final_eqns": eqns, "score": score, "eqn_form": eqn, "loss": loss}
        print(f"Final form is {eqn.rhs} = {factor(eqn.lhs)} with score {score} and loss {loss}")
        return score
    except Exception:
        df_dict[str(var_list)] = {"final_eqns": eqns, "score": -10}
        return -10


def main_mcts(initial_df, init_state):
    global df_dict
    dict_state = {"dfs": [initial_df]}
    df_dict = {str(init_state): dict_state}
    while "dfs" in dict_state:
        print("@@@@@@@@@@@@@@@@@@@@@@@")
        root = MonteCarloTreeSearchNode(initial_df, state=init_state)
        init_state, dict_state = root.best_action()
        print(init_state)
        df_dict = {str(init_state): dict_state}
    print(f"FINAL NODE STATE IS {init_state}")
    eqn = df_dict[str(init_state)]['eqn_form']
    final_score = df_dict[str(init_state)]['score']
    loss = df_dict[str(init_state)]['loss']
    print(f"Final form is {eqn.rhs} = {factor(eqn.lhs)} with score {final_score} and loss {loss}")
