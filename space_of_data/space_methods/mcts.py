# https://ai-boson.github.io/mcts/
import numpy as np
from statistics import fmean
from sympy import Eq, factor
from itertools import product
from collections import defaultdict

from utils import df_helper
from utils import losses as loss_helper
from space_of_laws.laws_methods.bacon1 import BACON_1
# from space_of_laws.laws_methods.mcts_bacon6 import BACON_6
from space_of_data.layer_methods.layer_select import layer


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

    def best_child(self, c_param=1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt(np.log(self.n()) / c.n()) for c in self.children]
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
        simulation_no = 100
        for idx in range(simulation_no):
            if idx % 10 == 0:
                print(f"SIMULATION NUMBER {idx}")
            v = self._tree_policy()
            reward = v.rollout()
            print(reward)
            v.backpropagate(reward)
        return self.best_child(c_param=0.).state, df_dict[str(self.best_child(c_param=0.).state)]


def get_legal_actions(var_list):
    epsilon = [0.001]
    delta = [0.08]
    actions = list(product(epsilon, delta))
    return actions


def is_game_over(var_list):
    if "dfs" in df_dict[str(var_list)]:
        return False
    return True


def move(var_list, action):
    dict_entry = df_dict[str(var_list)]
    dfs = dict_entry["dfs"]
    if "eqns" in dict_entry:
        eqns = dict_entry["eqns"]
    else:
        eqns = []
    symbols = list(sum(sym for sym in list(var_list)).free_symbols)
    new_dfs, final_eqns = [], []
    df_len = len(dfs[0].columns)
    for df in dfs:
        if df_len > 2:
            layer_in_context = layer(df, lambda df, col_1, col_2, afs:
                                     bacon_1(df, col_1, col_2, afs,
                                             epsilon=action[0], delta=action[1]),
                                     symbols, "prop_mse")
            new_df, symbols = layer_in_context.run_single_iteration()
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
            # eqn = df_helper.timeout(BACON_6(initial_df, eqns).run_iteration,
            #                         timeout_duration=3, default="fail")
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
            # print("equations not combining")
            df_dict[str(var_list)] = {"final_eqns": eqns, "score": -30000}
            return -30000

        print(f"Final form is {eqn.rhs} = {factor(eqn.lhs)}")

        score = -loss_helper.loss_calc(initial_df, eqn).loss()

        if isinstance(score, complex):
            # print("score is complex - likely solvable by hand")
            df_dict[str(var_list)] = {"final_eqns": eqns, "score": -20000, "eqn_form": eqn}
            return -20000
        else:
            num_var = len(eqn.free_symbols)
            if num_var < len(initial_df.columns):
                score -= 2000*(len(initial_df.columns) - num_var)
            df_dict[str(var_list)] = {"final_eqns": eqns, "score": score, "eqn_form": eqn}
            return score
    except Exception:
        df_dict[str(var_list)] = {"final_eqns": eqns, "score": -100000, "eqn_form": eqn}
        return -100000


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
    print(f"Final form is {eqn.rhs} = {factor(eqn.lhs)} with score {final_score}")
