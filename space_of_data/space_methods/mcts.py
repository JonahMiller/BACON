# https://ai-boson.github.io/mcts/
import numpy as np
from statistics import fmean
from sympy import Eq
from itertools import product
from collections import defaultdict

from utils import df_helper
from utils import losses as loss_helper
from space_of_laws.laws_methods.bacon1 import BACON_1
from space_of_data.layer_methods.min_mse import min_mse_layer


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
        self._untried_actions = get_legal_actions()
        return self._untried_actions

    def q(self):
        score = 0
        for key, val in self._results.items():
            score += key*val
        # print(f"score is {score}")
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
            possible_moves = get_legal_actions()
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

    def best_child(self, c_param=0.8):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
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
        simulation_no = 500
        for _ in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            print(reward)
            v.backpropagate(reward)
        return self.best_child(c_param=0.).state, df_dict[str(self.best_child(c_param=0.).state)]


def get_legal_actions():
    epsilon = [0.00001, 0.0001, 0.001, 0.01]
    delta = [0.05, 0.07, 0.1, 0.14, 0.18, 0.22]
    Delta = [0.1, 0.2, 0.3]
    actions = list(product(epsilon, delta, Delta))
    return actions


def is_game_over(var_list):
    if "dfs" in df_dict[str(var_list)]:
        return False
    print("game is over!")
    return True


def move(var_list, action):
    print(action)
    dict_entry = df_dict[str(var_list)]
    dfs = dict_entry["dfs"]
    if "eqns" in dict_entry:
        eqns = dict_entry["eqns"]
    else:
        eqns = []
    symbols = list(sum(sym for sym in list(var_list)).free_symbols)
    new_dfs = []
    final_eqns = []
    df_len = len(dfs[0].columns)

    for df in dfs:
        if df_len > 2:
            layer_in_context = min_mse_layer(df, lambda df, col_1, col_2, afs:
                                             bacon_1(df, col_1, col_2, afs,
                                                     epsilon=action[0], delta=action[1]),
                                             symbols)
            new_df, symbols = layer_in_context.run_single_iteration()

            if not new_df:
                print(f"No result with this action 1 - rerunning for new action: {var_list}")
                return var_list

            new_dfs.extend(new_df)

        elif df_len == 2:
            ave_df = df_helper.average_df(df)
            results = bacon_1(ave_df, ave_df.columns[0], ave_df.columns[1], symbols,
                              epsilon=action[0], delta=action[1])

            if results[0] is None:
                print(f"No result with this action 2 - rerunning for new action: {var_list}")
                return var_list

            final_eqns.append(Eq(results[1], fmean(results[0])))

    if df_len > 2:
        dfs, new_eqns = df_helper.check_const_col(new_dfs, eqns, delta=action[2], logging=False)
        var_list = [eqn.lhs for eqn in new_eqns] + [df.columns.tolist()[-1] for df in dfs] + [df_len - 1]
        if len(new_eqns) != 0:
            df_dict[str(var_list)] = {"dfs": dfs, "eqns": new_eqns}
        else:
            df_dict[str(var_list)] = {"dfs": dfs}
        return var_list
    else:
        final_eqns = eqns + final_eqns
        var_list = [eqn.lhs for eqn in final_eqns] + [1]
        df_dict[str(var_list)] = {"final_eqns": final_eqns}
        return var_list


def score_func(score, num_dummy, num_var, actual_var):
    reward = 0

    if num_var < actual_var:
        reward -= min(0.1*(actual_var - num_var), 0.4)
    if num_dummy > 2:
        reward -= min(0.2*(num_dummy - 2), 0.5)

    if score < 0.1:
        reward = 1
    elif score < 1:
        reward += 0.8
    elif score < 5:
        reward += 0.6
    elif score < 10:
        reward += 0.4
    return reward


def game_result(var_list, initial_df):
    dict_entry = df_dict[str(var_list)]
    eqns = dict_entry["final_eqns"]
    num_dummy = len(eqns) - 1
    # print(eqns)
    for var in list(reversed(initial_df.columns)):
        try:
            eqn = df_helper.timeout(loss_helper.simplify_eqns(initial_df,
                                                              eqns,
                                                              var).iterate_through_dummys,
                                    timeout_duration=1, default="fail")
            if eqn == "fail":
                # print("equations not combining")
                return -0.75

            score = loss_helper.loss_calc(initial_df, eqn).loss()

            if isinstance(score, complex):
                # print("score is complex - likely solvable by hand")
                return -0.5

            else:
                num_var = len(eqn.free_symbols)
                return score_func(score, num_dummy, num_var, len(initial_df.columns))
        except Exception as e:
            print(e)
            print(eqns)
            return -1


def main_mcts(initial_df, init_state):
    global df_dict
    dict_state = {"dfs": [initial_df]}
    df_dict = {str(init_state): dict_state}
    while dict_state["dfs"]:
    # for _ in range(1):
        print("@@@@@@@@@@@@@@@@@@@@@@@")
        root = MonteCarloTreeSearchNode(initial_df, state=init_state)
        init_state, dict_state = root.best_action()
        print(init_state)
        df_dict = {str(init_state): dict_state}
    print(f"FINAL NODE STATE IS {init_state}")
