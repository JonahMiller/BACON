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


def update_dict(key, value):
    if key not in df_dict:
        df_dict[key] = value


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
        self.state_class = node(self.initial_df, self.state)
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

    def untried_actions(self):
        self._untried_actions = self.state_class.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state_class.move(action)
        child_node = MonteCarloTreeSearchNode(self.initial_df, next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state_class.is_game_over()

    def rollout(self):
        current_rollout_state = self.state_class
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            new_state = current_rollout_state.move(action)
            current_rollout_state = node(self.initial_df, new_state)
        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=1):
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


class node:
    def __init__(self, initial_df, var_list):
        self.initial_df = initial_df
        self.var_list = var_list
        dict_entry = df_dict[str(var_list)]
        self.dfs = dict_entry["dfs"]
        self.eqns = dict_entry["eqns"]
        if dict_entry["dfs"]:
            self.symbols = list(sum(sym for sym in list(var_list)).free_symbols)

    def get_legal_actions(self):
        # epsilon = [0.00001, 0.0001, 0.001]
        epsilon = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
        # delta = [0.01, 0.04, 0.07]
        delta = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.2]
        actions = list(product(epsilon, delta))
        return actions

    def is_game_over(self):
        if df_dict[str(self.var_list)]["dfs"]:
            return False
        return True

    def move(self, action):
        print(action)
        new_dfs = []
        final_eqns = []
        df_len = len(self.dfs[0].columns)
        for df in self.dfs:
            if df_len > 2:
                layer_in_context = min_mse_layer(df, lambda df, col_1, col_2, afs:
                                                 bacon_1(df, col_1, col_2, afs,
                                                         epsilon=action[0], delta=action[1]),
                                                 self.symbols)
                new_df, self.symbols = layer_in_context.run_single_iteration()
                new_dfs.extend(new_df)
            elif df_len == 2:
                ave_df = df_helper.average_df(df)
                results = bacon_1(ave_df, ave_df.columns[1], ave_df.columns[0], self.symbols,
                                  epsilon=action[0], delta=action[1])
                final_eqns.append(Eq(results[1], fmean(results[0])))

        if df_len > 2:
            self.dfs, self.eqns = df_helper.check_const_col(new_dfs, self.eqns, 0.1, logging=False)
            var_list = [eqn.lhs for eqn in self.eqns] + [df.columns.tolist()[-1] for df in self.dfs] + [df_len - 1]
            update_dict(key=str(var_list), value={"dfs": self.dfs, "eqns": self.eqns})
            return var_list
        else:
            final_eqns = self.eqns + final_eqns
            var_list = [eqn.lhs for eqn in final_eqns] + [1]
            update_dict(key=str(var_list), value={"dfs": [], "eqns": final_eqns})
            return var_list

    @staticmethod
    def score_func(score, num_dummy, num_var, actual_var):
        reward = 0

        if num_var < actual_var:
            reward -= min(0.1*(actual_var - num_var), 0.4)
        if num_dummy > 2:
            reward -= min(0.2*(num_dummy - 2), 0.5)

        if score < 0.1:
            reward = 1
        if score < 1:
            reward += 0.8
        if score < 5:
            reward += 0.6
        if score < 10:
            reward += 0.4
        return reward

    def game_result(self):
        num_dummy = len(self.eqns) - 1
        print(self.eqns)
        for var in list(reversed(self.initial_df.columns)):
            try:
                eqn = df_helper.timeout(loss_helper.simplify_eqns(self.initial_df,
                                                                  self.eqns,
                                                                  var).iterate_through_dummys,
                                        timeout_duration=1, default="fail")
                if eqn == "fail":
                    print("equations not combining")
                    return -0.5

                score = loss_helper.loss_calc(self.initial_df, eqn).loss()

                if isinstance(score, complex):
                    print("score is complex - likely solvable by hand")
                    return -0.25

                else:
                    num_var = len(eqn.free_symbols)
                    return node.score_func(score, num_dummy, num_var, len(self.initial_df.columns))
            except Exception as e:
                print(e)
                return -1


def main_mcts(initial_df, init_state):
    global df_dict
    dict_state = {"dfs": [initial_df], "eqns": []}
    df_dict = {str(init_state): dict_state}
    # while dict_state["dfs"]:
    for _ in range(1):
        print("@@@@@@@@@@@@@@@@@@@@@@@")
        root = MonteCarloTreeSearchNode(initial_df, state=init_state)
        init_state, dict_state = root.best_action()
        df_dict = {str(init_state): dict_state}
        # print(df_dict)
        # print(init_state)
    print(f"FINAL NODE STATE IS {init_state}")
