# https://ai-boson.github.io/mcts/
import numpy as np
from itertools import product
from collections import defaultdict

from utils import df_helper as df_helper
from space_of_laws.laws_methods.bacon1 import BACON_1
from space_of_data.layer_methods.popularity import popular_layer


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
    def __init__(self, state, initial_df, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()

        self.initial_df = initial_df
        self.dfs = [initial_df]
        self.symbols = list(sum(sym for sym in list(initial_df)).free_symbols)
        self.iteration_level = 0
        self.eqns = []
        return

    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, self.initial_df, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
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
        simulation_no = 100
        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child(c_param=0.)

    def get_legal_actions(self):
        epsilon = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
        delta = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]
        actions = list(product(epsilon, delta))
        return actions

    def is_game_over(self):
        if self.iteration_level == len(self.initial_df.columns):
            return True
        return False

    def game_result(self):
        return df_helper.lolkay(self.initial_df, self.eqns)

    def move(self, action):
        for df in self.dfs:
            self.iteration_level += 1
            new_dfs = []
            layer_in_context = popular_layer(df, lambda df, col_1, col_2, afs:
                                             bacon_1(df, col_1, col_2, afs,
                                                     epsilon=action[0], delta=action[1]),
                                             self.symbols)
            new_df, self.symbols = layer_in_context.run_single_iteration()
            new_dfs.extend(new_df)

        self.dfs = new_df


def main():
    initial_state = 0
    root = MonteCarloTreeSearchNode(state=initial_state)
    selected_node = root.best_action()
    return
