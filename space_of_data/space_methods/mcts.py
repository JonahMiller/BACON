# https://ai-boson.github.io/mcts/
import numpy as np
from statistics import fmean
from sympy import Eq
from itertools import product
from collections import defaultdict

from utils import df_helper
from utils import losses as loss_helper
from space_of_laws.laws_methods.bacon1 import BACON_1
from space_of_data.layer_methods.popularity import popular_layer


df_dict = {}


def update_df_dict(key, value):
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
        update_df_dict(key=f'{state}', value={"dfs": [self.initial_df], "eqns": []})
        print(state)
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
        return

    def untried_actions(self):
        self._untried_actions = node(self.initial_df, self.state).get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = node(self.initial_df, self.state).move(action)
        child_node = MonteCarloTreeSearchNode(self.initial_df, next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return node(self.initial_df, self.state).is_game_over()

    def rollout(self):
        current_rollout_state = node(self.initial_df, self.state)
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


class node:
    def __init__(self, initial_df, var_list):
        self.initial_df = initial_df
        self.var_list = var_list
        self.dfs = df_dict[str(var_list)]["dfs"]
        self.eqns = df_dict[str(var_list)]["eqns"]
        self.symbols = list(sum(sym for sym in list(initial_df)).free_symbols)

    def get_legal_actions(self):
        epsilon = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
        delta = [0.01, 0.05, 0.1]
        # epsilon = [epsilon for epsilon in epsilon_poss if epsilon <= self.epsilon]
        # delta = [delta for delta in delta_poss if delta <= self.delta]
        actions = list(product(epsilon, delta))
        return actions

    def is_game_over(self):
        if len(self.dfs) == 0:
            return True
        return False

    def move(self, action):
        new_dfs = []
        for df in self.dfs:
            if len(df.columns) > 2:
                layer_in_context = popular_layer(df, lambda df, col_1, col_2, afs:
                                                 bacon_1(df, col_1, col_2, afs,
                                                         epsilon=action[0], delta=action[1]),
                                                 self.symbols)
                new_df, self.symbols = layer_in_context.run_single_iteration()
                new_dfs.extend(new_df)
            else:
                ave_df = df_helper.average_df(df)
                results = bacon_1(ave_df, ave_df.columns[1], ave_df.columns[0], self.symbols,
                                  epsilon=action[0], delta=action[1])
                self.eqns.append(Eq(results[1], fmean(results[0])))
        self.dfs, self.eqns = df_helper.check_const_col(new_dfs, self.eqns, 0.1, logging=False)
        var_list = [df.columns.tolist()[-1] for df in self.dfs]
        update_df_dict(key=str(var_list), value={"dfs": self.dfs, "eqns": self.eqns})
        return var_list

    def game_result(self):
        key_var = self.initial_df.columns[-1]
        try:
            eqn = loss_helper.simplify_eqns(self.initial_df, self.eqns, key_var).iterate_through_dummys()
            loss = timeout(loss_helper.loss_calc(self.initial_df, eqn).loss(), timeout_duration=1, default="fail")
            print(f"loss: {loss}")
            if loss == "fail":
                return -0.5
            if abs(loss) < 0.1:
                return 1
            if abs(loss) < 1:
                return 0.75
            if abs(loss) < 10:
                return 0.5
            return 0
        except Exception:
            print(f"fail: {self.eqns}")
            return -1


# https://stackoverflow.com/a/13821695
def timeout(func, args=(), kwargs=None, timeout_duration=1, default=None):
    kwargs = kwargs or {}
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        result = default
    finally:
        signal.alarm(0)

    return result
