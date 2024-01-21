from sympy import solve, Eq, lambdify
from itertools import filterfalse
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error as mse


class simplify_eqns:
    def __init__(self, initial_df, eqns, key_var):
        self.df = initial_df
        self.eqns = eqns
        self.key_var = key_var
    
    def get_dummy_vars(self):
        '''
        Determines which variables are dummies to be cancelled out rather than
        key symbols from the initial data.
        '''
        all_vars = set()
        for eqn in self.eqns:
            for fs in eqn.free_symbols:
                all_vars.add(fs)
        self.dummy_vars = list(filterfalse(list(self.df.columns).__contains__,
                                           list(all_vars)))
        
    def elim_dummy_var(self, dummy_var):
        '''
        Iteratively removes the dummy variables by iterating through each term.
        '''
        eqns_with_var = [eqn for eqn in self.eqns if dummy_var in eqn.free_symbols]
        assert len(eqns_with_var) == 2, "Too many variables to remove"
        self.eqns = list(filterfalse(eqns_with_var.__contains__,
                                    self.eqns))
        self.eqns.append(self.equate_eqns(eqns_with_var[0], eqns_with_var[1], dummy_var))

    def equate_eqns(self, eqn1, eqn2, dummy_var):
        '''
        Equates two equations via the dummy variable specified.
        '''
        solved_eqn1, solved_eqn2 = solve(eqn1, dummy_var), solve(eqn2, dummy_var)
        return Eq(solved_eqn1[0], solved_eqn2[0])
    
    def iterate_through_dummys(self):
        '''
        Iterates through each dummy variable until none remain.
        '''
        self.get_dummy_vars()
        for dummy_var in self.dummy_vars:
            self.elim_dummy_var(dummy_var)
        assert len(self.eqns) == 1, "Not all equations combined"
        final_form = solve(self.eqns[0], self.key_var)
        return Eq(final_form[0], self.key_var)


class loss_calc:
    def __init__(self, df, eqn):
        self.df = df.drop(eqn.rhs, axis=1)
        self.key_vals = df[eqn.rhs].to_numpy()
        self.eqn = eqn

    def calc_eqn_vals(self):
        '''
        Calculates the value of the equation determined by BACON.3 with the data
        passed into the system.
        '''
        init_vars = list(self.df.columns)
        f = lambdify([tuple(init_vars)], self.eqn.lhs)
        self.calc_vals = np.array([(f(tuple(val))) for val in self.df[init_vars].to_numpy().tolist()])

    def loss(self):
        '''
        Calculates the loss between the proposed BACON.3 equation and the real 
        values of the dependent term.
        '''
        self.calc_eqn_vals()
        return mse(self.calc_vals, self.key_vals)
