import argparse
import sympy as sym
import pandas as pd

import data.datasets as data

from bacon.bacon1 import BACON_1
from bacon.bacon3 import BACON_3
import bacon.losses as bl


def ParseArgs():
    parser = argparse.ArgumentParser(description="Pat Langley's BACON programs simulator")
    parser.add_argument("--dataset", type=str, choices=data.allowed_data(), metavar="D",
                        help="which dataset would you like to analyse")
    parser.add_argument("--bacon", type=int, default=3, metavar="B",
                        help="which BACON version to run on")
    parser.add_argument("--noise", type=float, default=0., metavar="N",
                        help="how much noise to add to dataset")
    parser.add_argument("--bacon_1_verbose", action="store_true",
                        help="activates verbose mode for the program's decisions at the BACON 1 level")
    parser.add_argument("--bacon_3_verbose", action="store_true",
                        help="activates verbose mode for the program's decisions at the BACON 3 level")
    
    args = parser.parse_args()
    return args

def main():
    args = ParseArgs()

    data_func = data.allowed_data()[args.dataset]
    init_data, init_symb = data_func(args.noise)
    initial_df = pd.DataFrame({v: d for v, d in zip(init_symb, init_data)})

    if args.bacon == 1:
        bacon_func = BACON_1
    elif args.bacon == 3:
        bacon_func = BACON_3
    else:
        return Exception("Invalid BACON value specified. Only allowed 1 or 3.")
    
    bacon = bacon_func(init_data, init_symb, 
                       bacon_1_info=args.bacon_1_verbose,
                       bacon_3_info=args.bacon_3_verbose)
    bacon.bacon_iterations()

    const_eqns = bacon.eqns
    key_var = init_symb[-1]

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("The constant equations found are:")
    for eqn in const_eqns:
        print(f"{eqn.rhs} = {eqn.lhs}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    eqn = bl.simplify_eqns(initial_df, const_eqns, key_var).iterate_through_dummys()
    loss = bl.loss_calc(initial_df, eqn).loss()
    print(f"Final form is {eqn.rhs} = {eqn.lhs} with loss {loss}.")
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

if __name__ == "__main__":
    main()