import argparse
import sympy as sym
import pandas as pd

import data.datasets as data

from bacon.bacon3 import BACON_3
from bacon.bacon4 import BACON_4

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
    parser.add_argument("--bacon_4_verbose", action="store_true",
                        help="activates verbose mode for the program's decisions at the BACON 4 level")
    
    args = parser.parse_args()
    return args

def main():
    args = ParseArgs()

    data_func = data.allowed_data()[args.dataset]
    init_data, init_symb = data_func(args.noise)
    initial_df = pd.DataFrame({v: d for v, d in zip(init_symb, init_data)})

    if args.bacon == 3:
        bacon = BACON_3(initial_df, 
                        bacon_1_info=args.bacon_1_verbose,
                        bacon_3_info=args.bacon_3_verbose)
    elif args.bacon == 4:
        bacon = BACON_4(initial_df, 
                        bacon_1_info=args.bacon_1_verbose,
                        bacon_3_info=args.bacon_3_verbose,
                        bacon_4_info=args.bacon_4_verbose)
    else:
        return Exception("Invalid BACON value specified. Only allowed 3 or 4.")
        
    bacon.bacon_iterations()


if __name__ == "__main__":
    main()