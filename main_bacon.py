import argparse
import time
import pandas as pd

import data.datasets as data
from utils.gp import ranking

from space_of_data.bacon3 import BACON_3
from space_of_data.bacon5 import BACON_5

import warnings
warnings.filterwarnings("ignore")


def ParseArgs():
    parser = argparse.ArgumentParser(description="Pat Langley's BACON programs simulator")
    parser.add_argument("--dataset", type=str, choices=data.allowed_data(), metavar="D",
                        help="which dataset would you like to analyse")
    parser.add_argument("--bacon", type=int, choices=[3, 4, 5], default=3, metavar="B",
                        help="which BACON version to run on")
    parser.add_argument("--noise", type=float, default=0., metavar="N",
                        help="how much noise to add to dataset")
    parser.add_argument("--epsilon", type=float, default=0.0001, metavar="N",
                        help="how much epsilon error to allow in calculations via BACON")
    parser.add_argument("--delta", type=float, default=0.01, metavar="N",
                        help="how much delta error to allow in calculations via BACON")
    parser.add_argument("--bacon_1_verbose", action="store_true",
                        help="activates verbose mode for the program's decisions at the BACON 1 level")
    parser.add_argument("--bacon_3_verbose", action="store_true",
                        help="activates verbose mode for the program's decisions at the BACON 3 level")
    parser.add_argument("--bacon_5_verbose", action="store_true",
                        help="activates verbose mode for the program's decisions at the BACON 5 level")
    parser.add_argument("--bacon_5_ranking", action="store_true",
                        help="ranks tree roots for best performance using BACON 5")

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
    elif args.bacon == 5:
        if args.bacon_5_ranking:
            initial_df, ranking_df = ranking(initial_df).rank_new_df()
        else:
            ranking_df = None

        bacon = BACON_5(initial_df,
                        ranking_df=ranking_df,
                        epsilon=args.epsilon,
                        delta=args.delta,
                        bacon_1_info=args.bacon_1_verbose,
                        bacon_5_info=args.bacon_5_verbose)

    bacon.bacon_iterations()


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Program took {time.time() - start}s!")
