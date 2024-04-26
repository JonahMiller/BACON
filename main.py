import argparse
import time
import json
import pandas as pd

import data.datasets as data

from space_of_laws.laws_manager import laws_main
from space_of_data.layer_manager import layer_main

from space_of_data.space_methods.data_space_manager import data_space
from space_of_data.space_methods.mixture import mixture
from space_of_data.space_methods.bacon5 import BACON_5
from space_of_data.space_methods.mcts import MonteCarloTreeSearchNode

import warnings
warnings.filterwarnings("ignore")


def ParseArgs():
    parser = argparse.ArgumentParser(description="Multitude of methods to best find the invariants of a dataset")
    parser.add_argument("--dataset", type=str, choices=data.allowed_data(), metavar="D",
                        help="which dataset would you like to analyse")
    parser.add_argument("--noise", type=float, default=0., metavar="N",
                        help="how much noise to add to the dataset")
    parser.add_argument("--space_of_data", type=str,
                        choices=["bacon.3", "bacon.5", "gp_ranking",
                                 "popularity", "mixture", "mcts"],
                        default=None, metavar="SD",
                        help="how to traverse the space of data")
    parser.add_argument("--space_of_laws", type=str,
                        choices=["bacon.1", "bacon.6", "pysr"],
                        default="bacon.1", metavar="SL",
                        help="how to traverse the space of laws")
    parser.add_argument("--additional_args", type=str, metavar="f", default=None,
                        help="json file for args used in space of data setting")
    args = parser.parse_args()
    return args


def main():
    args = ParseArgs()

    data_func = data.allowed_data()[args.dataset]
    init_data, init_symb = data_func(args.noise)
    initial_df = pd.DataFrame({v: d for v, d in zip(init_symb, init_data)})

    if len(init_symb) > 2 and not args.space_of_data:
        raise Exception("Can only run without data space method if 2 columns in dataframe")

    if args.additional_args:
        with open(args.additional_args, "r") as j:
            all_args = json.loads(j.read())
            layer_args = all_args["layer_args"]
            laws_args = all_args["laws_args"]
            data_space_args = all_args["data_space_args"]
    else:
        layer_args = {}
        laws_args = {}
        data_space_args = {}

    if args.space_of_data == "bacon.5":
        ds = BACON_5(initial_df,
                     laws_main(args.space_of_laws, laws_args),
                     **layer_args)
        ds.run_iterations()
    elif args.space_of_data == "mixture":
        ds = mixture(initial_df,
                     laws_main(args.space_of_laws, laws_args),
                     **data_space_args)
        ds.run_iterations()
    elif args.space_of_data == "mcts":
        root = MonteCarloTreeSearchNode(state=0,
                                        initial_df=initial_df)
        selected_node = root.best_action()
    else:
        ds = data_space(initial_df,
                        layer_main(args.space_of_data, layer_args),
                        laws_main(args.space_of_laws, laws_args),
                        **data_space_args)
        ds.run_iterations()


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Program took {time.time() - start}s!")
