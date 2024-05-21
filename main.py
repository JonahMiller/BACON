import argparse
import time
import json
import pandas as pd

import datasets as data
from utils.gp import gp

from space_of_laws.laws_manager import laws_main
from space_of_data.layer_manager import layer_main

from space_of_data.space_methods.data_space_manager import data_space
from space_of_data.space_methods.bacon5 import BACON_5
from space_of_data.space_methods.mcts import main_mcts

import warnings
warnings.filterwarnings("ignore")


def ParseArgs():
    parser = argparse.ArgumentParser(description="Multitude of methods to best find the invariants of a dataset")
    parser.add_argument("--dataset", type=str, choices=data.allowed_data(), metavar="D",
                        help="which dataset would you like to analyse")
    parser.add_argument("--noise", type=float, default=0., metavar="N",
                        help="how much noise to add to the dataset")
    parser.add_argument("--space_of_data", type=str, default=None, metavar="SD",
                        choices=["bacon.3", "gp_ranking", "min_mse", "weight_mses", "satisfy_equality",
                                 "user_input", "popular", "bacon.5", "mcts"],
                        help="how to traverse the space of data")
    parser.add_argument("--space_of_laws", type=str, default="bacon.1", metavar="SL",
                        choices=["bacon.1", "bacon.6", "pysr"],
                        help="how to traverse the space of laws")
    parser.add_argument("--args", type=str, metavar="f", default=None,
                        help="json file for args used in space of data setting")
    parser.add_argument("--denoise", action="store_true",
                        help="denoises dataset using a gaussian process")
    args = parser.parse_args()
    return args


def main():
    args = ParseArgs()

    data_func = data.allowed_data()[args.dataset]
    init_data, init_symb = data_func(args.noise)
    initial_df = pd.DataFrame({v: d for v, d in zip(init_symb, init_data)})

    if args.denoise:
        initial_df = gp(initial_df).denoise()

    if args.args:
        with open(args.args, "r") as j:
            all_args = json.loads(j.read())
            layer_args = all_args["layer_args"]
            laws_args = all_args["laws_args"]
            data_space_args = all_args["data_space_args"]
            layer_method = all_args["layer_method"]
            laws_method = all_args["laws_method"]
    else:
        layer_args = {}
        laws_args = {}
        data_space_args = {}
        layer_method = args.space_of_data
        laws_method = args.space_of_laws

    if len(init_symb) > 2 and not layer_method:
        raise Exception("Can only run without data space method if 2 columns in dataframe")

    if layer_method == "bacon.5":
        ds = BACON_5(initial_df,
                     laws_main(laws_method, laws_args),
                     **layer_args)
        ds.run_iterations()
    elif layer_method == "mcts":
        init_state = [init_symb[-1], len(init_symb)]
        main_mcts(initial_df, init_state,
                  laws_arg=laws_args,
                  mcts_arg=data_space_args)
    else:
        ds = data_space(initial_df,
                        layer_main(layer_method, layer_args),
                        laws_main(laws_method, laws_args),
                        **data_space_args)
        ds.run_iterations()


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Program took {time.time() - start}s!")
