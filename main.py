import argparse
import time
import json
import pandas as pd

import data.datasets as data

from space_of_laws.laws_manager import laws_main
from space_of_data.layer_manager import layer_main

from space_of_data.data_space_manager import data_space

import warnings
warnings.filterwarnings("ignore")


def ParseArgs():
    parser = argparse.ArgumentParser(description="Multitude of methods to best find the invariants of a dataset")
    parser.add_argument("--dataset", type=str, choices=data.allowed_data(), metavar="D",
                        help="which dataset would you like to analyse")
    parser.add_argument("--noise", type=float, default=0., metavar="N",
                        help="how much noise to add to the dataset")
    parser.add_argument("--delta", type=float, default=0.1, metavar="d",
                        help="delta error tolerance for constant values in data space")
    parser.add_argument("--space_of_data", type=str,
                        choices=["bacon.3", "bacon.5", "gp_ranking", "popularity"],
                        default="gp_ranking", metavar="SD",
                        help="how to traverse the space of data")
    parser.add_argument("--space_of_laws", type=str,
                        choices=["bacon.1", "pysr"],
                        default="bacon.1", metavar="SL",
                        help="how to traverse the space of laws")
    parser.add_argument("--data_space_verbose", action="store_true",
                        help="activates verbose mode for the data space manager")
    parser.add_argument("--layer_space_args", type=str, metavar="f1", default=None,
                        help="json file for args used in space of data setting")
    parser.add_argument("--laws_space_args", type=str, metavar="f2", default=None,
                        help="json file for args used in space of laws setting")
    args = parser.parse_args()
    return args


def main():
    args = ParseArgs()

    data_func = data.allowed_data()[args.dataset]
    init_data, init_symb = data_func(args.noise)
    initial_df = pd.DataFrame({v: d for v, d in zip(init_symb, init_data)})

    if args.layer_space_args:
        with open(args.layer_space_args, "r") as j:
            layer_args = json.loads(j.read())
    else:
        layer_args = None

    if args.laws_space_args:
        with open(args.laws_space_args, "r") as j:
            laws_args = json.loads(j.read())
    else:
        laws_args = None

    ds = data_space(initial_df,
                    layer_main(args.space_of_data, layer_args),
                    laws_main(args.space_of_laws, laws_args),
                    args.data_space_verbose)

    ds.run_iterations()


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Program took {time.time() - start}s!")
