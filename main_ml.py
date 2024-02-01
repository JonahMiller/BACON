import argparse

import data.datasets as data

from ml_methods.symbolic_regression import main as pysrmain


def ParseArgs():
    parser = argparse.ArgumentParser(description="PySR CLI")
    parser.add_argument("--dataset", type=str, choices=data.allowed_data(), metavar="D",
                        help="which dataset would you like to analyse")
    parser.add_argument("--noise", type=float, default=0., metavar="N",
                        help="how much noise to add to dataset")

    args = parser.parse_args()
    return args


def main():
    args = ParseArgs()

    data_func = data.allowed_data()[args.dataset]
    init_data, init_symb = data_func(args.noise)

    pysrmain(init_data, init_symb)


if __name__ == "__main__":
    main()
