import argparse

import data

from bacon.bacon1 import BACON_1
from bacon.bacon3 import BACON_3


def ParseArgs():
    parser = argparse.ArgumentParser(description="Pat Langley's BACON programs simulator")
    parser.add_argument("--verbose", action="store_true",
                        help="activates verbose mode for the program's decisions")
    parser.add_argument("--dataset", type=str, choices=data.allowed_data(), metavar="D",
                        help="which dataset would you like to analyse")
    parser.add_argument("--bacon", type=int, default=3, metavar="B",
                        help="which BACON version to run on")
    parser.add_argument("--noise", type=float, default=0., metavar="N",
                        help="how much noise to add to dataset")
    
    args = parser.parse_args()
    return args

def main():
    args = ParseArgs()

    data_func = data.allowed_data()[args.dataset]
    init_data, init_symb = data_func(args.noise)

    if args.bacon == 1:
        bacon_func = BACON_1
    elif args.bacon == 3:
        bacon_func = BACON_3
    else:
        return Exception("Invalid BACON value specified. Only allowed 1 or 3.")
    
    bacon = bacon_func(init_data, init_symb, info=args.verbose)
    bacon.bacon_iterations()


if __name__ == "__main__":
    main()