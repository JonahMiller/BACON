import pandas as pd
import argparse
import time
from sklearn.metrics import mean_squared_error as mse

import sys
sys.path.append("..")
import data.datasets as data  # noqa

from pysr import PySRRegressor  # noqa


def run_pysr(data, variables):

    y = data[-1]

    model = PySRRegressor(
        niterations=30,
        maxsize=15,
        binary_operators=["+", "*", "/", "-", "^"],
        # unary_operators=["sqrt(x::T) where {T} = (x >= 0) ? x : T(-1e9)"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # extra_sympy_mappings={"inv": lambda x: 1 / x,
        #                       "sqrt": lambda x: x**(1/2)},
        # nested_constraints={"sqrt": {"sqrt": 0}},
        loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="accuracy",
        temp_equation_file=True,
        delete_tempfiles=True
    )

    X = pd.DataFrame({str(v): d for v, d in zip(variables[:-1], data[:-1])})

    model.fit(X, y)
    print(f"{variables[-1]} = {model.sympy()}")
    print(f"MSE difference {mse(model.predict(X), y)}")


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

    run_pysr(init_data, init_symb)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Program took {time.time() - start}s!")
