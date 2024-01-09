import numpy as np
import pandas as pd

from pysr import PySRRegressor

import data as d

def main():
    data, variables = d.ideal_gas()

    y = data[-1]

    model = PySRRegressor(
        niterations=100,
        maxsize=25,
        binary_operators=["+", "*", "/", "-"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        loss="loss(prediction, target) = (prediction - target)^2",
    )

    X = pd.DataFrame({str(v): d for v, d in zip(variables[:-1], data[:-1])})

    model.fit(X, y)
    print(model)

if __name__ == "__main__":
    main()

