import pandas as pd

from pysr import PySRRegressor


def main(data, variables):

    y = data[-1]

    model = PySRRegressor(
        niterations=30,
        maxsize=15,
        binary_operators=["+", "*", "/", "-"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="accuracy",
        temp_equation_file=True,
        delete_tempfiles=True
    )

    X = pd.DataFrame({str(v): d for v, d in zip(variables[:-1], data[:-1])})

    model.fit(X, y)
    print(model.sympy())
