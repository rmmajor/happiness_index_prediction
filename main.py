import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gradient_descent.fibonacci_search import fibonacci_search


class LinearRegression:
    def __init__(self):
        self.w = np.array([], dtype=np.longdouble)
        self.alpha = None
        self.prev_w = np.array([], dtype=np.longdouble)
        self.iterations_count = 0
        self.eps = 0.000001
        self.errors = []

    def learn(self, X: np.array, y: np.array) -> np.array:
        X = self.normalize(X)
        X = self.add_intercept(X)
        self.w = np.zeros(X.shape[1])

        while True:
            self.iterations_count += 1
            self.errors.append(self._loss(X, y, self.w))
            print(self.iterations_count, self.w, self.errors[-1])

            gradient: np.array = self._grad_loss(X, y, self.w)

            self.prev_w = self.w
            self.alpha: float = fibonacci_search(
                func=lambda a: self._loss(X, y, self.w - a * gradient),
                a=0,
                b=0.001,
                eps=self.eps,
                to_print=False,
            )
            self.w = self.w - self.alpha * gradient

            if np.linalg.norm(self.w - self.prev_w) < self.eps:
                break

        return self.w

    @staticmethod
    def add_intercept(X: np.array) -> np.array:
        """
        Add column of ones to the left of X in order to have intercept in the model
        so w[0] will be w[0]*x[0] where x[0] == 1
        """
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def _loss(self, w: np.array, X: np.array, y: np.array) -> float:
        ans = 0
        for x, target in zip(X, y):
            ans += (target - self.f(x, w)) ** 2
        return ans

    def _grad_loss(self, X: np.array, y: np.array, w: np.array):
        return np.sum([(self.f(x, w) - target) * x for x, target in zip(X, y)], axis=0)

    @staticmethod
    def f(x, w):
        if len(x.shape) == 1:
            return float(np.sum(w * x))
        else:
            raise ValueError("x should be 1d array")

    def predict(self, X: np.array, y: np.array) -> np.array:
        X = self.normalize(X)
        X = self.add_intercept(X)
        return np.sum(self.w * X, axis=1), (np.sum(self.w * X, axis=1) - y) ** 2

    def normalize(self, x):
        """
        Normalize x by subtracting mean and dividing by standard deviation
        """
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    def draw_errors(self):
        # make the graph log scale
        plt.plot(self.errors)
        # plt.yscale('log')
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.show()


def read_csv_columns(file_path, columns) -> pd.DataFrame:
    df = pd.read_csv(file_path, usecols=columns)
    return df


def main():
    file_path = "data/report_2018-2019.csv"
    x_df = read_csv_columns(
        file_path,
        [
            "GDP per capita",
            "Social support",
            "Healthy life expectancy",
            "Freedom to make life choices",
            "Generosity",
            "Perceptions of corruption",
        ],
    )

    y_df = read_csv_columns(file_path, ["Score"])
    train_size = int(0.75 * len(x_df))

    # print(x_df_to_learn, y_df_to_learn)
    x_to_learn = x_df[:train_size].reset_index(drop=True).to_numpy(dtype=np.longdouble)
    y_to_learn = y_df[:train_size].reset_index(drop=True).to_numpy(dtype=np.longdouble)

    model = LinearRegression()
    model.learn(x_to_learn, y_to_learn)
    model.draw_errors()
    print("weights", model.w)

    x_df_to_predict = x_df[train_size:].reset_index(drop=True)
    x_to_predict = x_df_to_predict.to_numpy(dtype=np.longdouble)

    happiness_index = read_csv_columns(file_path, ["Country or region", "Score"])[
        train_size:
    ].reset_index(drop=True)

    y_actual = y_df[train_size:].reset_index(drop=True).to_numpy(dtype=np.longdouble)
    predicted_score, loss = model.predict(x_to_predict, y_actual)

    happiness_index = happiness_index.rename(columns={"Score": "Actual score"})
    happiness_index["Predicted score"] = predicted_score
    happiness_index["Relative error"] = (
        abs(
            (happiness_index["Actual score"] - happiness_index["Predicted score"])
            / happiness_index["Actual score"]
        )
        * 100
    )

    print(happiness_index)
    print("Mean relative error", np.average(happiness_index["Relative error"]))
    happiness_index.to_csv("data/output.csv", index=False)


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
