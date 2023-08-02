import numpy as np
from scipy.optimize import minimize
import sympy as sp

from gradient_descent.fibonacci_search import fibonacci_search

np.set_printoptions(suppress=True)


class GradientDescent:
    def __init__(
        self, func, func_grad, initial_point, eps: float = 0.01, to_print=False
    ):
        self.func = func

        if func_grad is None:
            self.func_grad = sp.lambdify(
                sp.symbols("x y"), sp.derive_by_array(func, sp.symbols("x y")), "numpy"
            )
        else:
            self.func_grad = func_grad

        self.point: np.array = initial_point
        self.eps: float = eps
        self.to_print: bool = to_print

        self.prev_point: np.array = None
        self.alpha: float = 0.1
        self.iterations_count: int = 0

    def calculate(self) -> np.array:
        while True:
            self.iterations_count += 1
            if self.to_print:
                print(self.iterations_count, self.point)

            gradient: np.array = self.func_grad(self.point)
            self.prev_point = self.point
            self.alpha: float = fibonacci_search(
                func=lambda a: self.func(self.point - a * gradient),
                a=0,
                b=10,
                eps=self.eps,
                to_print=False,
            )
            self.point = self.point - self.alpha * gradient

            if np.linalg.norm(self.point - self.prev_point) < self.eps:
                break

        return self.point


def f(x: np.array) -> float:
    return x[0] ** 2 - 2 * x[0] * x[1] + 6 * x[1] ** 2 + x[0] - x[1]


def f_gradient(x: np.array) -> np.array:
    return np.array([2 * x[0] - 2 * x[1] + 1, -2 * x[0] + 12 * x[1] - 1])


def f_test(x: np.array) -> float:
    return x[0] ** 2 + 5 * x[1] ** 2 + 8 * x[0] - 10 * x[1] + 3


def f_test_gradient(x: np.array) -> np.array:
    """Градієнт функції це вектор часткових похідних,
    які вказують на швидкість зростання функції по кожному напрямку"""
    return np.array([2 * x[0] + 8, 10 * x[1] - 10])


def main():
    a, b = map(float, input("Введіть початкову точку: ").split())
    eps: float = float(input("Введіть потрібну точність"))

    lab = GradientDescent(
        func=f, func_grad=f_gradient, initial_point=np.array([a, b]), eps=eps
    )

    point = lab.calculate()
    print("Точка екстремуму:", point)
    print("Екстремум:", "% .6f" % f(point))

    res = minimize(f, np.array([a, b]), method="CG", jac=f_gradient, tol=eps)

    print("Бібліотечна функція:")
    print("Точка екстремуму: ", res.x)
    print("Екстремум: ", "% .6f" % res.fun)


if __name__ == "__main__":
    main()
