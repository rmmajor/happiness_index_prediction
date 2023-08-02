import math


def _f(x):
    return math.exp(x) - (1 / 3 * x**3) - 2 * x


def fibonacci_search(
    func, a: float = -1.5, b: float = -1, eps: float = 0.01, to_print: bool = True
) -> float:
    vars = round((b - a) / eps)
    n = _get_iteration(vars) + 1

    x1 = a + (_fibonacci(n - 2) / _fibonacci(n)) * (b - a)
    x2 = a + (_fibonacci(n - 1) / _fibonacci(n)) * (b - a)
    k = 0
    iterations_count = 0

    while True:
        iterations_count += 1
        if to_print:
            print(iterations_count, a, b)

        if func(x1) <= func(x2):
            b = x2
            x2 = x1
            x1 = a + (_fibonacci(n - k - 3) / _fibonacci(n - k - 1)) * (b - a)

        else:
            a = x1
            x1 = x2
            x2 = a + (_fibonacci(n - k - 2) / _fibonacci(n - k - 1)) * (b - a)

        k += 1

        if abs(b - a) < eps or iterations_count >= 1000:
            break

    return (a + b) / 2


def _fibonacci(n: int) -> int:
    i, j, c = 1, 1, 1

    while c < n:
        k = i
        i = j
        j = k + i
        c = c + 1

    return i


def _get_iteration(n: int) -> int:
    i, j, c = 1, 1, 0

    while i <= n:
        k = i
        i = j
        j = k + i
        c += 1

    return c


def main():
    a, b = map(float, input("Введіть межі: ").split())
    eps: float = float(input("Введіть потрібну точність"))
    ans = fibonacci_search(_f, a, b, eps)
    print(_f(ans))


if __name__ == "__main__":
    main()
