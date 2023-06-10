import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from sympy import symbols
from sympy.plotting import plot


def cubic_spline_coefficients2(x, y):
    n = len(x) - 1
    h = x[1:] - x[:-1]
    A = np.reshape(np.zeros((n ** 2) - (2 * n) + 1), (n - 1, n - 1))
    b = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)

    row1 = [2 * (h[0] + h[1]), h[1]]
    rown = [h[n - 2], 2 * (h[n - 2] + h[n - 1])]

    if n > 3:
        A[n - 2] = list(np.zeros(n - 3)) + rown
        b[n - 2] = (y[n] - y[n - 1]) / h[n - 1] - (y[n - 1] - y[n - 2]) / h[n - 2]
        b[n - 3] = (y[n - 1] - y[n - 2]) / h[n - 2] - (y[n - 2] - y[n - 3]) / h[n - 3]
        A[0] = row1 + list(np.zeros(n - 3))
        for i in range(0, n-1):
            if i < n-3:
                A[i + 1] = list(np.zeros(i)) + [h[i + 1], 2 * (h[i + 1] + h[i + 2]), h[i + 2]] + list(np.zeros(n - i - 4))
            b[i] = ((y[i + 2] - y[i + 1]) / h[i + 1]) - ((y[i + 1] - y[i]) / h[i])

    elif n == 3:
        A[0] = row1
        A[1] = rown
        for i in range(0, 2):
            b[i] = ((y[i + 2] - y[i + 1]) / h[i + 1]) - ((y[i + 1] - y[i]) / h[i])
    elif n == 2:
        m = (((y[2] - y[1]) / h[1]) - ((y[1] - y[0]) / h[0])) / (2 * (h[0] + h[1]))
        return m, h

    m = np.linalg.solve(A, b)
    return np.around(m, decimals=4), h


def cubic_spline_interpolation(t, y):
    n = len(t) - 1
    m, h = cubic_spline_coefficients2(t, y)
    print(m)
    print(h)
    d = np.zeros(n)
    c = np.zeros(n)
    splines = ['a' for i in range(0, n)]
    x = symbols('x')
    a2 = plot(x**2, show=False)
    for i in range(0, n):
        if i == 0:
            d[i] = y[i]
            c[i] = (y[i+1] - y[i])/h[i] - h[i] * (m[i])
            # splines[i] = f'(x - {y[i]})^3 * {m[i]}/{h[i]} + {c[i]}(x - {y[i]}) + {d[i]}'
            a1 = plot((x - t[i])**3 * m[i]/h[i] + c[i]*(x - t[i]) + d[i], (x, t[i], t[i+1]), show=False)
            a2.append(a1[0])
        elif i == n-1:
            d[i] = y[i] - (h[i] ** 2) * m[i-1]
            c[i] = (y[i + 1] - y[i]) / h[i] + h[i] * (m[i-1])
            # splines[i] = f'-(x - {y[i + 1]})^3 * {m[i-1]}/{h[i]} + {c[i]}(x - {y[i]}) + {d[i]}'
            a1 = plot(-(x - t[i + 1])**3 * m[i-1]/h[i] + c[i]*(x - t[i]) + d[i], (x, t[i], t[i+1]), show=False)
            a2.append(a1[0])
        else:
            d[i] = y[i] - (h[i] ** 2) * m[i-1]
            # print(i)
            c[i] = (y[i + 1] - y[i]) / h[i] + h[i] * (m[i-1] - m[i])
            # splines[i] = -(x - y[i+1])**3 * m[i-1]/h[i] + (x - y[i])**3 * m[i]/h[i] + c[i]*(x - y[i]) + d[i]
            a1 = plot(-(x - t[i+1])**3 * m[i-1]/h[i] + (x - t[i])**3 * m[i]/h[i] + c[i]*(x - t[i]) + d[i], (x, t[i], t[i+1]), show=False)
            a2.append(a1[0])
    a2.show()
    # for i in range(0, 5):
    #     print(splines[i])
    # plot(splines[0], (x, -5, 5))


X = np.array([0, 1, 2, 3, 4, 5])
Y = np.array([0, 1, 4, 9, 16, 25])
cubic_spline_interpolation(X, Y)

# plt.plot(X, Y, 'o')
# plt.plot(Xi, Yi)
# plt.show()
