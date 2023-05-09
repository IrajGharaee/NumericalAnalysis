import itertools

import numpy as np
import sympy as sym
from collections import OrderedDict

n = int(input('How many distinct point you have? '))
points: dict[float, float] = OrderedDict()

S = []
a = []
b = []
c = []
d = []
print('Now type x and ys in order')

for i in range(0, n):
    x = float(input(f'x_{i} is: '))
    y = float(input(f'y_{i} is: '))
    points[x] = y
    if i != n - 1:
        a[i] = y

for i in range(0, n - 1):
    x_i = next(itertools.islice(points.items(), i - 1, i))[0]
    S[i][0][x] = a[i] + b[i]*(x - x_i) + c[i]*(x - x_i)**2 + d[i]*(x - x_i)**3
    S[i][1][x] = b[i] + 2*c[i]*(x - x_i) + 3*d[i]*(x - x_i)**2
    S[i][2][x] = 2*c[i] + 6*d[i]*(x - x_i)

#   equations:
#       for i in range(0, n - 1):
#           if i != 0 and i != n - 1:
#               x_i = next(itertools.islice(points.items(), i - 1, i))[0]
#               x_j = next(itertools.islice(points.items(), i, i + 1))[0]
#               S[i][0][i] = S[i][0][j]
#               S[i][1][i] = S[i][1][j]
#               S[i][2][i] = S[i][2][j]
#           elif i == 0:
#               S[i][0][i] = next(iter(points))
#               S[i][1][i] = S[i][1][j]
#               S[i][2][i] = S[i][2][j]
#           else:
#               S[i][0][i] = S[i][0][j]
#               S[i][1][i] = S[i][1][j]
#               S[i][2][i] = next(itertools.islice(points.items(), n - 1, n))[0]


print()

# # Create a symbolic variable 'x'
# x = sym.symbols('x')
#
# # Define a symbolic expression using 'x'
# expr = x**2 + 2*x + 1
#
# # Find the derivative of the expression with respect to 'x'
# dv = sym.diff(expr, x)
#
# # Print the derivative
# print(dv)








