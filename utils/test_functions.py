"""
Testing function for optimization

Version: 2022-06-01
Author: Quan-En, Li

Function:
- Rastrigin_f: Rastrigin function
- Sphere_f: Sphere function
- Ackley_f: Ackley function
- FF_f: Fonseca-Fleming function
- Viennet_f: Viennet function
- MultiVariate_pdf: Muti-variate (independent) Gaussian distribution pdf
- FFS_f: Combine Fonseca-Fleming function and Sphere function
- Himmelblau_f: Himmelblau's function
- ST_f: Styblinski-Tang function
- SumSquare_f: Sum square function
- Griewank_f: Griewank function
- Schaffer_f2: Schaffer function N.2
- Levy_f13: Levy function N.13
- Eggholder_f: Eggholder function
- other: sin2D, new_FF_f, obj_f7, test_f, abs_f

"""

from math import pi
import numpy as np
from scipy.stats import multivariate_normal

# Rastrigin function
def Rastrigin_f(x, a=10):
    row, col = x.shape
    part_1 = a * col
    part_2 = np.sum(x ** 2, axis=1)
    part_3 = a * np.sum(np.cos(2 * pi * x), axis=1)
    result = part_1 + part_2 - part_3
    return result

# Sphere function
def Sphere_f(x):
    return np.sum(x ** 2, axis=1)

# Ackley function
def Ackley_f(x, a=20, b=0.2, c=2):
    part1 = -a * np.exp(-b * np.sqrt(0.5 * np.sum(x**2, axis=1))).reshape(-1)
    part2 = -np.exp(0.5 * np.sum(np.cos(c * pi * x), axis=1))
    part3 = np.exp(1) + a
    return part1 + part2 + part3

# Fonseca-Fleming function
def FF_f(x):
    sample_size, dimension = x.shape
    shift_constant = 1 / np.sqrt(dimension)
    result_1 = 1 - np.exp(-np.sum((x - shift_constant) ** 2, axis=1))
    result_2 = 1 - np.exp(-np.sum((x + shift_constant) ** 2, axis=1))
    return result_1, result_2

# Viennet function
def Viennet_f(x):
    # x should be (n, 2) dimension, n is sample size
    sum_of_square = x[:, 0]**2 + x[:, 1]**2
    result_1 = 0.5 * (sum_of_square) + np.sin(sum_of_square)
    result_2 = (1/8) * (3 * x[:, 0] - 2 * x[:, 1] + 4)**2 + (1/27) * (x[:, 0] - x[:, 1] + 1)**2 + 15
    result_3 = (1/(sum_of_square+1)) - 1.1 * np.exp(-sum_of_square)
    return result_1, result_2, result_3

# Muti-variate (independent) Gaussian distribution pdf
def MultiVariate_pdf(x):
    sample_size, dimension = x.shape
    mn_rv = multivariate_normal(mean=[0] * dimension, cov=1)

    y = mn_rv.pdf(x)
    normalize_y = mn_rv.pdf([0] * dimension)

    return y / normalize_y, 1 - (y / normalize_y)

# Combine Fonseca-Fleming function and Sphere function
def FFS_f(x):
    sample_size, dimension = x.shape
    shift_constant = 1 / np.sqrt(dimension)
    result_1_1 = 1 - np.exp(-np.sum((x - shift_constant) ** 2, axis=1))
    result_2_1 = 1 - np.exp(-np.sum((x + shift_constant) ** 2, axis=1))
    
    result_1_2 = np.sum((x-1) ** 2, axis=1) / ((5 ** dimension) * dimension)
    result_2_2 = np.sum((x+1) ** 2, axis=1) / ((5 ** dimension) * dimension)
    
    result_1 = 0.5*(result_1_1 + result_1_2)
    result_2 = 0.5*(result_2_1 + result_2_2)
    
    return result_1, result_2

# Himmelblau's function
# https://en.wikipedia.org/wiki/Himmelblau%27s_function
def Himmelblau_f(x):
    return (x[:,0] ** 2 + x[:,1] - 11) ** 2 + (x[:,0] + x[:,1] ** 2 - 7) ** 2

# Styblinski-Tang function
# https://www.sfu.ca/~ssurjano/stybtang.html
def ST_f(x):
    return 0.5 * np.sum((x ** 4) - 16 * (x ** 2) + (5 * x), axis=1)

# Sum square function
# https://www.sfu.ca/~ssurjano/sumsqu.html
def SumSquare_f(x):
    return x[:,0]**2 + 2 * (x[:,1]**2)

# Rosenbrock function
# https://en.wikipedia.org/wiki/Rosenbrock_function
def Rosenbrock_f(x, a=1, b=100):
    part1 = b * (x[:,1] - (x[:,0] ** 2))**2
    part2 = np.sum((a - x)**2, axis=1)
    return part1+part2

# Griewank function
# https://en.wikipedia.org/wiki/Griewank_function
def Griewank_f(x):
    part1 = np.sum((x ** 2) / 4000, axis=1)
    part2 = np.cos(x[:,0]) * np.cos(x[:,1] / np.sqrt(2))
    return part1 - part2 + 1

# Schaffer function N.2
# https://www.sfu.ca/~ssurjano/schaffer2.html
def Schaffer_f2(x):
    numerator = (np.sin(x[:,0]**2 - x[:,1]**2))**2 - 0.5
    denominator = (1 + 0.001*(x[:,0]**2 + x[:,1]**2))**2
    return 0.5 + (numerator / denominator)

# Levy function N.13
# https://www.sfu.ca/~ssurjano/levy13.html
def Levy_f13(x):
    part1 = np.sin(3 * pi * x[:,0]) ** 2
    part2 = ((x[:,0] - 1) ** 2) * (1 + np.sin(3 * pi * x[:,1]) ** 2)
    part3 = ((x[:,1] - 1) ** 2) * (1 + np.sin(2 * pi * x[:,1]) ** 2)
    return part1 + part2 + part3

# Eggholder function
# https://www.sfu.ca/~ssurjano/egg.html
def Eggholder_f(x):
    part1 = (-1) * (x[:,1] + 47) * np.sin(np.sqrt(np.abs(x[:,1] + 0.5 * x[:,0] + 47)))
    part2 = (-1) * x[:,0] * np.sin(np.sqrt(np.abs(x[:,0] - x[:,1] - 47)))
    return part1 + part2


# Other

def sin2D(x):
    return np.sin(x).sum(axis=1)

def new_FF_f(x):
    sample_size, dimension = x.shape
    shift_constant = 1 / np.sqrt(dimension)
    
    result_1 = 1 - np.exp(-np.sum((x - shift_constant) ** 2, axis=1))
    result_2 = (-1) * result_1 + 1
    return result_1, result_2

def obj_f7(x):
    result_1 = (
        np.cos(7 * np.pi * x[:, 0] / 4) * np.exp(0.7 * x[:, 0] / 2)
        + x[:, 1]
        + 5 * np.sin(1 * np.pi * x[:, 1])
    )
    result_2 = (
        np.cos(7 * np.pi * x[:, 0] / 4) * np.exp(0.7 * x[:, 0] / 2)
        + x[:, 1]
        + 5 * np.sin(2 * np.pi * x[:, 1])
    )
    result_3 = (
        np.cos(7 * np.pi * x[:, 0] / 4) * np.exp(0.7 * x[:, 0] / 2)
        + x[:, 1]
        + 5 * np.sin(3 * np.pi * x[:, 1])
    )
    return result_1, result_2, result_3

def test_f(x):
    part1 = 1 - np.sqrt(np.abs(x[:,0] - 2))
    part2 = 2 * (x[:,1] - np.sin(6 * pi * np.abs(x[:,0] - 2) + pi))**2
    return part1 + part2

def abs_f(x, c=0, axis=0):
    return np.abs(x[:,axis] - c)