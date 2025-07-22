import numpy as np

"""
    Reads a QKP instance from a file and returns the problem data.
"""


def read_instance(filename):
    """
    Reads a QKP instance from a file and returns:
    - n: number of variables
    - profits: a numpy array of shape (n, n) with the profit matrix Q
    - weights: a list of item weights
    - capacity: the knapsack capacity

    Parameters:
        filename: path to the instance file

    Returns:
        Tuple: (n, profits, weights, capacity)
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != '']

    n = int(lines[1])
    profits = np.zeros((n, n), dtype=int)

    # Diagonal elements (line 2)
    diag = list(map(int, lines[2].split()))
    for i in range(n):
        profits[i][i] = diag[i]

    # Upper triangle (lines 3 to 3+n-2)
    for i in range(n - 1):
        row = list(map(int, lines[3 + i].split()))
        for j, val in enumerate(row):
            profits[i][i + 1 + j] = val
            profits[i + 1 + j][i] = val  # fill symmetric entry

    # Capacity and weights

    if lines[-2].startswith("Best Value"):
        capacity = int(lines[-5])
        weights = list(map(int, lines[-4].split()))
    
    else:    
        capacity = int(lines[-2])
        weights = list(map(int, lines[-1].split()))

    return n, profits, weights, capacity