import random

# ================================================================
# Module: instances_generator.py (Based on Ahmed Khemili Documentation)
# Description:
#     This module provides functionality for generating standard instances
#     of the Quadratic Knapsack Problem (QKP). It includes functions to
#     create QKP instances and write them to file in a standard format.
#     Intended to be imported and used by other scripts. 
# ================================================================

MSIZE = 400    # Maximum number of items

def maketest(n, r, pct):
    """
    Generates an instance of the Quadratic Knapsack Problem (QKP).

    Parameters:
      n   : number of items
      r   : range of coefficients (profits and weights will be in [1, r])
      pct : instance density (% of non-zero profits)

    Returns:
      A dictionary containing:
        - n : number of items
        - p : symmetric matrix (n x n) used as matrix Q
        - w : weight vector (size n)
        - c : knapsack capacity
    """
    # Initialize profit matrix (Q) and weight vector
    p = [[0] * n for _ in range(n)]
    w = [0] * n

    # Fill the symmetric profit matrix
    for i in range(n):
        for j in range(i + 1):
            if random.randrange(100) >= pct:
                val = 0
            else:
                val = random.randrange(r) + 1  # value in [1, r]
            p[i][j] = val
            p[j][i] = val

    # Random weight generation: weights in [1, r//2]
    for i in range(n):
        w[i] = random.randrange(max(r // 2, 1)) + 1

    # Compute total weight
    wsum = sum(w)
    if wsum - 50 <= 0:
        raise Exception("Total weight too small to generate capacity.")

    # Capacity c in [50, wsum-1]
    c = random.randrange(wsum - 50) + 50

    return {
        'n': n,
        'p': p,
        'w': w,
        'c': c
    }

def print_instance(instance, file, reference):
    """
    Saves the instance to the given file,
    in the format expected by a QKP file reader.

    Format:
      - Line 1: Instance reference
      - Line 2: Number of variables (n)
      - Line 3: Linear coefficients (diagonal of Q)
      - Lines 4 to n+2: n-1 lines of quadratic coefficients (upper triangular part of Q)
      - Empty line
      - Next line: Constraint type (e.g., 1)
      - Next line: Capacity C
      - Final line: Item weights
    """
    n = instance['n']
    Q = instance['p']
    weights = instance['w']
    c = instance['c']

    file.write(reference + "\n")
    file.write(f"{n}\n")
    file.write(" ".join(str(Q[i][i]) for i in range(n)) + "\n")
    for i in range(n - 1):
        file.write(" ".join(str(Q[i][j]) for j in range(i+1, n)) + "\n")
    file.write("\n")
    file.write("1\n")
    file.write(f"{c}\n")
    file.write(" ".join(str(w) for w in weights) + "\n")

def generate_and_save_instance(n, r, pct, test_id=1):
    """
    Generates a QKP instance and saves it to a file named "instance_qkp_<test_id>.txt".

    Parameters:
        n      : number of items
        r      : range of profit and weight values
        pct    : percentage of non-zero profit entries
        test_id: identifier for the test instance file name
    """
    if n > MSIZE:
        raise ValueError(f"Error: n exceeds maximum allowed size ({MSIZE})")

    random.seed(test_id + n + r + pct)  # seed for reproducibility
    instance = maketest(n, r, pct)

    filename = f"QKP\Instances\instance_qkp_{test_id}.txt"
    with open(filename, "w") as out_file:
        reference = f"TestInstance{test_id}"
        print_instance(instance, out_file, reference)