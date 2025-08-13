import random
import os
from typing import Dict, List

# ================================================================
# Module: instances_generator.py
# Description:
#     This module provides functionality for generating instances of the
#     Quadratic Knapsack Problem (QKP), including both standard QKP instances
#     and Densest k-Subgraph variants formulated as QKP.
#     Instances are written to file in a standardized QKP format.
# ================================================================

MSIZE = 400  # Maximum number of items

def standard_instance(n, r, pct):
    """
    Generates a standard instance of the Quadratic Knapsack Problem (QKP).

    Parameters:
        n   : number of items
        r   : range of coefficients (profits and weights in [1, r])
        pct : density percentage (0-100) for non-zero profits

    Returns:
        dict: instance with fields 'n', 'p', 'w', 'c'
    """
    p = [[0] * n for _ in range(n)]
    w = [random.randint(1, max(r // 2, 1)) for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            if random.randrange(100) < pct:
                val = random.randint(1, r)
                p[i][j] = p[j][i] = val

    wsum = sum(w)
    while wsum <= 30:
        wsum += random.randint(1, max(w))
    c = random.randint(30, wsum - 1)

    return {'n': n, 'p': p, 'w': w, 'c': c}


def densest_instance(n: int, r: int, density_pct: float):
    if n < 2:
        raise ValueError("n must be at least 2.")

    prob = max(0.0, min(100.0, density_pct)) / 100.0 

    while True:
        Q: List[List[int]] = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < prob:
                    w_ij = int(round(r * random.gauss(0.0, 1.0)))
                    if w_ij == 0:
                        w_ij = 1 if random.random() < 0.5 else -1
                    Q[i][j] = Q[j][i] = w_ij

        has_pos = any(Q[i][j] > 0 for i in range(n) for j in range(i + 1, n))
        has_neg = any(Q[i][j] < 0 for i in range(n) for j in range(i + 1, n))
        all_zero = all(Q[i][j] == 0 for i in range(n) for j in range(n))

        if not all_zero:  # ensure not all entries are zero
            if not (has_pos and has_neg):
                edges = [(abs(Q[i][j]), i, j) for i in range(n) for j in range(i + 1, n) if Q[i][j] != 0]
                if edges:
                    edges.sort()
                    _, i, j = edges[0]
                    Q[i][j] = Q[j][i] = -Q[i][j]
            break  # exit loop only when valid instance is generated

    w: List[int] = [1] * n
    c = 2 if n <= 3 else random.randint(2, n - 2)
    return {"n": n, "p": Q, "w": w, "c": c}


def print_instance(instance, file, reference):
    """
    Saves the instance to the given file in standard QKP format.
    """
    n = instance['n']
    Q = instance['p']
    weights = instance['w']
    c = instance['c']

    file.write(reference + "\n")
    file.write(f"{n}\n")
    file.write(" ".join(str(Q[i][i]) for i in range(n)) + "\n")
    for i in range(n - 1):
        file.write(" ".join(str(Q[i][j]) for j in range(i + 1, n)) + "\n")
    file.write("\n1\n")
    file.write(f"{c}\n")
    file.write(" ".join(str(w) for w in weights) + "\n")

def instance_generator(n, r=0, pct=0, instance_type="standard", test_id=1):
    """
    Generates a QKP instance (standard or densest) and writes to file.

    Parameters:
        n             : number of items
        r             : range of values for standard instance
        pct           : density percentage (0-100)
        instance_type : 'standard' or 'densest'
        test_id       : integer identifier for output file name
    """
    if n > MSIZE:
        raise ValueError(f"Error: n exceeds maximum allowed size ({MSIZE})")

    random.seed(test_id + n + r + pct)
    if instance_type == "standard":
        instance = standard_instance(n, r, pct)
        fname = f"instance_standard_{test_id}.txt"
    elif instance_type == "densest":
        instance = densest_instance(n, r, pct)
        fname = f"instance_densest_{test_id}.txt"
    else:
        raise ValueError("instance_type must be either 'standard' or 'densest'")

    folder_path = os.path.join("QKP", "Instances", instance_type)
    subfolder_path = os.path.join(folder_path, f"n{n}_pct{pct}")
    os.makedirs(subfolder_path, exist_ok=True)
    folder_path = subfolder_path
    filepath = os.path.join(folder_path, fname)

    with open(filepath, "w") as out_file:
        reference = f"TestInstance_{instance_type}_{test_id}"
        print_instance(instance, out_file, reference)
