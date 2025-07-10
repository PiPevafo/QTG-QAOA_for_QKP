import numpy as np
from docplex.mp.model import Model
from QKP.Instances.read_instances import read_instance

# ================================================================
# Module: classical_solution.py
# Description:
#     This module provides classical solvers for the Quadratic Knapsack Problem (QKP).
#     It contains:
#         - greedy_solution: a heuristic greedy method based on profit-to-weight ratio
#         - solve_qkp_cplex: an exact solver using IBM CPLEX optimizer
#     Both methods read the QKP instance from a file and write the solution to the same file.
# ================================================================

def greedy_solution(n_items, weights, profits, capacity):
    value_per_weight = [profits[i][i] / weights[i] for i in range(n_items)]
    sorted_indices = np.argsort(value_per_weight)[::-1]

    selected_items = []
    total_weight = 0
    total_value = 0

    for i in sorted_indices:
        if total_weight + weights[i] <= capacity:
            selected_items.append(i)
            total_weight += weights[i]
            total_value += profits[i][i]

    greedy_ansatz = [0] * n_items 
    for i in selected_items:
        greedy_ansatz[i] = 1

    return greedy_ansatz

def solve_qkp_cplex(filename):
    """
    Solves the QKP using IBM CPLEX optimizer.

    Parameters:
        filename (str): path to the QKP instance file.

    Returns:
        Tuple: (best_value, best_solution)
            - best_value: Optimal objective value found by CPLEX.
            - best_solution: Binary string representing selected items.
    """
    n, profits, weights, capacity = read_instance(filename)

    mdl = Model("QKP_CPLEX")
    x = mdl.binary_var_list(n, name="x")

    # Objective: maximize sum_{i,j} Q_{i,j} x_i x_j
    mdl.maximize(mdl.sum(profits[i][j] * x[i] * x[j] for i in range(n) for j in range(n)))

    # Capacity constraint
    mdl.add_constraint(mdl.sum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Solve
    solution = mdl.solve()

    if solution is None:
        raise RuntimeError("CPLEX did not find a feasible solution.")

    bitstring = ''.join(['1' if solution.get_value(x[i]) > 0.5 else '0' for i in range(n)])
    best_value = sum(profits[i][j] for i in range(n) for j in range(n) 
                     if bitstring[i] == '1' and bitstring[j] == '1')

    # with open(filename, 'a') as f:
    #     f.write("\nClassical solution (CPLEX):\n")
    #     f.write(f"Best Value:  {best_value}\n")
    #     f.write(f"Best Solution: {bitstring}\n")

    return best_value, bitstring
