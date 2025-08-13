import os
import time
from QKP.instances_generator import instance_generator
from QKP.solve_QKP import solve_QKP
from QKP.classical_solution import solve_qkp_cplex
import numpy as np

def run_multiple_experiments(n_items, r, pct, n_experiments, instance_type, layers=5, shots=1000):

    total_gap_cplex = 0.0
    total_gap_greedy = 0.0
    total_time = 0.0

    for test in range(1, n_experiments + 1):
        print(f"Running experiment {test}")
        # Generate instance
        instance_generator(n=n_items, r=r, pct=pct, instance_type=instance_type, test_id=test)
        instance_path = f"QKP/Instances/{instance_type}/n{n_items}_pct{pct}/instance_{instance_type}_{test}.txt"

        # Classical solution using CPLEX
        classical_solution = solve_qkp_cplex(instance_path)

        # Quantum solution using QTG-QAOA
        start_time = time.time()
        quantum_solution, greedy_ansatz, objective_func_vals = solve_QKP(instance_path, instance_type, reps=layers, shots=shots,
                                        tol=1e-5, iterations=100, biased=1/2, convergence=False, callback_bool=True)
        end_time = time.time()

        execution_time = (end_time - start_time) / 60
        total_time += execution_time
 
        # Compute gap
        if classical_solution[0] == 0:
            gap_cplex = 0.0
        else:
            gap_cplex = ((classical_solution[0] - quantum_solution[0]) / np.abs(classical_solution[0])) * 100

        if greedy_ansatz[0] == 0:
            gap_greedy = 0.0
        else:
            gap_greedy = ((greedy_ansatz[0] - quantum_solution[0]) / np.abs(greedy_ansatz[0])) * 100

        total_gap_cplex += gap_cplex
        total_gap_greedy += gap_greedy

        # Append execution time to original instance file
        with open(instance_path, "a") as f:
            f.write(f"\nQTG-QAOA execution time: {execution_time:.2f} minutes\n")

    # Average gap and time
    average_gap_cplex = total_gap_cplex / n_experiments
    average_gap_greedy = total_gap_greedy / n_experiments
    average_time = total_time / n_experiments

    # Save summary result
    summary_filename = f"summary_{instance_type}_n{n_items}_pct{pct}_exp{n_experiments}.txt"
    subfolder_path = os.path.join(f"n{n_items}_pct{pct}")
    summary_path = os.path.join("QKP", "Instances", instance_type , subfolder_path, summary_filename)
    with open(summary_path, "w") as f:
        f.write(f"Experiment Summary\n")
        f.write(f"Instance Type: {instance_type}\n")
        f.write(f"Number of Items: {n_items}\n")
        f.write(f"Density (%): {pct}\n")
        f.write(f"Number of Experiments: {n_experiments}\n")
        f.write(f"Average CPLEX Gap: {average_gap_cplex:.2f}%\n")
        f.write(f"Average Greedy Gap: {average_gap_greedy:.2f}%\n")
        f.write(f"Average QTG-QAOA Execution Time: {average_time:.2f} minutes\n")

    print(f"Finished {n_experiments} experiments. Average CPLEX gap: {average_gap_cplex:.2f}%, Average Greedy gap: {average_gap_greedy:.2f}%, Average time: {average_time:.2f} min")