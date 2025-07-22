import os
import time
from QKP.instances_generator import instance_generator
from QKP.solve_QKP import solve_QKP
from QKP.classical_solution import solve_qkp_cplex
import os
import time

def run_multiple_experiments(n_items, pct, n_experiments, output_dir="QKP/Results", instance_type="densest"):
    os.makedirs(output_dir, exist_ok=True)

    total_gap = 0.0
    total_time = 0.0

    for test in range(1, n_experiments + 1):
        print(f"Running experiment {test}")
        # Generate instance
        instance_generator(n=n_items, pct=pct, instance_type=instance_type, test_id=test)
        instance_path = f"QKP/Instances/{instance_type}/instance_{instance_type}_{test}.txt"

        # Classical solution using CPLEX
        classical_value, classical_solution = solve_qkp_cplex(instance_path)

        # Quantum solution using QTG-QAOA
        start_time = time.time()
        quantum_value, quantum_solution = solve_QKP(instance_path, reps=5, shots=100)
        end_time = time.time()

        execution_time = (end_time - start_time) / 60
        total_time += execution_time

        # Compute gap
        if classical_value == 0:
            gap = 0.0
        else:
            gap = ((classical_value - quantum_value) / classical_value) * 100

        total_gap += gap

        # Append execution time to original instance file
        with open(instance_path, "a") as f:
            f.write(f"\nQTG-QAOA execution time: {execution_time:.2f} minutes\n")

    # Average gap and time
    average_gap = total_gap / n_experiments
    average_time = total_time / n_experiments

    # Save summary result
    summary_filename = f"summary_{instance_type}_n{n_items}_pct{pct}_exp{n_experiments}.txt"
    summary_path = os.path.join(output_dir, summary_filename)
    with open(summary_path, "w") as f:
        f.write(f"Experiment Summary\n")
        f.write(f"Instance Type: {instance_type}\n")
        f.write(f"Number of Items: {n_items}\n")
        f.write(f"Density (%): {pct}\n")
        f.write(f"Number of Experiments: {n_experiments}\n")
        f.write(f"Average Gap: {average_gap:.2f}%\n")
        f.write(f"Average QTG-QAOA Execution Time: {average_time:.2f} minutes\n")

    print(f"Finished {n_experiments} experiments. Average gap: {average_gap:.2f}%, Average time: {average_time:.2f} min")