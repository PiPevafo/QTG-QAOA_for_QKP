from QKP.instances_generator import instance_generator
from QKP.solve_QKP import solve_QKP
from QKP.classical_solution import solve_qkp_cplex
from QKP.Instances.run_experiments import run_multiple_experiments
import time

# Example usage:
if __name__ == "__main__":

        n_items = 10    # Number of items
        r = 50          # range of coefficients
        pct = 50        # Instance density
        test = 1        # Test number
        instance_type = "densest"  # Type of instance to generate

        run_multiple_experiments(n_items=n_items, pct=pct, n_experiments=10, instance_type=instance_type)

        # instance_generator(n=10, pct=50, instance_type="densest", test_id=test)  # Generate a QKP instance and save it to a file

        # # Solve the QKP instance using CPLEX
        # best_value_cplex, best_solution_cplex = solve_qkp_cplex(f"QKP\Instances\densest\instance_densest_{test}.txt")
        # print(f"Best Value using CPLEX: {best_value_cplex}")
        # print(f"Best Solution using CPLEX: {best_solution_cplex}")
        
        # # Solve the QKP instance using QTG-QAOA approach
        # start_time = time.time()
        # best_value, best_solution = solve_QKP(f"QKP\Instances\densest\instance_densest_{test}.txt", instance_type=instance_type, reps=5, shots=100)
        # end_time = time.time()
        # print(f"Best Value using QTG-QAOA: {best_value}")
        # print(f"Best Solution using QTG-QAOA: {best_solution[0]}")
        # print(f"QTG-QAOA execution time: {(end_time - start_time) / 60:.2f} minutes")