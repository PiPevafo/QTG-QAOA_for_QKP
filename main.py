from QKP.instances_generator import generate_and_save_instance
from QKP.solve_QKP import solve_QKP
from QKP.classical_solution import solve_qkp_cplex
import time

# Example usage:
if __name__ == "__main__":

        n_items = 7     # Number of items
        r = 40          # range of coefficients
        pct = 200       # Instance density   
        test = 1        # Test number  

        generate_and_save_instance(n_items, r, pct, test) # Generate a QKP instance and save it to a file
        
        # Solve the QKP instance using CPLEX
        best_value_cplex, best_solution_cplex = solve_qkp_cplex(f"QKP/Instances/instance_qkp_{test}.txt")
        print(f"Best Value using CPLEX: {best_value_cplex}")
        print(f"Best Solution using CPLEX: {best_solution_cplex}")
        
        # Solve the QKP instance using QTG-QAOA approach
        start_time = time.time()
        best_value, best_solution = solve_QKP(f"QKP/Instances/instance_qkp_{test}.txt", reps=5)
        end_time = time.time()
        print(f"Best Value using QTG-QAOA: {best_value}")
        print(f"Best Solution using QTG-QAOA: {best_solution[0]}")
        print(f"QTG-QAOA execution time: {(end_time - start_time) / 60:.2f} minutes")