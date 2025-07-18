from QKP.instances_generator import instance_generator
from QKP.solve_QKP import solve_QKP
from QKP.classical_solution import solve_qkp_cplex
import time

# Example usage:
if __name__ == "__main__":

        n_items = 6     # Number of items
        r = 50          # range of coefficients
        pct = 90        # Instance density   
        test = 3        # Test number  

        instance_generator(n=20, r=100, pct=50, instance_type="standard", test_id=1)  # Generate a QKP instance and save it to a file

        # Solve the QKP instance using CPLEX
        best_value_cplex, best_solution_cplex = solve_qkp_cplex(f"QKP\Instances\instance_qkp_{test}.txt")
        print(f"Best Value using CPLEX: {best_value_cplex}")
        print(f"Best Solution using CPLEX: {best_solution_cplex}")
        
        # Solve the QKP instance using QTG-QAOA approach
        start_time = time.time()
        best_value, best_solution = solve_QKP(f"QKP\Instances\instance_qkp_{test}.txt", reps=5, shots=100)
        end_time = time.time()
        print(f"Best Value using QTG-QAOA: {best_value}")
        print(f"Best Solution using QTG-QAOA: {best_solution[0]}")
        print(f"QTG-QAOA execution time: {(end_time - start_time) / 60:.2f} minutes")