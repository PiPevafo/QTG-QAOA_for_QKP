from QKP.instances_generator import instance_generator
from QKP.solve_QKP import solve_QKP
from QKP.classical_solution import solve_qkp_cplex
from QKP.Instances.run_experiments import run_multiple_experiments
import time

if __name__ == "__main__":

        n_items = 5                    # Number of items
        r = 50                         # range of coefficients
        pcts = [25, 50, 75, 100]        # Instance density
        instance_type = "standard"     # Type of instance to generate

        for pct in pcts:
                run_multiple_experiments(n_items=n_items, r=r, pct=pct, n_experiments=1, instance_type=instance_type)
