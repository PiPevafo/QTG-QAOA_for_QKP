from __future__ import annotations

"""
Module: execute_simulation.py
----------------------------
High-level utilities for executing QAOA / QTG-style quantum circuits in a
simulator and classically optimising their variational parameters with SciPy.
The helpers are deliberately lightweight wrappers around Qiskit Aer so they can
be dropped into small research scripts without additional infrastructure.

Main capabilities
-----------------
* **Circuit execution** - `circuit_simulator` runs an arbitrary
  `QuantumCircuit` on the *Aer* simulator backend and returns *normalised*
  counts.

* **Feasible-state sampling** - `extract_feasible_states` measures the
  state- and capacity-qubits of a *Quantum Tree Generator* (QTG) circuit and
  returns the probability distribution over states that satisfy the capacity
  constraint for knapsack problem.

* **Cost-function evaluation** - `cost_func_estimator` binds a parameter vector
  to a variational ansatz (e.g. QAOA) and evaluates the expectation value of a
  given cost Hamiltonian using `AerEstimator`.

* **Classical optimisation** - `run_optimization` minimises the cost function
  with `scipy.optimize.minimize` (COBYLA by default) and returns the optimal
  parameter list.

----------------

"""

from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer

def circuit_simulator(qc: QuantumCircuit, shots: int = 1024) -> dict[str, float]:
    
    simulator = Aer.get_backend('aer_simulator')
    transpiled_circuit = transpile(qc, simulator)
    job = simulator.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    total_counts = sum(counts.values())
    
    if total_counts == 0:
        raise ValueError("No counts returned from the simulation.")
    
    return {state: count / total_counts for state, count in counts.items()}


def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    """
    Computes the cost function for the QAOA circuit using an estimator.
    Args:
        params (list): The parameters for the ansatz circuit.
        ansatz (QuantumCircuit): The QAOA ansatz circuit.
        hamiltonian (PauliSumOp): The Hamiltonian representing the cost function.
        estimator (AerEstimator): The estimator to run the circuit.
    Returns:
        float: The estimated cost function value.
    """
    circuit_bound = ansatz.assign_parameters(params, inplace=False)
    job = estimator.run([circuit_bound], [hamiltonian])
    result = job.result()
    cost = result.values[0]
    # print(f"Cost function value: {cost}")
    return cost

def callback(xk):
    callback.iteration += 1
    print(f"Iteration {callback.iteration}: parameters = {xk}, \n")
callback.iteration = 0


def run_optimization(qaoa_circuit: QuantumCircuit, init_params: list[float], cost_hamiltonian: SparsePauliOp, shots: int) -> list[float]:
    """
    Runs the QAOA circuit on a simulator and extracts feasible states.
    
    Args:
        qaoa_circuit (QuantumCircuit): The QAOA circuit to be executed.
        shots (int): The number of shots to run the circuit.
        
    Returns:
        list[float]: The optimized parameters for the QAOA circuit.
    """

    estimator = AerEstimator()
    estimator.set_options(shots=shots)

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(qaoa_circuit, cost_hamiltonian, estimator),
        method="COBYLA", 
        tol=1e-5,
        #callback=callback,
        options={'maxiter': 2000}
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
 
    return result.x
