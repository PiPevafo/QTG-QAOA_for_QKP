from QTG.qtg_builder import QTG
from QKP.classical_solution import greedy_deletion_solution
from QAOA.execute_simulation import run_optimization, circuit_simulator
from QAOA.cost_operator import build_qkp_cost_hamiltonian, operator_extend
from QAOA.qtg_mixer import build_qtg_mixer
from QKP.Instances.read_instances import read_instance
from QAOA.hamming_weight_mixer import build_hamming_weight_mixer
from qiskit.circuit.library import QAOAAnsatz
from qiskit import ClassicalRegister
import numpy as np


def solve_QKP(filename, instance_type, reps=3, shots=100, tol=1e-5,
              iterations=2000, biased=1/2, convergence = False, callback_bool=False, save=True):
    """
    Solves the Quadratic Knapsack Problem (QKP) using a QTG-QAOA hybrid approach.

    Parameters:
        filename (str): Path to the QKP instance file.
        reps (int): Number of repetitions for the QAOA ansatz.
    
    Returns:
        Tuple: (best_value, best_solution)
            - best_value: Maximum profit achievable within the knapsack capacity.
            - best_solution: List indicating which items are included in the optimal solution.
    """
    n, profits, weights, capacity = read_instance(filename)
    greedy_ansatz = greedy_deletion_solution(n, weights, profits, capacity)

    
    if biased != 0:
        qtg_circuit = QTG(
            num_state_qubits=n,
            weights=weights,
            y_ansatz=greedy_ansatz,  
            biased=n*biased, 
            capacity=capacity
        )
    else:
        qtg_circuit = QTG(
            num_state_qubits=n,
            weights=weights,  
            capacity=capacity
        )

    cost_hamiltonian = build_qkp_cost_hamiltonian(n, profits)
    cost_hamiltonian_ext = operator_extend(cost_hamiltonian, qtg_circuit.num_qubits)
    
    if instance_type == "standard":
        constraint_mixer =  build_qtg_mixer(qtg_circuit)
    if instance_type == "densest":
        constraint_mixer = build_hamming_weight_mixer(n_items=n, total_qubits=qtg_circuit.num_qubits, capacity=capacity)
    
    qaoa_circuit = QAOAAnsatz(
            cost_operator=cost_hamiltonian_ext,
            mixer_operator=constraint_mixer.decompose().decompose(),
            initial_state=qtg_circuit,
            reps=reps,
            name="QAOA_QKP"
        )
    
    initial_gamma = np.pi
    initial_beta = np.pi/2
    init_params = [initial_gamma, initial_beta] * reps
    
    optimized_params, objective_func_vals = run_optimization(
        qaoa_circuit=qaoa_circuit.decompose().decompose(),
        init_params=init_params,
        cost_hamiltonian=cost_hamiltonian_ext,
        shots=shots,
        tol=tol,
        iterations=iterations,
        convergence=convergence,
        callback_bool=callback_bool
    )

    qc_optimized = qaoa_circuit.assign_parameters(optimized_params)
    reg = ClassicalRegister(n, 'reg')
    qc_optimized.add_register(reg)
    qc_optimized.measure(qc_optimized.qubits[0:n], reg)
    state = circuit_simulator(qc_optimized.decompose().decompose(), shots=shots)
    state = {key[::-1]: val for key, val in state.items()} # Reverse keys to match qubit order
    
    bestsolution = max(state.items(), key=lambda kv: kv[1])
    best_value = sum(profits[i][j] for i in range(n) for j in range(n) if bestsolution[0][i] == '1' and bestsolution[0][j] == '1')
    greedy_ansatz = ''.join(str(x) for x in greedy_ansatz)
    
    greedy_value = sum(profits[i][j] for i in range(n) for j in range(n) if greedy_ansatz[i] == '1' and greedy_ansatz[j] == '1')
    
    if save:    
        with open(filename, 'a') as f:
            f.write("\nQTG-QAOA solution:\n")
            f.write(f"Best Value:  {best_value}\n")
            f.write(f"Best Solution: {bestsolution[0]}\n")

            f.write("\nGreedy solution:\n")
            f.write(f"Best Value:  {greedy_value}\n")
            f.write(f"Best Solution: {greedy_ansatz}\n")

    return [best_value, bestsolution], [greedy_value, greedy_ansatz], objective_func_vals if convergence else None
