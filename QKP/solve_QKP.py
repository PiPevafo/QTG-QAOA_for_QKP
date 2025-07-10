from QTG.qtg_builder import QTG
from QTG.feasible_circuit import build_feasible_circuit
from QKP.classical_solution import greedy_solution
from QAOA.execute_simulation import extract_feasible_states, run_optimization, circuit_simulator
from QAOA.qaoa import build_qaoa_circuit, build_qkp_cost_hamiltonian
from QAOA.constraint_mixer import build_constraint_mixer_operator
from QKP.Instances.read_instances import read_instance
from qiskit import ClassicalRegister
import numpy as np


def solve_QKP(filename, reps=1):
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
    greedy_ansatz = greedy_solution(n, weights, profits, capacity)

    qtg_circuit = QTG(
        num_state_qubits=n,
        weights=weights,
        y_ansatz=greedy_ansatz,  
        biased=n/4, 
        capacity=capacity
    )
    
    feasible_states = extract_feasible_states(qtg_circuit, shots=1000)
    qc_feasible = build_feasible_circuit(feasible_states, uniform=False)
    
    constraint_mixer = build_constraint_mixer_operator(
        states_dict=feasible_states,
        strategy="hamming1",
        weight=1.0,
        sparse=True
    )
    
    
    qaoa_circuit = build_qaoa_circuit(
        n_items=n,
        profits=profits,
        initial_circuit=qc_feasible,
        mixer=constraint_mixer,
        reps=reps
    )
    
    cost_hamiltonian = build_qkp_cost_hamiltonian(n, profits)
    
    initial_gamma = np.pi
    initial_beta = np.pi/2
    init_params = [initial_gamma, initial_beta] * reps
    
    optimized_params = run_optimization(
        qaoa_circuit=qaoa_circuit,
        init_params=init_params,
        cost_hamiltonian=cost_hamiltonian,
        shots=1000
    )

    qc_optimized = qaoa_circuit.assign_parameters(optimized_params)
    reg = ClassicalRegister(n, 'reg')
    qc_optimized.add_register(reg)
    qc_optimized.measure(qc_optimized.qubits[0:n], reg)
    state = circuit_simulator(qc_optimized, shots=1000)
    state = {key[::-1]: val for key, val in state.items()} # Reverse keys to match qubit order
    
    bestsolution = max(state.items(), key=lambda kv: kv[1])
    best_value = sum(profits[i][j] for i in range(n) for j in range(n) if bestsolution[0][i] == '1' and bestsolution[0][j] == '1')

    with open(filename, 'a') as f:
        f.write("\nQTG-QAOA solution:\n")
        f.write(f"Best Value:  {best_value}\n")
        f.write(f"Best Solution: {bestsolution[0]}\n")

    return best_value, bestsolution
