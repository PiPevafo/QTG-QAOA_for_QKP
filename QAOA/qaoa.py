from __future__ import annotations

"""
Module: qaoa.py
----------------

Provides core routines for building the QAOA circuit for the Quadratic Knapsack
Problem (QKP), including the cost Hamiltonian encoding the profit terms, and
constructing the QAOA ansatz with a custom mixer and initial state based on Quantum Tree Generator (QTG).

Contents
--------
* `build_qkp_cost_hamiltonian(n_items, profits)` - builds the cost Hamiltonian
  for the QKP using only the quadratic profit matrix. Returns a `SparsePauliOp`
  suitable for QAOA.

* `build_qaoa_circuit(n_items, profits, initial_circuit, mixer, reps)` -
  constructs the QAOA circuit from the provided data, including a user-supplied
  initial state and mixer, wrapped in `QAOAAnsatz` from Qiskit.

Usage assumes pre-constructed circuits for the initial state and the mixer,
and focuses solely on assembling the cost function and ansatz.
"""

from collections import defaultdict
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

def build_qkp_cost_hamiltonian(n_items, profits):
    """
    Build the Hamiltonian QUBO to the QKP just including the profits without constraints.
    Returns a sparsepauliop to use in QAOAAnsatz from qiskit.
    """

    pauli_dict = defaultdict(float)

    # Quadratic terms
    for i in range(n_items):
        for j in range(i, n_items):
            pauli_i = ["I"] * n_items
            pauli_i[i] = "Z"
            coeff = - 1/4 * profits[i][j]
            if i == j: 
                pauli_str = "".join(pauli_i)[::-1]
                pauli_dict[pauli_str] = coeff 
            else:
                pauli_j = pauli_i.copy()
                pauli_j[j] = "Z"
                pauli_str = "".join(pauli_j)[::-1]
                pauli_dict[pauli_str] += coeff * 2
        

    # Linear terms
    for i in range(n_items):
        pauli_i = ["I"] * n_items
        pauli_i[i] = "Z"
        pauli_str = "".join(pauli_i)[::-1] 
        coeff = 0
        for j in range(0, n_items):
            coeff += 1/2 * profits[i][j]
        pauli_dict[pauli_str] += coeff

  
    return SparsePauliOp.from_list([(p, coeff) for p, coeff in pauli_dict.items() if abs(coeff) > 1e-8])


def build_qaoa_circuit(n_items, profits, initial_circuit, mixer, reps):
    """
    Build the QAOA circuit for the QKP using the provided parameters and initial circuit.
    
    Args:
        n_items (int): Number of items in the knapsack problem.
        profits (list): Profit matrix for the knapsack problem (n_items x n_items).
        initial_circuit (QuantumCircuit): Initial circuit to start with.
        mixer (QuantumCircuit): Mixer circuit to be used in QAOA.
        reps (int): Number of repetitions for the QAOA ansatz.

    Returns:
        QuantumCircuit: The complete QAOA circuit.
    """

    cost_hamiltonian = build_qkp_cost_hamiltonian(n_items, profits)
    
    qaoa = QAOAAnsatz(
        cost_operator=cost_hamiltonian,
        mixer_operator=mixer,
        initial_state=initial_circuit,
        reps=reps,
        name="QAOA_QKP"
    )
    
    return qaoa