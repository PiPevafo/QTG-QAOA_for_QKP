from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RXXGate, RYYGate

def build_hamming_weight_mixer(n_items: int, total_qubits: int, capacity: int) -> QuantumCircuit:
    """
    Build a Hamming-weight preserving mixer circuit for QAOA.
    This mixer consists of applying RXX + RYY gates to all pairs of qubits,
    preserving the Hamming weight of the quantum state.

    Args:
        n_items (int): Number of qubits representing the items (first n qubits).
        total_qubits (int): Total number of qubits in the circuit.
        capacity (int): The fixed Hamming weight (i.e., number of qubits that must be in state |1⟩).
                        Also used as the number of qubits, assuming weight-1 items in QKP.

    Returns:
        QuantumCircuit: The mixer circuit as a QuantumCircuit object.
    """
    
    beta = Parameter("beta")

    qc_mixer = QuantumCircuit(total_qubits, name="Hamming_Weight_Mixer")

    for i in range(n_items):
        for j in range(i + 1, n_items):
            qc_mixer.append(RXXGate(2 * beta), [i, j])
            qc_mixer.append(RYYGate(2 * beta), [i, j])

    return qc_mixer
