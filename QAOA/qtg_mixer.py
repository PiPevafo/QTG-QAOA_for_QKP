from __future__ import annotations

"""QTG-based Grover mixer acting. HAY QUE REVISAR

This is a brief implementation of the QTG-based Grover mixer as described in the paper
by Paul Christiansen et al. The Quantum tree generator improves QAOA state-of-the-art for the knapsack problem.
Based on the fact to know the feasible states of the knapsack problem |KP⟩, we can build a Grover mixer
that preserves the feasible states and applies a phase shift into the feasible space.
        
            UM(β) |ψ⟩ = |ψ⟩ - (1 - e^(-iβ)) ⟨KP|ψ⟩ |KP⟩
                   
"""
    
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit import Parameter

def build_qtg_mixer(
    qtg_circuit: QuantumCircuit
) -> QuantumCircuit:
    """
   Build a Grover mixer circuit based on a QTG circuit that prepares
    The feasible state \ ket {kp}, as described in article 2411.00518.

    ARGS:
        qtg_circuit (quantumcircuit): circuit preparing the state \ ket {kp} from | 0>.
        You must include all the necessary records (items + capacity).

    Returns:
        QuantumCircuit: circuit implemented by the mixer operator.  
    """

    # Clone QTG records
    qregs = qtg_circuit.qregs
    all_qubits = [q for qreg in qregs for q in qreg]
    qc_mixer = QuantumCircuit(*qregs, name="QTG_Mixer")

    # Paso 1: aplicar el inverso del QTG
    qc_mixer.append(qtg_circuit.inverse(), all_qubits)

    # Paso 2: convertir |0> -> |1> en todos los qubits
    qc_mixer.x(all_qubits)

    # Step 3: Apply a phase rotation only if all qubits are in |1>
    # This implements exp(-i beta |0><0|) in the current basis
    beta = Parameter("beta")
    qc_mixer.mcp(2 * beta, all_qubits[:-1], all_qubits[-1])

    # Step 4: Revert the X gates
    qc_mixer.x(all_qubits)

    # Step 5: Prepare the state |KP⟩ again
    qc_mixer.append(qtg_circuit, all_qubits)

    return qc_mixer
