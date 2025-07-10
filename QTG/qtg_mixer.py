from __future__ import annotations

"""QTG-based Grover mixer acting. HAY QUE REVISAR

This is a brief implementation of the QTG-based Grover mixer as described in the paper
by Paul Christiansen et al. The Quantum tree generator improves QAOA state-of-the-art for the knapsack problem.
Based on the fact to know the feasible states of the knapsack problem |KP⟩, we can build a Grover mixer
that preserves the feasible states and applies a phase shift into the feasible space.
        
            UM(β) |ψ⟩ = |ψ⟩ - (1 - e^(-iβ)) ⟨KP|ψ⟩ |KP⟩
                   
The public helper :func:`build_qtg_mixer` returns a parameterised
:class:`~qiskit.circuit.QuantumCircuit` implementing

            UM(β) = G [ 1 - (1 - e^(iβ) ) |0 ... 0> <0 ... 0| ] G†,
            
where

* G : prepares the feasible-space superposition via 
  :class:`QTG : |KP>` class; it is **recursively decomposed** (`.decompose(recurse=True)`) so
  the final mixer contains only basis gates (`u`, `cx`, etc.) and safe
  synth-sized objects. 
  
Usage remains unchanged:

```python
from qtg_mixer import build_qtg_mixer
mixer_qtg = build_qtg_mixer(n_items, weights=weights, capacity=capacity)
```
"""

from typing import Sequence, Optional, Iterable

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import MCPhaseGate

# ---------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------


def build_qtg_mixer(
    num_state_qubits: int,
    *,
    weights: Sequence[int],
    capacity: int,
    y_ansatz: Optional[Sequence[int]] = None,
    name: str = "QTG_Mixer",
) -> QuantumCircuit:
    """Return a Grover-type mixer acting only on the **state** qubits.

    Parameters
    ----------
    num_state_qubits
        Number of qubits that encode item choice (``n`` in the paper).
    weights, capacity
        Instance data forwarded to :class:`QTG <QTG.QTG>`.
    y_ansatz
        Optional bitstring *y* to pre-select items (defaults to all 0s).
    name
        Name of the returned :class:`~qiskit.circuit.QuantumCircuit`.
    """

    # ------------------------------------------------------------------
    # 1) Build the state‑preparation circuit G (and decompose!)
    # ------------------------------------------------------------------
    if y_ansatz is None:
        y_ansatz = [0] * num_state_qubits

    from QTG.qtg_builder import QTG  # local import to avoid circular deps

    prep_raw = QTG(
        num_state_qubits=num_state_qubits,
        weights=weights,
        y_ansatz=y_ansatz,
        capacity=capacity,
    )

    # Decompose recursively to **eliminate** library objects that break HLS
    prep = prep_raw.decompose()

    total_qubits = prep.num_qubits  # incluye ancillas

    # Index of Logic qubits
    state_ids: Iterable[int] = range(num_state_qubits)

    # ------------------------------------------------------------------
    # 2) Allocate mixer circuit
    # ------------------------------------------------------------------
    mixer = QuantumCircuit(total_qubits, name=name)

    # ------------------------------------------------------------------
    # 3) U_M = G · (I⊗R_sub) · G†  (R_sub acts only in state qubits)
    # ------------------------------------------------------------------
    mixer.compose(prep, inplace=True)

    beta = Parameter("β")

    # Reflection about |0…0> on the *subsystem* only
    #   a) X map |0> → |1>
    mixer.x(list(state_ids))

    #   b) Multi‑controlled phase when  |1>
    if num_state_qubits == 1:
        mixer.rz(-beta, state_ids.start)  # type: ignore[attr-defined]
    else:
        mcphase = MCPhaseGate(-beta, num_state_qubits - 1)
        mixer.append(mcphase, list(state_ids))

    #   c) Undo X map |1> → |0>
    mixer.x(list(state_ids))

    #   d) G† 
    mixer.compose(prep.inverse(), inplace=True)

    return mixer
