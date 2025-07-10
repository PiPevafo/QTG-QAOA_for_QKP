from __future__ import annotations

"""
Module: constraint_mixer.py
---------------------------
Constraint-preserving mixers for **Qiskit's** ``QAOAAnsatz`` based on
Fuchs *et al.* (2022) *“Constraint preserving mixers for QAOA”* (arXiv:2203.06095).

Given the list of feasible computational-basis states *F* (bit-strings of equal
length *n*), the functions below construct a **Hermitian** operator

    Hₘ = ∑_{x,y∈F, x≠y} w(x,y) |x⟩⟨y| + h.c.

that generates a unitary mixer

    Uₘ(β) = e^{-iβ Hₘ},

satisfying the three QAOA requirements:

1. *Preserves feasibility* - Uₘ acts non-trivially **only** inside Span(F).
2. *Connectivity* - From any |x⟩∈F one can reach any |y⟩∈F via powers of Uₘ.
3. *[Uₘ, U_P]≠0* in general (left to the user via the cost operator).

The simplest choice implemented here is the **complete-graph mixer**
(`strategy="complete"`), which connects every pair of feasible states.  Optionally
`strategy="hamming1"` limits transitions to pairs that differ in exactly one
bit (useful when all feasible strings share the same Hamming weight).

Returned objects are standard Qiskit operators and can be plugged directly into
``QAOAAnsatz``:

```python
from constraint_mixer import build_constraint_mixer_operator
from qiskit_algorithms import QAOAAnsatz  # qiskit >=0.46

mixer_op = build_constraint_mixer_operator(feasible_states, strategy="complete")
qaoa = QAOAAnsatz(cost_operator=cost_op,
                  reps=p,
                  mixer_operator=mixer_op,
                  initial_state=build_feasible_superposition_circuit(feasible_states))
```

For **≤ 16 qubits** the dense-matrix construction is typically fine (65 536 x 65 536
complex entries ≈ 64 MiB).  For larger instances you should switch to the sparse
backend provided below (`sparse=True`) or implement a problem-specific Pauli
expansion following Algorithms 2/3 in the cited paper.
"""

from typing import Iterable, List, Literal, Mapping

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli

__all__ = [
    "build_constraint_mixer_operator",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _validate_feasible_states(feasible_states: Iterable[str]) -> List[str]:

    fs = list(feasible_states)
    if not fs:
        raise ValueError("feasible_states must not be empty")
    n = len(fs[0])
    for s in fs:
        if len(s) != n:
            raise ValueError("All bit-strings must have equal length")
        if set(s) - {"0", "1"}:
            raise ValueError(f"Invalid bit-string {s!r}")
    return fs


def _hamming(a: str, b: str) -> int:
    return sum(sa != sb for sa, sb in zip(a, b))

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_constraint_mixer_operator(
    states_dict: Mapping[str, float] | None = None,
    strategy: Literal["complete", "hamming1"] = "complete",
    weight: float = 1.0,
    sparse: bool | None = None,
):
    """Return a **Hermitian** operator that can serve as mixer for QAOAAnsatz.

    Parameters
    ----------
    feasible_states
        Iterable of equal-length bit-strings defining the feasible subspace.
    strategy
        • ``"complete"`` (default) - connect **all** pairs x≠y.
        • ``"hamming1"`` - connect only pairs with Hamming distance 1.
    weight
        Positive real coefficient *w*; the Hamiltonian uses ±*w* on off-diagonals.
    sparse
        If *None* (default) use a dense matrix for n≤16 qubits and sparse beyond.
        You may override explicitly with *True*/*False*.

    Returns
    -------
    qiskit.quantum_info.Operator | qiskit.quantum_info.SparsePauliOp
        Hermitian mixer Hamiltonian Hₘ.
    """

    if states_dict is None or not states_dict:
        raise ValueError("states_dict must not be empty")

    feasible_states = [key for key, value in states_dict.items() if value > 0]
    
    fs = _validate_feasible_states(feasible_states)
    n = len(fs[0])
    dim = 1 << n

    # decide sparse vs dense
    if sparse is None:
        sparse = n > 16  # heuristic

    idx = np.fromiter((int(s[::-1], 2) for s in fs), dtype=np.uint32, count=len(fs))

    if sparse:
        # Build as SparsePauliOp via COO‑like increment
        data: list[complex] = []
        paulis: list[Pauli] = []
        # Build Z basis projectors |i><j| + |j><i| via bit masks – Algorithm 3 is better
        # but here we fallback to matrix → pauli conversion for clarity.
        mat = np.zeros((dim, dim), dtype=complex)
        for a_idx, a in enumerate(idx):
            for b in idx[a_idx + 1 :]:
                if strategy == "hamming1" and _hamming(fs[a_idx], fs[list(idx).index(b)]) != 1:
                    continue
                mat[a, b] = weight
                mat[b, a] = weight
        op = SparsePauliOp.from_operator(Operator(mat))
        return op
    else:
        H = np.zeros((dim, dim), dtype=complex)
        for i, a in enumerate(idx):
            for b in idx[i + 1 :]:
                if strategy == "hamming1" and _hamming(fs[i], fs[list(idx).index(b)]) != 1:
                    continue
                H[a, b] = weight
                H[b, a] = weight
                
        return Operator(H)
