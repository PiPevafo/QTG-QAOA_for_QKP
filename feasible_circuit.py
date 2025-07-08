"""
Module: feasible_superposition.py
---------------------------------
Utilities to prepare *any* desired superposition of **feasible** bit‑strings in
Qiskit.  Two usage modes are now supported:

1. **Uniform** superposition              (legacy behaviour)
2. **Arbitrary probability distribution** supplied as a *dict*

-------------------------------------------------------------------------------
Quick start
-------------------------------------------------------------------------------
```python
from feasible_superposition import build_feasible_superposition_circuit

# 1) Uniform over a list (as before):
feasible = ["100", "011", "110"]
qc = build_feasible_superposition_circuit(feasible)

# 2) Weighted distribution — keep original probabilities:
prob_dict = {
    "0100100": 0.0018,
    "1110110": 0.0579,
}
qc_w = build_feasible_superposition_circuit(prob_dict)
```

For convenience, the *keys* of the probability dictionary may also be tuples
whose **first element** is the bit‑string, e.g. `("0100100", 6)` → `0.0018`.
Only the first element is interpreted as the computational basis state; the rest
is ignored.

-------------------------------------------------------------------------------
Public API
-------------------------------------------------------------------------------
``build_feasible_superposition_circuit(feasible)``
    *feasible* can be either:
    • `Iterable[str]`                           → uniform amplitudes
    • `Mapping[key, float]`(`dict`)             → weighted by the given probabilities

``synthesis_via_isometry(feasible, optimisation_level=3)``
    Hardware‑friendly decomposition without ``initialize``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import List, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnitarySynthesis

__all__ = [
    "build_feasible_superposition_circuit",
    "synthesis_via_isometry",
]

# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _extract_bit_probs(
    feasible: Union[Iterable[str], Mapping[Union[str, Tuple], float]]
) -> Tuple[List[str], np.ndarray]:
    """Return *(bitstrings, probabilities)* after extensive validation.

    • If *feasible* is an **Iterable** of strings → uniform probs.
    • If *feasible* is a **Mapping** → probabilities taken from values.
      – Keys may be str *or* tuple where element 0 is the bit‑string.
    """
    # -----------------------------
    # Case 1: mapping → weighted
    # -----------------------------
    if isinstance(feasible, Mapping):
        bitstrs: List[str] = []
        probs: List[float] = []
        for k, p in feasible.items():
            bitstr = k[0] if isinstance(k, tuple) else k
            bitstrs.append(bitstr)
            probs.append(float(p))
        # validate bit‑strings
        _validate_bitstrings(bitstrs)
        probs_arr = np.array(probs, dtype=float)
        if np.any(probs_arr < 0):
            raise ValueError("Probabilities must be non‑negative")
        if probs_arr.sum() == 0:
            raise ValueError("At least one probability must be > 0")
        probs_arr /= probs_arr.sum()  # normalise if user’s sum ≠ 1
        return bitstrs, probs_arr

    # -----------------------------
    # Case 2: list / other iterable → uniform
    # -----------------------------
    bitstrs = list(feasible)
    _validate_bitstrings(bitstrs)
    m = len(bitstrs)
    probs_arr = np.full(m, 1.0 / m)
    return bitstrs, probs_arr


def _validate_bitstrings(bitstrs: Iterable[str]) -> None:
    bitstrs = list(bitstrs)
    if not bitstrs:
        raise ValueError("feasible set must not be empty")
    n = len(bitstrs[0])
    for s in bitstrs:
        if len(s) != n:
            raise ValueError("All bit‑strings must have equal length")
        if set(s) - {"0", "1"}:
            raise ValueError(f"Invalid bit‑string {s!r}")

# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def build_feasible_superposition_circuit(
    feasible: Union[Iterable[str], Mapping[Union[str, Tuple], float]]
) -> QuantumCircuit:
    """Create a circuit with initial state matching the **given distribution**.

    *feasible* may be either an **Iterable** of strings → uniform amplitudes, or
    a **dict**/Mapping whose values are *probabilities*.
    """
    bitstrs, probs = _extract_bit_probs(feasible)
    n_items = len(bitstrs[0])

    # Build amplitude vector in little‑endian indexing.
    dim = 1 << n_items
    amplitudes = np.zeros(dim, dtype=complex)
    for bitstr, p in zip(bitstrs, probs):
        index = int(bitstr[::-1], 2)
        amplitudes[index] = math.sqrt(p)

    # Renormalise in case of floating error.
    norm = math.sqrt(float(np.sum(np.abs(amplitudes) ** 2)))
    amplitudes /= norm

    qc = QuantumCircuit(n_items, name="feasible_superposition")
    qc.initialize(amplitudes, qc.qubits)
    return qc


def synthesis_via_isometry(
    feasible: Union[Iterable[str], Mapping[Union[str, Tuple], float]],
    *,
    optimisation_level: int = 3,
) -> QuantumCircuit:
    """Hardware‑friendly variant of :func:`build_feasible_superposition_circuit`."""
    base_circ = build_feasible_superposition_circuit(feasible)
    sv = Statevector(base_circ)
    n_items = base_circ.num_qubits

    circ = QuantumCircuit(n_items)
    circ.isometry(sv.data.reshape((1, -1)), list(range(n_items)), None)

    pm = PassManager(UnitarySynthesis(basis_gates=["u", "cx"]))
    return pm.run(circ, optimisation_level=optimisation_level)
