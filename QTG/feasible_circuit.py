from __future__ import annotations

"""
Module: feasible_circuit.py
---------------------------------
Utilities for preparing a superposition of *feasible* bit-strings for the
Knapsack Quantum Problem (KQP) with Qiskit.

Input
-----
A single ``dict`` whose **keys** are the feasible bit-strings (strings of 0/1
with equal length) and whose **values** are the inherited probabilities, e.g.

    {
        "0100100": 0.0018,
        "1110110": 0.0579
    }

Operating modes
---------------
* **Inherited distribution** *(default)* - amplitudes ∝ √p.
* **Uniform** - set ``uniform=True`` to ignore the values and build an equal-amplitude superposition over the same keys.

Public API
----------
``build_feasible_circuit(states_dict, *, uniform=False)``
    Build the circuit preparing the requested state.
"""

import math
from collections.abc import Mapping
from typing import List

import numpy as np
from qiskit import QuantumCircuit


def _extract_bit_probs(
    states_dict: Mapping[str, float], uniform: bool
) -> tuple[List[str], np.ndarray]:
    """Return *(bitstrings, probabilities)*.

    Only checks that *states_dict* is non-empty; the caller guarantees that all
    bit-strings have equal length and contain only ``'0'``/``'1'``.
    """
    if not states_dict:
        raise ValueError("Input dictionary must not be empty.")

    bitstrs: List[str] = list(states_dict.keys())

    if uniform:
        m = len(bitstrs)
        probs = np.full(m, 1.0 / m, dtype=float)
        return bitstrs, probs

    probs = np.fromiter(states_dict.values(), dtype=float)
    probs /= probs.sum()
    return bitstrs, probs


def build_feasible_circuit(
    states_dict: Mapping[str, float], *, uniform: bool = False
) -> QuantumCircuit:
    """Return a circuit whose initial state matches the requested distribution."""
    bitstrs, probs = _extract_bit_probs(states_dict, uniform)
    n = len(bitstrs[0])

    dim = 1 << n
    amps = np.zeros(dim, dtype=complex)

    for s, p in zip(bitstrs, probs):
        index = int(s[::-1], 2)         # Qiskit uses little-endian indexing
        amps[index] = math.sqrt(p)

    amps /= math.sqrt(float(np.sum(np.abs(amps) ** 2)))   # Normalise defensively

    qc = QuantumCircuit(n, name="feasible_superposition")
    qc.initialize(amps, qc.qubits)
    return qc
