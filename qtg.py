import numpy as np
import math
from typing import List, Optional
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import IntegerComparatorGate, BlueprintCircuit, UnitaryGate

"""
Quantum Tree Generator (QTG) -state preparation circuit (without profit register)

30-Jun-2025 (v2)
-------------------------------------------------------------------------------
This script builds, **in Qiskit**, the Quantum Tree Generator state |ψ_QTG⟩ for the
0-1 Knapsack Problem exactly as described by Wilkening *et al.* (2024), **omitting**
the profit register.  The resulting circuit prepares a superposition of **all and
only** *feasible* item selections, i.e. those that respect the single-capacity
constraint ∑ wᵢxᵢ≤C.

Main entry point
================
    build_qtg(weights: list[int], capacity: int,
              y: list[int] | None = None, b: float = 0.0) -> QuantumCircuit

Parameters
----------
weights   : classical list w=[w₀,…,w_{n-1}] with wᵢ>0                (integers)
capacity  : knapsack capacity C                                      (integer)
y         : heuristic “intermediate” solution (default all-zeros).    (bit list)
            Its bits yᵢ determine the **biased Hadamard** applied to xᵢ.
b         : Hadamard bias b ≥ 0 (b_opt ≈ n/(2πΔ) in the paper).

Returned value
--------------
QuantumCircuit with (n + m + a) qubits where
    n = len(weights)                    - path qubits  |x⟩
    m = ceil(log₂(capacity+1))          - capacity register |C⟩ (LSB-on-top)
    a = ancillas required by arithmetic/comparator sub-modules
After executing the circuit on |0⋯0⟩|C⟩|0⋯0⟩ and discarding ancillas, the system is

          Σ_{x∈B} αₓ |x⟩|C - Σ wᵢxᵢ⟩ ,     B = { x ∈ {0,1}ⁿ | Σ wᵢxᵢ ≤ C }.

Dependencies
------------
$ pip install qiskit

Copyright 2025 - released under the MIT Licence.
"""

###############################################################################
# Helper: biased Hadamard gate H_b^{(y_i)}                                   #
###############################################################################

def biased_hadamard(y_bit: int, b: float) -> UnitaryGate:
    """Return the 1-qubit biased-Hadamard gate H_b^{(y)} defined in Eq.(17).

    H_b^{(0)} = 1/√(b+2) * [[√(1+b), 1], [1, √(1+b)]]
    H_b^{(1)} = 1/√(b+2) * [[1, √(1+b)], [√(1+b), 1]]
    """
    if b < 0:
        raise ValueError("Bias b must be non-negative.")
    s = math.sqrt(1.0 + b)
    norm = math.sqrt(b + 2.0)
    if y_bit == 0:
        mat = np.array([[s, 1.0], [1.0, -s]], dtype=complex) / norm
    else:
        mat = np.array([[1.0, s], [s, -1.0]], dtype=complex) / norm
    # global phase irrelevant – UnitaryGate will normalise automatically.
    return UnitaryGate(mat, label=f"H₍b={b:.2f}₎^{y_bit}")

###############################################################################
# QTG builder                                                                 #
###############################################################################


class QTG(BlueprintCircuit):
    """Assemble the QTG circuit for a single-constraint (1-D) knapsack.

    The profit register and Grover mixer are **not** included - only the
    preparation circuit that maps |0⋯0⟩|C⟩ → |ψ_QTG⟩.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_state_qubits: Optional[int] = None,
        weights: Optional[List[int]] = None,
        capacity: Optional[int] = None,
        y_ansatz: Optional[List[int]] = None,
        biased: float = 0.0,
        name: str = "QTG",
    ) -> None:
        super().__init__(name=name)
        self._weights: Optional[List[int]] = None
        self._num_state_qubits: Optional[int] = None
        self._capacity: Optional[int] = None
        self.weights = weights
        self.num_state_qubits = num_state_qubits
        self.capacity = capacity
        self.biased = biased
        if y_ansatz is not None:
            self.y_ansatz = y_ansatz
        else:
            n = num_state_qubits if num_state_qubits is not None else (len(weights) if weights is not None else 0)
            self.y_ansatz = [0] * n

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def num_sum_qubits(self) -> int:
        return int(np.floor(np.log2(max(sum(self.weights), 1))) + 1)

    @property
    def weights(self) -> List[int]:
        if self._weights is not None:
            return self._weights
        return [1] * self.num_state_qubits if self.num_state_qubits else []

    @weights.setter
    def weights(self, weights: Optional[List[int]]):
        if weights is not None:
            for i, w in enumerate(weights):
                if not np.isclose(w, np.round(w)) or w < 0:
                    raise ValueError("Weights must be non-negative integers.")
                weights[i] = int(round(w))
        self._invalidate()
        self._weights = weights
        self._reset_registers()

    @property
    def num_state_qubits(self) -> Optional[int]:
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, n: Optional[int]):
        if n is not None and n <= 0:
            raise ValueError("num_state_qubits must be positive.")
        if n != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = n
            self._reset_registers()

    @property
    def capacity(self) -> Optional[int]:
        return self._capacity

    @capacity.setter
    def capacity(self, cap: Optional[int]):
        if cap is not None and cap < 0:
            raise ValueError("capacity must be non-negative.")
        if cap != self._capacity:
            self._invalidate()
            self._capacity = cap

    # ------------------------------------------------------------------
    # Derived counts
    # ------------------------------------------------------------------

    @property
    def num_carry_qubits(self) -> int:
        return max(self.num_sum_qubits - 1, 0)

    @property
    def num_control_qubits(self) -> int:
        return int(self.num_sum_qubits > 2)

    @property
    def num_flag_qubits(self) -> int:
        return 1

    # ------------------------------------------------------------------
    # Register allocation
    # ------------------------------------------------------------------

    def _reset_registers(self):
        self.qregs = []
        if self.num_state_qubits is None:
            return
        qr_state = QuantumRegister(self.num_state_qubits, "state")
        qr_sum = QuantumRegister(self.num_sum_qubits, "sum")
        self.qregs = [qr_state, qr_sum]
        if self.num_carry_qubits:
            self.add_register(AncillaRegister(self.num_carry_qubits, "carry"))
        if self.num_control_qubits:
            self.add_register(AncillaRegister(self.num_control_qubits, "control"))
        self.add_register(AncillaRegister(self.num_flag_qubits, "flag"))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_configuration(self, raise_on_failure=True):
        if self.num_state_qubits is None:
            if raise_on_failure:
                raise AttributeError("num_state_qubits missing")
            return False
        if self.capacity is None:
            if raise_on_failure:
                raise AttributeError("capacity missing")
            return False
        if len(self.weights) != self.num_state_qubits:
            if raise_on_failure:
                raise ValueError("len(weights) ≠ num_state_qubits")
            return False
        return True

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _build(self):
        if self._is_built:
            return
        super()._build()
        self._check_configuration()

        n_sum = self.num_sum_qubits
        qc = QuantumCircuit(*self.qregs)

        qr_state = qc.qubits[: self.num_state_qubits]
        qr_sum = qc.qubits[self.num_state_qubits : self.num_state_qubits + n_sum]
        idx_carry = self.num_state_qubits + n_sum
        qr_carry = qc.qubits[idx_carry : idx_carry + self.num_carry_qubits]
        idx_ctrl = idx_carry + self.num_carry_qubits
        qr_ctrl = qc.qubits[idx_ctrl : idx_ctrl + self.num_control_qubits]
        flag = qc.qubits[idx_ctrl + self.num_control_qubits]

        # --------------------------------------------------------------
        for i, w_i in enumerate(self.weights):
            q_state = qr_state[i]
            cmp_val = self.capacity + 1 - w_i
            cmp_gate = IntegerComparatorGate(n_sum, cmp_val, geq=False, label=f"cmp_{i}")
            qc.append(cmp_gate, qr_sum + [flag])
            bh = biased_hadamard(self.y_ansatz[i], self.biased)
            c_bh = bh.control(1)
            qc.append(c_bh, [flag, q_state])  # apply biased Hadamard on x_i controlled by cmp flag
            qc.append(cmp_gate.inverse(), qr_sum + [flag])  # uncompute

            # ---------- weighted adder (unchanged from original by Qiskit.IBM) ---------
            if w_i == 0:
                continue
            wb = f"{w_i:b}".rjust(n_sum, "0")[::-1]
            for j, bit in enumerate(wb):
                if bit == "1":
                    if n_sum == 1:
                        qc.cx(q_state, qr_sum[j])
                    elif j == 0:
                        qc.ccx(q_state, qr_sum[j], qr_carry[j])
                        qc.cx(q_state, qr_sum[j])
                    elif j == n_sum - 1:
                        qc.cx(q_state, qr_sum[j])
                        qc.ccx(q_state, qr_carry[j - 1], qr_sum[j])
                    else:
                        qc.x(qr_sum[j])
                        qc.x(qr_carry[j - 1])
                        qc.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j], qr_ctrl, mode="v-chain")
                        qc.cx(q_state, qr_carry[j])
                        qc.x(qr_sum[j])
                        qc.x(qr_carry[j - 1])
                        qc.cx(q_state, qr_sum[j])
                        qc.ccx(q_state, qr_carry[j - 1], qr_sum[j])
                else:
                    if n_sum == 1 or j == 0:
                        pass
                    elif j == n_sum - 1:
                        qc.ccx(q_state, qr_carry[j - 1], qr_sum[j])
                    else:
                        qc.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j], qr_ctrl, mode="v-chain")
                        qc.ccx(q_state, qr_carry[j - 1], qr_sum[j])
            for j in reversed(range(len(wb))):
                bit = wb[j]
                if bit == "1":
                    if n_sum == 1:
                        pass
                    elif j == 0:
                        qc.x(qr_sum[j])
                        qc.ccx(q_state, qr_sum[j], qr_carry[j])
                        qc.x(qr_sum[j])
                    elif j == n_sum - 1:
                        pass
                    else:
                        qc.x(qr_carry[j - 1])
                        qc.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j], qr_ctrl, mode="v-chain")
                        qc.cx(q_state, qr_carry[j])
                        qc.x(qr_carry[j - 1])
                else:
                    if n_sum == 1 or j == 0 or j == n_sum - 1:
                        pass
                    else:
                        qc.x(qr_sum[j])
                        qc.mcx([q_state, qr_sum[j], qr_carry[j - 1]], qr_carry[j], qr_ctrl, mode="v-chain")
                        qc.x(qr_sum[j])
        # --------------------------------------------------------------
        self.append(qc.to_gate(label=self.name), self.qubits)
