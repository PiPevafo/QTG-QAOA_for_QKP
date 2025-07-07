# -*- coding: utf-8 -*-
"""
Subsystem‑aware QAOAAnsatz
=========================

This module defines an ansatz for the Quantum Approximate
Optimization Algorithm (QAOA) that **acts only on a designated
sub‑system of qubits** (the *system*), while leaving any extra
ancillary qubits totally untouched.  The class itself is built
*solely* on the subsystem: it does **not** include ancilla
registers, so when it is used as an instruction the purple‑box
("SubQAOA") will appear only on the system lines.  If you also
need ancillas in the global circuit, the helper function
:func:`subsystem_qaoa_ansatz` returns a *wrapper* circuit that first
prepares a user‑supplied initial state (over system+ancillas) and
then inserts the QAOA ansatz **mapped exclusively to the subsystem
wires**.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.parametervector import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.library import EvolvedOperatorAnsatz  # type: ignore

__all__ = [
    "SubsystemQAOAAnsatz",
    "subsystem_qaoa_ansatz",
]

# ---------------------------------------------------------------------------
# Utility: check if an operator is identity (up to global phase)
# ---------------------------------------------------------------------------

def _is_identity(op) -> bool:
    """Return **True** if *op* is (equivalent to) the identity.

    Accepts a ``BaseOperator``, ``SparsePauliOp`` or ``QuantumCircuit``.
    """
    if op is None:
        return True

    if isinstance(op, QuantumCircuit):  # empty circuit ⇒ identity
        return len(op.data) == 0

    try:
        mat_op = Operator(op)
    except Exception:  # pragma: no cover
        return False

    dim = mat_op.dim[0]
    id_op = Operator(np.eye(dim))
    return mat_op.equiv(id_op)


# ---------------------------------------------------------------------------
# Main class: QAOA acting ONLY on the subsystem
# ---------------------------------------------------------------------------


class SubsystemQAOAAnsatz(EvolvedOperatorAnsatz):
    r"""QAOA ansatz on *n_sys* qubits (sub‑system).

    The class inherits from :class:`qiskit.circuit.library.EvolvedOperatorAnsatz`
    but overrides configuration checks so that it is valid **only** when the
    total qubit count equals the subsystem size.  No ancillas are ever added
    inside the object; therefore, when converted into an instruction it spans
    exactly ``n_sys`` wires and leaves any other qubits in the parent circuit
    untouched.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        cost_operator: BaseOperator | SparsePauliOp | None = None,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        mixer_operator: BaseOperator | SparsePauliOp | QuantumCircuit | None = None,
        name: str = "SubQAOA",
        flatten: bool | None = None,
    ) -> None:
        super().__init__(reps=reps, name=name, flatten=flatten)

        self._cost_operator = None  # set below
        self._reps = reps
        self._initial_state = initial_state
        self._mixer = mixer_operator
        self._bounds: List[Tuple[float | None, float | None]] | None = None

        # Setting cost_operator defines n_sys and allocates q_sys register
        self.cost_operator = cost_operator

    # ------------------------------------------------------------------
    # Qubit counts
    # ------------------------------------------------------------------
    @property
    def n_sys(self) -> int:
        """Number of qubits in the *system*."""
        return 0 if self._cost_operator is None else self._cost_operator.num_qubits

    # NOTE: *num_qubits* equals *n_sys* always (no ancillas inside)

    # ------------------------------------------------------------------
    # cost_operator and registers
    # ------------------------------------------------------------------
    @property
    def cost_operator(self):
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, op):
        self._cost_operator = op
        if op is not None:
            self.qregs = [QuantumRegister(op.num_qubits, "q_sys")]
        self._invalidate()

    # ------------------------------------------------------------------
    # Initial state (optional, must match n_sys)
    # ------------------------------------------------------------------
    @property
    def initial_state(self) -> QuantumCircuit | None:
        if self._initial_state is not None:
            return self._initial_state
        circ = QuantumCircuit(self.n_sys)
        circ.h(range(self.n_sys))
        return circ

    @initial_state.setter
    def initial_state(self, state):
        self._initial_state = state
        self._invalidate()

    # ------------------------------------------------------------------
    # Mixer
    # ------------------------------------------------------------------
    @property
    def mixer_operator(self):
        if self._mixer is not None:
            return self._mixer
        if self._cost_operator is None:
            return None
        n = self.n_sys
        terms = [("I" * i + "X" + "I" * (n - i - 1), 1) for i in range(n)]
        return SparsePauliOp.from_list(terms)

    @mixer_operator.setter
    def mixer_operator(self, op):
        self._mixer = op
        self._invalidate()

    # ------------------------------------------------------------------
    # Operators for evolution
    # ------------------------------------------------------------------
    @property
    def operators(self):
        return [self.cost_operator, self.mixer_operator]

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------
    def _check_configuration(self, raise_on_failure=True) -> bool:
        if self._cost_operator is None:
            if raise_on_failure:
                raise ValueError("cost_operator must be provided")
            return False
        if (
            self._initial_state is not None
            and self._initial_state.num_qubits != self.n_sys
        ):
            if raise_on_failure:
                raise ValueError(
                    "initial_state must act on exactly n_sys qubits (no ancillas in ansatz)."
                )
            return False
        if self.mixer_operator is not None and self.mixer_operator.num_qubits != self.n_sys:
            if raise_on_failure:
                raise ValueError("mixer_operator dimension mismatch")
            return False
        return True

    # ------------------------------------------------------------------
    # Circuit construction (identical to parent, but we re‑order parameters)
    # ------------------------------------------------------------------
    def _build(self):
        if self._is_built:
            return

        super()._build()  # builds |ψ₀〉 and evolution layers on n_sys wires

        # Re‑order parameters γ, β as in canonical QAOA
        num_cost = 0 if _is_identity(self.cost_operator) else 1
        num_mixer = (
            self.mixer_operator.num_parameters
            if isinstance(self.mixer_operator, QuantumCircuit)
            else (0 if _is_identity(self.mixer_operator) else 1)
        )
        betas = ParameterVector("β", self.reps * num_mixer)
        gammas = ParameterVector("γ", self.reps * num_cost)
        permuted: List = []
        for rep in range(self.reps):
            permuted.extend(gammas[rep * num_cost : (rep + 1) * num_cost])
            permuted.extend(betas[rep * num_mixer : (rep + 1) * num_mixer])
        self.assign_parameters(dict(zip(self.ordered_parameters, permuted)), inplace=True)


# ---------------------------------------------------------------------------
# Helper function: wrap ansatz with ancillas if needed
# ---------------------------------------------------------------------------

def subsystem_qaoa_ansatz(
    cost_operator: BaseOperator,
    reps: int = 1,
    initial_state: QuantumCircuit | None = None,
    mixer_operator: BaseOperator | QuantumCircuit | None = None,
    insert_barriers: bool = False,
    name: str = "SubQAOA",
    flatten: bool = True,
):
    """Return a ready object *or* a wrapped circuit with ancillas.

    • If *initial_state* acts on exactly ``n_sys`` qubits (no ancillas), the
      function returns a plain :class:`SubsystemQAOAAnsatz`.

    • If *initial_state* involves extra qubits (system + ancillas), the
      function builds a **wrapper circuit**:

        1. Apply *initial_state* on all qubits.
        2. Insert the QAOA ansatz (which acts on the first ``n_sys`` wires only).

      Visually this ensures the purple *SubQAOA* box covers only the subsystem
      lines, leaving ancilla wires blank.
    """

    n_sys = cost_operator.num_qubits
    n_init = initial_state.num_qubits if initial_state is not None else n_sys

    # Build the subsystem‑only ansatz (no ancillas inside)
    ansatz_core = SubsystemQAOAAnsatz(
        cost_operator=cost_operator,
        reps=reps,
        initial_state=None if n_init > n_sys else initial_state,
        mixer_operator=mixer_operator,
        name=name,
        flatten=flatten,
    )

    # Case A: no ancillas -> return the ansatz directly
    if n_init == n_sys:
        if insert_barriers:
            qc = QuantumCircuit(n_sys, name=name)
            qc.compose(ansatz_core, inplace=True)
            return qc
        return ansatz_core

    # Case B: ancillas are present -> build wrapper circuit
    ancillas = n_init - n_sys
    outer = QuantumCircuit(n_init, name=name)

    # 1) prepare the global initial state
    if initial_state is not None:
        outer.compose(initial_state, inplace=True)

    # optional barrier for clarity
    if insert_barriers:
        outer.barrier()

    # 2) insert the subsystem QAOA on the first n_sys wires
    # 2) insert the subsystem QAOA on the first n_sys wires **as an Instruction**
    outer.append(ansatz_core.to_instruction(), qargs=list(range(n_sys)))

    return outer
