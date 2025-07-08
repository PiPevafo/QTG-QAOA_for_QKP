from __future__ import annotations

"""QTG‑based Grover mixer acting **only on the logical (item) qubits** (safe for transpiler).

**Key change 2025‑07‑04**  ▶  *The internal state‑preparation circuit is now fully
**decomposed to basis gates** before being wrapped, which prevents the
`AttributeError: 'Gate' object has no attribute 'value'` raised by the
High‑Level‑Synthesis pass when it encounters library objects such as
``IntegerComparator`` that were previously converted to a plain
:class:`~qiskit.circuit.Gate`.*

The public helper :func:`build_qtg_mixer` returns a parameterised
:class:`~qiskit.circuit.QuantumCircuit` implementing

.. math::

    U_M(\beta)\;=\;G\,[e^{-\mathrm i\,2\beta\,|0\dots 0\rangle\langle0\dots 0|}\otimes\mathbb{I}]\,G^{\dagger},

where

* ``G`` prepares the feasible‑space superposition via your
  :class:`QTG <QTG.QTG>` class; it is **recursively decomposed** (`.decompose(recurse=True)`) so
  the final mixer contains only basis gates (`u`, `cx`, etc.) and safe
  synth‑sized objects.  No library gate survives, so the transpiler no
  longer tries to apply the faulty IntegerComparator synthesis plugin.
* The phase operator acts **exclusivamente** sobre los *num_state_qubits*
  (ítems); las ancillas de ``G`` se dejan invariantes.

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
    """Return a Grover‑type mixer acting only on the **state** qubits.

    Parameters
    ----------
    num_state_qubits
        Number of qubits that encode item choice (``n`` in the paper).
    weights, capacity
        Instance data forwarded to :class:`QTG <QTG.QTG>`.
    y_ansatz
        Optional bitstring *y* to pre‑select items (defaults to all 0s).
    name
        Name of the returned :class:`~qiskit.circuit.QuantumCircuit`.
    """

    # ------------------------------------------------------------------
    # 1) Build the state‑preparation circuit G (and decompose!)
    # ------------------------------------------------------------------
    if y_ansatz is None:
        y_ansatz = [0] * num_state_qubits

    from qtg import QTG  # local import to avoid circular deps

    prep_raw = QTG(
        num_state_qubits=num_state_qubits,
        weights=weights,
        y_ansatz=y_ansatz,
        capacity=capacity,
    )

    # Decompose recursively to **eliminate** library objects that break HLS
    prep = prep_raw.decompose()

    total_qubits = prep.num_qubits  # incluye ancillas

    # Indices de los qubits *lógicos* (se asume que van primeros)
    state_ids: Iterable[int] = range(num_state_qubits)

    # ------------------------------------------------------------------
    # 2) Allocate mixer circuito
    # ------------------------------------------------------------------
    mixer = QuantumCircuit(total_qubits, name=name)

    # ------------------------------------------------------------------
    # 3) U_M = G · (I⊗R_sub) · G†  (R_sub actúa sólo en los qubits de estado)
    # ------------------------------------------------------------------
    mixer.compose(prep, inplace=True)

    beta = Parameter("β")

    # Reflection about |0…0> on the *subsystem* only
    #   a) X para mapear |0>→|1>
    mixer.x(list(state_ids))

    #   b) Multi‑controlled phase cuando TODAS están en |1>
    if num_state_qubits == 1:
        mixer.rz(-2 * beta, state_ids.start)  # type: ignore[attr-defined]
    else:
        mcphase = MCPhaseGate(-beta, num_state_qubits - 1)
        mixer.append(mcphase, list(state_ids))

    #   c) Deshacer las X
    mixer.x(list(state_ids))

    #   d) G†  (ya descompuesto)
    mixer.compose(prep.inverse(), inplace=True)

    # Mantener parámetro simbólico en el circuito
    return mixer
