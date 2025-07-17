# QTG-QAOA for QKP

This repository implements a quantum algorithm for solving the **Quadratic Knapsack Problem (QKP)** using the **Quantum Approximate Optimization Algorithm (QAOA)**. The approach introduces a novel **Quantum Tree Generator (QTG)** as the initial state to guide the optimization process more effectively.

---

## Overview

The Quadratic Knapsack Problem is a combinatorial optimization problem that generalizes the classical knapsack problem by including pairwise profit interactions between items. This makes it suitable for testing the performance of quantum algorithms in solving hard non-linear problems.

This project proposes:

* A new initial state construction called **Quantum Tree Generator (QTG)** to encode structured information about feasible solutions. QTG originally introduced by Sören Wilkening et al. in <em>"A Quantum Algorithm for Solving 0–1 Knapsack Problems"</em>.
* A customized **mixer Hamiltonian** that ensures the quantum evolution preserves feasibility (capacity constraints). Inspired by Paul Christiansen et al. in <em>"Quantum tree generator improves QAOA state-of-the-art for the knapsack problem"</em>.
* A flexible and modular environment for **instance generation**, circuit construction, execution, and solution evaluation.

---

## Repository Structure

```bash
QTG-QAOA_for_QKP/
│
├── QTG/                  # QTG initial state preparation
│   └── qtg_builder.py
│
├── QAOA/                 # QAOA setup, cost Hamiltonian, and custom mixer
│   ├── qtg_mixer.py
│   ├── qaoa.py
│   └── execute_simulation.py
│
├── QKP/                  # QKP instance generation, classical solutions, utilities
│   ├── instance_generator.py
│   ├── classical_solution.py
│   └── solve_QKP.py
│
├── summary.ipynb         # Jupyter notebook: theoretical introduction and results
├── main.py               # Run all
└── README.md             # Project description (this file)
```

---

## Requirements

The project uses **Python 3.10+** and the following main packages:

* `qiskit`
* `numpy`
* `matplotlib`
* `cplex` (optional, for classical optimal solutions)
* `notebook`

---

## How to Use

1. **Generate a QKP instance**:

```python
from QKP.instance_generator import standard_instance_generator

standard_instance_generator(n_items=6, ...)
```

2. **Run Solve QKPr**:

```python
from QKP.solve_QKP import solve_QKP

result = solve_QKP(f"QKP\Instances\instance_qkp_{test}.txt", reps, shots)
```

3. **Compare with classical solution**:

Use `classical_solution.py` to compute the optimal solution using CPLEX.

---

## Results

The notebook `summary.ipynb` includes:

* A theoretical overview of QKP and QAOA.
* Motivation for using the QTG initial state.
* Circuit design for cost Hamiltonian and mixer.
* Empirical results comparing QTG with and without biased hadamard gates.

---

## Author

Developed by **Andres Valencia**, Physics undergraduate student, as part of a research project on quantum algorithms for combinatorial optimization at GERAD UQAM.

---

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

