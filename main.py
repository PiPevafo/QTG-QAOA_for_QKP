from QTG import QTG, QTGGate
from qiskit import QuantumCircuit, ClassicalRegister
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

###############################################################################
# Demonstration stub (executed when run as a script)                          #
###############################################################################
if __name__ == "__main__":
    # ----------- parámetros del problema ----------------------
    n      = 3                  # número de qubits de estado
    weights = [1, 2, 3]         # w₀, w₁, w₂  (enteros ≥ 0)
    capacity = 2                # C  (entero ≥ 0)
    circuit = QuantumCircuit(n)  # circuito cuántico con n qubits de estado
    # ----------- construcción del circuito --------------------
    circuit = QTG(
            num_state_qubits=n,  # número de qubits de estado
            weights=weights,
            y_ansatz=[0, 0, 0],
            capacity=capacity)

    #circuit = qtg  # la instancia es un QuantumCircuit listo para usar
    #agregar qtg al circuito
    #circuit.append(qtg, circuit.qubits[0:n])  # agregar el QTG al circuito
    #agregar registros de medición
    reg = ClassicalRegister(n, 'reg')  # qubits de estado
    circuit.add_register(reg)
    #medir solo los qubits de estado
    circuit.measure(circuit.qubits[0:n], reg)  # medir los qubits de estado
    # ----------- visualización del circuito ------------------
    circuit.draw(output='mpl', idle_wires=False, fold=-1)
    plt.title("QTG circuit for Knapsack Problem")
    plt.show()
    # Transpilar para el simulador
   
    simulator = Aer.get_backend('aer_simulator')
    transpiled_circuit = transpile(circuit, simulator)

    # # Ejecutar el circuito
    # shots = 10_000
    # job = simulator.run(transpiled_circuit, backend=simulator, shots=shots)
    # result = job.result()


    # # Obtener conteos
    # counts = result.get_counts()
    # most_probable_state = max(counts, key=counts.get)
    # decimal_value = int(most_probable_state, 2)
    # #histograma
    # print(f"Estado más probable: {most_probable_state} (Decimal: {decimal_value})")

    # distribution = {key: val/shots for key, val in counts.items()}
    # plot_histogram(distribution)
    # plt.title("Distribución de probabilidad de los estados medidos")
    # plt.show()