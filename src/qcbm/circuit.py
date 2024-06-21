import numpy as np
from qiskit.circuit.library import RXGate, RZGate, RXXGate
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

class LineEntanglingLayerBuilder:
    """
    Build entangling layers according to the line topology.
    The 1st qubit will entangle with the 2nd, 2nd with 3rd, and so on.
    """

    def __init__(self, n_qubits: int):
        """Entangler according to the line topology.

        Args:
            n_qubits: number of qubits in the circuit.
        """
        self.n_qubits = n_qubits
        self.adjacency_matrix = np.zeros((n_qubits, n_qubits), dtype=np.int8)
        
        for i in range(n_qubits - 1):
            self.adjacency_matrix[i][i + 1] = self.adjacency_matrix[i + 1][i] = 1

    def build_layer(self, params, gate=RXXGate,use_rxx=True):
        """Builds the entangling layer with the given gate and parameters."""
        layer = []
        param_idx = 0
        for i in range(self.n_qubits - 1):
            if use_rxx:
                layer.append((gate(params[param_idx]), i, i + 1))
            else:
                layer.append((params[param_idx], i, i + 1))
            param_idx += 1
        return layer
    

class EntanglingLayerAnsatz:
    def __init__(self, n_qubits: int, n_layers: int, entangling_layer_builder: LineEntanglingLayerBuilder, use_rxx = True):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangling_layer_builder = entangling_layer_builder
        self.use_rxx = use_rxx

    @property
    def number_of_params(self) -> int:
        return 2 * self.n_qubits * self.n_layers + (self.n_qubits - 1) * self.n_layers

    @property
    def number_of_qubits(self) -> int:
        return self.n_qubits

    def get_executable_circuit(self, params):
        circuit = QuantumCircuit(self.n_qubits)
        param_idx = 0
        if self.use_rxx:
            for layer_idx in range(self.n_layers):
                # Apply RX and RZ gates
                for qubit in range(self.n_qubits):
                    circuit.append(RXGate(params[param_idx]), [qubit])
                    param_idx += 1
                    circuit.append(RZGate(params[param_idx]), [qubit])
                    param_idx += 1
                # Apply entangling layer
                entangling_layer = self.entangling_layer_builder.build_layer(params[param_idx:], RXXGate)
                for gate, qubit1, qubit2 in entangling_layer:
                    circuit.append(gate, [qubit1, qubit2])
                param_idx += self.n_qubits - 1
            circuit.measure_all()
        else:
            for layer_idx in range(self.n_layers):
                # Apply RX and RZ gates
                for qubit in range(self.n_qubits):
                    circuit.append(RXGate(params[param_idx]), [qubit])
                    param_idx += 1
                    circuit.append(RZGate(params[param_idx]), [qubit])
                    param_idx += 1
                # Apply entangling layer
                entangling_layer = self.entangling_layer_builder.build_layer(params[param_idx:],use_rxx=False)
                for theta, qubit1, qubit2 in entangling_layer:
                    circuit.cx(qubit1, qubit2)
                    circuit.rz(2 * theta, qubit2)
                    circuit.cx(qubit1, qubit2)
                param_idx += self.n_qubits - 1
            circuit.measure_all()
        return circuit

    def draw_circuit(self):
        params = [Parameter(f'theta_{i}') for i in range(self.number_of_params)]
        circuit = self.get_executable_circuit(params)
        return circuit.draw(output='mpl')

