import numpy as np
from qiskit.circuit.library import RXGate, RZGate, RXXGate
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit import transpile
import numpy as np
from scipy.optimize import minimize



class ScipyOptimizer:
    def __init__(self, method='COBYLA', options=None):
        self.method = method
        self.options = options if options else {}

    def minimize(self, loss_fn, initial_params):
        result = minimize(loss_fn, initial_params, method=self.method, options=self.options)
        return result

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

    def build_layer(self, params, gate):
        """Builds the entangling layer with the given gate and parameters."""
        layer = []
        param_idx = 0
        for i in range(self.n_qubits - 1):
            layer.append((gate(params[param_idx]), i, i + 1))
            param_idx += 1
        return layer

class EntanglingLayerAnsatz:
    def __init__(self, n_qubits: int, n_layers: int, entangling_layer_builder: LineEntanglingLayerBuilder):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangling_layer_builder = entangling_layer_builder

    @property
    def number_of_params(self) -> int:
        return 2 * self.n_qubits * self.n_layers + (self.n_qubits - 1) * self.n_layers

    @property
    def number_of_qubits(self) -> int:
        return self.n_qubits

    def get_executable_circuit(self, params):
        circuit = QuantumCircuit(self.n_qubits)
        param_idx = 0
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
        return circuit

    def draw_circuit(self):
        params = [Parameter(f'theta_{i}') for i in range(self.number_of_params)]
        circuit = self.get_executable_circuit(params)
        return circuit.draw(output='mpl')





class SingleBasisQCBM:
    def __init__(self, ansatz, optimizer, distance_measure=None, choices=(-1.0, 1.0), param_initializer=None):
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.num_qubits = ansatz.number_of_qubits
        self.distance_measure = distance_measure if distance_measure else self._default_distance_measure
        self.choices = choices
        self.params = self._get_initial_parameters(param_initializer)

    def _default_distance_measure(self, target_probs, model_probs):
        epsilon = 1e-10
        return np.sum(target_probs * np.log(target_probs / (model_probs + epsilon) + epsilon))

    def _get_initial_parameters(self, initializer):
        if initializer:
            return initializer(self.ansatz.number_of_params)
        return np.random.uniform(-np.pi / 2, np.pi / 2, self.ansatz.number_of_params)

    def _get_model_object(self, parameters, sampler, backend):
        qc = self.ansatz.get_executable_circuit(parameters)
        qc_transpiled = transpile(qc, backend)
        job = sampler.run(circuits=[qc_transpiled])
        result = job.result()
        quasi_dist = result.quasi_dists[0]
        counts = quasi_dist.binary_probabilities()
        shots = sum(counts.values())
        probs = np.array([counts.get(f"{i:0{self.num_qubits}b}", 0) / shots for i in range(2**self.num_qubits)])
        return probs

    def _get_generator_fn(self, sampler, backend, random_seed=None):
        def generator(n_samples, parameters):
            qc = self.ansatz.get_executable_circuit(parameters)
            qc_transpiled = transpile(qc, backend)
            job = sampler.run(circuits=[qc_transpiled])
            result = job.result()
            quasi_dist = result.quasi_dists[0]
            counts = quasi_dist.binary_probabilities()
            samples = np.array([list(map(int, k)) for k, v in counts.items() for _ in range(int(v * n_samples))])
            return samples
        return generator

    def train_on_batch(self, X, Y, sampler, backend, n_epochs):
        target_probs = np.zeros(2**self.num_qubits)
        for x, y in zip(X, Y):
            index = int("".join(map(str, x)), 2)
            target_probs[index] = y

        for epoch in range(n_epochs):
            def loss_fn(parameters):
                model_probs = self._get_model_object(parameters, sampler, backend)
                return self.distance_measure(target_probs, model_probs)

            result = self.optimizer.minimize(loss_fn, self.params)
            self.params = result.x
        return result

    def generate(self, num_samples, sampler, backend):
        generator = self._get_generator_fn(sampler, backend)
        samples = generator(num_samples, self.params)
        unique_samples, counts = np.unique(samples, axis=0, return_counts=True)
        probabilities = counts / num_samples
        return unique_samples, probabilities



# Initialize Qiskit Runtime Service with specific credentials
service = QiskitRuntimeService(name="ibm_uoft")
backend = service.backend("ibm_quebec")  # Using IBM Quebec backend

num_qubits = 4
depth = 3
X = np.array([[1, 1, 1,1], [0, 1, 1,0]])
Y = np.array([0.2, 0.8])

entangling_layer_builder = LineEntanglingLayerBuilder(num_qubits)
ansatz = EntanglingLayerAnsatz(num_qubits, depth, entangling_layer_builder)

options = {
    'maxiter': 10,   # Maximum number of iterations
    'tol': 1e-6,      # Tolerance for termination
    'disp': True      # Display convergence messages
}

optimizer = ScipyOptimizer(method='COBYLA', options=options)

qcbm = SingleBasisQCBM(ansatz, optimizer)

n_epochs = 10




# Start a session
with Session(service=service, backend=backend) as session:
    sampler = Sampler(session=session)
    result = qcbm.train_on_batch(X, Y, sampler, backend, n_epochs)
    num_samples = 1000
    unique_samples, probabilities = qcbm.generate(num_samples, sampler, backend)
    for sample, prob in zip(unique_samples, probabilities):
        print(f"Sample: {sample}, Probability: {prob}")

print(qcbm.params)