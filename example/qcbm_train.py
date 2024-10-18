# %%
import numpy as np
from qiskit.circuit.library import RXGate, RZGate, RXXGate
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qcbm.circuit import LineEntanglingLayerBuilder,EntanglingLayerAnsatz
from qcbm.qcbm_ibm import SingleBasisQCBM
from qcbm.loss import ExactNLLTorch
from qcbm.optimizer import ScipyOptimizer
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler,  Batch
import numpy as np
from itertools import combinations
import random

def generate_data(num_qubits, cardinality, num_samples):
    # Generate all possible binary vectors of length num_qubits with the given cardinality
    vectors = list(combinations(range(num_qubits), cardinality))
    
    # Convert combinations to binary vectors
    X = []
    for vec in vectors:
        binary_vec = [0] * num_qubits
        for idx in vec:
            binary_vec[idx] = 1
        X.append(binary_vec)
    
    # If the number of samples is less than the number of possible vectors, randomly sample
    if num_samples < len(X):
        X = random.sample(X, num_samples)
    
    # Generate random probabilities for each binary vector
    Y = np.random.uniform(0, 1, len(X))
    
    # Normalize probabilities to sum to 1
    Y = Y / np.sum(Y)
    
    return np.array(X), Y

num_qubits = 10
cardinality = 5
num_samples = 500

X, Y = generate_data(num_qubits, cardinality, num_samples)

# Print some examples of the generated data
print("Example X:", X[:5])
print("Example Y:", Y[:5])

# %%

# Initialize Qiskit Runtime Service with specific credentials
service = QiskitRuntimeService(name="ibm_uoft")
backend = service.backend("ibm_quebec")  # Using IBM Quebec backend

# num_qubits = 4
depth = 3
# X = np.array([[1, 1, 1,1], [0, 1, 0,0],[0, 1, 1,0], [0, 0, 0,0], [0, 0, 1,0]])
# Y = np.array([0.2, 0.2,0.2,0.2,0.2])

entangling_layer_builder = LineEntanglingLayerBuilder(num_qubits)
ansatz = EntanglingLayerAnsatz(num_qubits, depth, entangling_layer_builder,use_rxx=False)

options = {
    'maxiter': 10,   # Maximum number of iterations
    'tol': 1e-6,      # Tolerance for termination
    'disp': True      # Display convergence messages
}
#Powell
optimizer = ScipyOptimizer(method='COBYLA', options=options)

qcbm = SingleBasisQCBM(ansatz, optimizer,distance_measure=ExactNLLTorch())


# qcbm.load_params('hw_param_train_10_qubits_3_layer_linear_0.json')
# trained_param = qcbm.params.copy()
# qcbm = SingleBasisQCBM(ansatz, optimizer,distance_measure=ExactNLLTorch(),param_initializer=trained_param)


# %%
ansatz.draw_circuit()

# %%

# Start a session
for i in range(0,20):
    n_epochs = 5
    with Batch(service=service, backend=backend) as batch:
        sampler = Sampler(mode=batch)
        # qcbm.load_params(f'hw_param_train_10_qubits_3_layer_linear_{i}.json')
        result,loss_values = qcbm.train_on_batch(X, Y, sampler, backend, n_epochs)
        num_samples = 1000
        unique_samples, probabilities = qcbm.generate(num_samples, sampler, backend)
        j = i+1
        qcbm.save_params(f"hw_param_train_10_qubits_3_layer_linear_{j}.json")
        for sample, prob in zip(unique_samples, probabilities):
            print(f"Sample: {sample}, Probability: {prob}")
        print(f"loss value: {np.min(loss_values)}")
        print(np.sum(unique_samples.sum(axis=1)==5)/len(unique_samples))
            



# %%

