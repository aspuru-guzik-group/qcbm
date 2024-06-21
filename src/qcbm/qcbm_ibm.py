from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit import transpile
import numpy as np
import json
from tqdm import tqdm

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
        if np.any(initializer):
            return initializer
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
        loss_values = []
        for x, y in zip(X, Y):
            index = int("".join(map(str, x)), 2)
            target_probs[index] = y
        with tqdm(total=n_epochs, desc="Training Epochs") as pbar:
            for epoch in range(n_epochs):
                def loss_fn(parameters):
                    model_probs = self._get_model_object(parameters, sampler, backend)
                    return self.distance_measure(target_probs, model_probs)

                result = self.optimizer.minimize(loss_fn, self.params)
                loss_values.append(loss_fn(self.params))
                self.params = result.x
                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=min(loss_values))
            
        return result,loss_values

    def generate(self, num_samples, sampler, backend):
        generator = self._get_generator_fn(sampler, backend)
        samples = generator(num_samples, self.params)
        unique_samples, counts = np.unique(samples, axis=0, return_counts=True)
        probabilities = counts / num_samples
        return unique_samples, probabilities

    def save_params(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.params.tolist(), f)

    def load_params(self, filename):
        with open(filename, 'r') as f:
            self.params = np.array(json.load(f))

