from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit import transpile
import numpy as np
import json
from tqdm import tqdm
import torch

class SingleBasisQCBM:
    def __init__(self, ansatz, optimizer, distance_measure=None, choices=(-1.0, 1.0), param_initializer=None):
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.num_qubits = ansatz.number_of_qubits
        self.distance_measure = distance_measure if distance_measure else self._default_distance_measure
        self.choices = choices
        self.params = self._get_initial_parameters(param_initializer)

    def _default_distance_measure(self, target_probs, model_probs):
        epsilon = 1e-2
        return np.sum(target_probs * np.log(target_probs / (model_probs + epsilon) + epsilon))

    def _get_initial_parameters(self, initializer):
        if np.any(initializer):
            return initializer
        return np.random.uniform(-np.pi / 2, np.pi / 2, self.ansatz.number_of_params)

    def _get_model_object(self, parameters, sampler, backend):
        qc = self.ansatz.get_executable_circuit(parameters)
        qc_transpiled = transpile(qc, backend)
        job = sampler.run([qc_transpiled])
        result = job.result()
        quasi_dist = result[0].data
        counts = quasi_dist.meas.get_counts()
        shots = sum(counts.values())
        probs = np.array([counts.get(f"{i:0{self.num_qubits}b}", 0) / shots for i in range(2**self.num_qubits)])
        return probs

    def _get_generator_fn(self, sampler, backend, random_seed=None):
        def generator(n_samples, parameters):
            qc = self.ansatz.get_executable_circuit(parameters)
            qc_transpiled = transpile(qc, backend)
            job = sampler.run([qc_transpiled])
            result = job.result()
            quasi_dist = result[0].data
            counts = quasi_dist.meas.get_counts()
            
            # Convert probabilities to a list of samples
            samples_list = [list(map(int, k)) for k, v in counts.items() for _ in range(int(v * n_samples))]
            
            # Calculate the number of missing samples
            num_missing_samples = n_samples - len(samples_list)
            
            # If we have fewer samples than needed, we need to resample
            if num_missing_samples > 0:
                # Get additional samples based on the probabilities
                additional_samples = np.random.choice(
                    list(counts.keys()),
                    size=num_missing_samples,
                    p=list(counts.values())
                )
                additional_samples = [list(map(int, sample)) for sample in additional_samples]
                samples_list.extend(additional_samples)
            
            # Convert list of samples to a numpy array
            samples = np.array(samples_list[:n_samples])
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
        return torch.Tensor(samples),unique_samples,probabilities

    def save_params(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.params.tolist(), f)

    def load_params(self, filename):
        with open(filename, 'r') as f:
            self.params = np.array(json.load(f))

