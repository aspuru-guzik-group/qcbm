import numpy as np

epsilon = 1e-5


class ExactNLL:
    def measure_distance(self, true_probs: np.ndarray, pred_probs: np.ndarray) -> float:
        return -np.dot(
            true_probs, np.log(np.clip(pred_probs, a_min=epsilon, a_max=None)),
        )

