from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
from metrics import rmse

from .scaler import Scaler


class Window(ABC):
    def __init__(self, features: Union[str, List[Union[str, Tuple[str, float]]]], size: int,  scaler: Scaler, p: float = .6):
        assert .0 < p < 1., "'p' threshold must be between 0 and 1."
        if isinstance(features, (str, tuple)):
            features = [features]
        features = [([*f]+[1.]*max(0, 2-len(f)))[:2] if isinstance(f, (tuple, list)) else (f, 1.)
                    for f in features]
        features, weights = zip(*features)
        self._features = np.asarray(features, dtype=object)
        self._weights = np.asarray(weights)
        self._index = 0
        self._scaler = scaler
        self._p = p
        self.size = size

    def extract_features(self, x: Dict):
        return {k: x[k] if k in x else .0 for k in self._features}

    @abstractmethod
    def _reference_average(self):
        pass

    @abstractmethod
    def _update(self, x: np.array):
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(features={len(self._features)}, size={self.size}, scaler={self._scaler}, p={self._p})'


class NumericalWindow(Window):
    def __init__(self, features: Union[str, List[Union[str, Tuple[str, float]]]], size: int, scaler: Scaler = None):
        super().__init__(features, size, scaler)
        shape = (size, len(self._features))
        self._reference = np.zeros(shape)
        self._current = np.zeros(shape)

    def _reference_average(self):
        return np.average(self._reference, axis=0) * self._weights

    def _update(self, x: np.array):
        if self._index < self._current.shape[0]:
            self._current[self._index] = x
            self._index += 1
        else:
            self._index = 0
            self._reference[:] = self._current
            self._current[self._index] = x
            self._current[self._index+1:] = 0

    # -> mean_score, features, score_per_feature

    def learn_one(self, x: Dict, should_score: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
        x = self.extract_features(x)
        if self._scaler:
            x = self._scaler.scale(x)
        x = np.fromiter(x.values(), dtype=np.float64)  # to numpy array
        self._update(x)
        out = np.zeros(len(self._features))
        if not should_score:
            return 0., self._features, out
        ref = self._reference_average()
        loss = rmse(ref, x) * self._weights
        out[:] = loss
        idx = np.argsort(out)[::-1]
        return np.mean(out, axis=0), self._features[idx], out[idx]
