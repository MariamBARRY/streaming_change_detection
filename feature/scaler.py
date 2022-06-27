import stats
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from math import sqrt
from typing import Dict

from utils import non_zero_div


class Scaler(ABC):
    @abstractmethod
    def scale(self, x: Dict[str, float]):
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class StandardScaler(Scaler):
    def __init__(self):
        self._counts = Counter()
        self._means = defaultdict(float)
        self._variances = defaultdict(float)

    def scale(self, x: Dict[str, float]):
        for k, v in x.items():
            self._counts[k] += 1
            prev_mean = self._means[k]
            self._means[k] += (v-prev_mean)/self._counts[k]
            self._variances[k] += (((v-prev_mean) *
                                   (v-self._means[k])) / self._counts[k])
        return {k: non_zero_div(v-self._means[k], sqrt(self._variances[k])) for k, v in x.items()}


class MinMaxScaler(Scaler):
    def __init__(self):
        self.min = defaultdict(stats.Min)
        self.max = defaultdict(stats.Max)

    def scale(self, x: Dict[str, float]):
        for k, v in x.items():
            self.min[k].update(v)
            self.max[k].update(v)
        return {k: max(1e-2, non_zero_div(v - self.min[k].get(), self.max[k].get() - self.min[k].get()))
                for k, v in x.items()}
