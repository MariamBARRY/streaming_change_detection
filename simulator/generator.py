from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple, Union

import matplotlib.pyplot as plt


class Changepoint(NamedTuple):
    pt: int
    mean: float = 0.
    std: float = 1.
    steps: int = 1


class Generator(object):
    def __init__(self):
        object.__init__(self)
        self._position = -1

    def _update_position(self, step: int = 1):
        self._position += step

    def reset(self):
        self._position = -1


class DistributionGenerator(Generator, ABC):
    def __init__(self):
        Generator.__init__(self)

    @abstractmethod
    def get(self) -> Tuple[float, float]:
        pass

    def generate(self, iters: int, **kwargs):
        return [self.get() for _ in range(iters)]

    def plot(self, iters: int, **kwargs):
        self.reset()
        n = self.generate(iters, **kwargs)
        _min, _max = round(min(n)), round(max(n))
        plt.plot(n)
        plt.yticks((_min, round((_min+_max)/2), _max))
        plt.show()

    def is_changepoint(self, position: int = None):
        return False

    def is_driftpoint(self, position: int = None):
        return False


class ChangingDistributionGenerator(DistributionGenerator):
    def __init__(self, changepoints: Union[Tuple[float, ...], List[Tuple[float, ...]]]):
        DistributionGenerator.__init__(self)
        if not isinstance(changepoints, list):
            changepoints = [changepoints]
        self._changepoints = sorted(
            [Changepoint(*cpt[:4]) for cpt in changepoints], key=lambda cpt: cpt.pt)
        self._grad = None
        self._alpha = 0

    def plot(self, iters: int):
        self.reset()
        n = self.generate(iters)
        _min, _max = round(min(n)), round(max(n))
        plt.plot(n)
        plt.yticks((_min, round((_min+_max)/2), _max))
        for cpt in self._changepoints:
            plt.vlines(cpt.pt, _max, _min, color='black')
        plt.show()

    def is_changepoint(self, position: int = None):
        if position is None:
            position = self._position
        return any(position == cpt.pt for cpt in self._changepoints)

    def is_driftpoint(self, position: int = None):
        if position is None:
            position = self._position
        return any(position >= cpt.pt and position < cpt.pt + cpt.steps for cpt in self._changepoints[::-1])
