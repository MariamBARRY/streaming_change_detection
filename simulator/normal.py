from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from .generator import ChangingDistributionGenerator, DistributionGenerator


class NormalDistribution(DistributionGenerator):
    def __init__(self, mean: float, std: float):
        DistributionGenerator.__init__(self)
        assert isinstance(mean, (int, float)) and isinstance(std, (int, float))
        self.init_mean = mean
        self.init_std = std

    def get(self):
        self._update_position()
        return np.random.normal(self.init_mean, self.init_std), 0

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.init_mean}, std={self.init_std})'


class ChangingNormalDistribution(NormalDistribution, ChangingDistributionGenerator):
    def __init__(self, mean: float, std: float, changepoints: Union[Tuple[float, ...], List[Tuple[float, ...]]]):
        NormalDistribution.__init__(self, mean, std)
        ChangingDistributionGenerator.__init__(self, changepoints)
        self._curr_mean = mean
        self._curr_std = std
        self._prev_mean = mean
        self._prev_std = std
        self._alpha = 0

    def reset(self):
        self._position = -1
        self._curr_mean = self.init_mean
        self._curr_std = self.init_std
        self._prev_mean = self.init_mean
        self._prev_std = self.init_std
        self._alpha = 0
        self._grad = None

    def get(self):
        self._update_position()
        if self._alpha == 1:
            self._prev_mean = self._curr_mean
            self._prev_std = self._curr_std
            self._grad = None
            self._alpha = 0
        for i, (pt, mean, std, steps) in enumerate(self._changepoints):
            pt = int(pt)
            if self._position >= pt and self._position < pt + steps:
                if self._grad is None:
                    self._grad = np.linspace(
                        max(1e-2, self._alpha) if steps > 1 else 1, 1, steps)
                self._alpha = self._grad[self._position - pt]
                self._curr_mean, self._curr_std = mean, std
                if (i+1 < len(self._changepoints)
                        and self._changepoints[i+1].pt < pt + steps
                        and self._position + 1 == self._changepoints[i+1].pt):
                    self._prev_mean = self._curr_mean
                    self._prev_std = self._curr_std
                    self._grad = None
        x = ((1-self._alpha)*np.random.normal(self._prev_mean, self._prev_std)
             + self._alpha*np.random.normal(self._curr_mean, self._curr_std))
        return x, self._alpha

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.init_mean}, std={self.init_std}, changepoints={self._changepoints})'
