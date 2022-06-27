from abc import ABC, abstractmethod
from numbers import Number


class Stat(ABC):
    @abstractmethod
    def update(self, x: Number):
        pass

    @abstractmethod
    def get(self) -> float:
        pass


class Min(Stat):
    def __init__(self):
        self.min = float('inf')

    def update(self, x: Number):
        self.min = min(self.min, x)

    def get(self):
        return self.min

    def __str__(self) -> str:
        return str(self.min)

    def __repr__(self) -> str:
        return f'Min({self.min})'


class Max(Stat):
    def __init__(self):
        self.max = -float('inf')

    def update(self, x: Number):
        self.max = max(self.max, x)

    def get(self) -> float:
        return self.max

    def __str__(self) -> str:
        return str(self.max)

    def __repr__(self) -> str:
        return f'Max({self.max})'
