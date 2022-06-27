from collections import deque
from numbers import Number
from typing import Any, Dict, List, Tuple, Type, Union

import networkx as nx
import numpy as np
from typing_extensions import Literal

from feature.scaler import Scaler
from feature.window import NumericalWindow


class ChangeGraph:
    def __init__(self,
                 num_features: Union[str, List[Union[str, Tuple[str, float]]]],
                 cat_features: Union[str, List[Union[str, Tuple[str, float]]]],
                 window_size: int,
                 num_scaler: Type[Scaler] = None,
                 cat_scaler: Type[Scaler] = None,
                 nodes: Union[str, List[Union[str, Tuple[str, float]]]] = [],
                 edges: Union[Tuple[str, str], List[Tuple[str, str]]] = [],
                 custom_start: int = None,
                 threshold: float = .9):
        assert len(num_features) or len(
            cat_features), 'Must provide feature names.'
        # node preprocessing, apply default weights (1)
        if not isinstance(nodes, list):
            nodes = [nodes]
        nodes = [n[:2] if isinstance(n, (tuple, list))
                 else (n, 1.) for n in nodes]
        # edge verification
        assert isinstance(edges, (tuple, list)), \
            'Must provide a list of edges or an edge tuple.'
        assert all(isinstance(e, (tuple, list)) and len(e) == 2 for e in edges) \
            or (not all(isinstance(e, (tuple, list)) for e in edges) and len(edges) == 2), \
            'non-valid edge list.'
        # features
        self.G = None
        self.features: Dict[Union[str, Literal[0]],
                            Tuple[Any, NumericalWindow]] = {}
        self.node_weights = dict(nodes)
        if not len(nodes):
            self.features[0] = (None,  # categorical window
                                NumericalWindow(num_features, window_size,
                                                None if not num_scaler else num_scaler()))
        else:
            for node, _ in nodes:
                self.features[node] = (None,  # categorical window
                                       NumericalWindow(num_features, window_size,
                                                       None if not num_scaler else num_scaler()))
            # graph
            self.G = nx.Graph()
            self.G.add_node(nodes)
            if len(edges):
                self.G.add_edges_from(edges)

        self._window_size = window_size
        self._custom_start = custom_start if custom_start else window_size
        self.num_feature_count = len(num_features) if isinstance(
            num_features, list) else 1
        self.cat_feature_count = len(cat_features) if isinstance(
            cat_features, list) else 1

        # threshold detector
        self.detector = ThresholdChangeDetector(
            mean_threshold=threshold, window_size=window_size, min_samples=window_size)

    # -> (avg_score, features, score_per_feature)*2

    def learn_one(self, i: int, x: Dict, node_id: str) -> Tuple[Tuple[float, Tuple[str, ...], np.ndarray], Tuple[float, Tuple[str, ...], np.ndarray]]:
        f = self.features[node_id if node_id else 0]
        _, num_window = f
        should_score = i+1 >= self._custom_start if not self._custom_start is None \
            else i+1 > self._window_size
        triggered = False
        num_avg, num_features, num_scores = num_window.learn_one(
            x, should_score)
        # same thing for categorical features
        # ...
        avg = num_avg  # + cat_features
        if should_score:
            triggered = self.detector.step(avg)
        return triggered, avg, (0, None, None), (num_avg, num_features, num_scores)


class ThresholdChangeDetector:
    def __init__(self, mean_threshold: float, window_size: int = 10, min_samples: int = 100):
        assert .0 < mean_threshold < 1.
        self._changepoint = None
        self._min_samples = min_samples
        self._window = deque(maxlen=window_size)
        self._mean_threshold = mean_threshold
        self._min_samples = min_samples
        self._N = 0
        self._mean = 0

    def step(self, x: Number) -> bool:
        self._window.append(x)
        self._N += 1
        prev_mean = self._mean
        self._mean += (x - prev_mean) / self._N
        triggered = False
        if self._N > self._min_samples:
            window_mean = sum(self._window) / len(self._window)
            mean_ratio = window_mean / (self._mean + 1e-6)
            if any([mean_ratio > (1. + self._mean_threshold),
                    mean_ratio < (1. - self._mean_threshold)]):
                triggered = True
                self._changepoint = self._N
        return triggered
