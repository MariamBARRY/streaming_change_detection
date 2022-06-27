from typing import Dict, List, Set, Union

import numpy as np

from utils import non_zero_div, partition_from_cps


def rmse(y: np.ndarray, x: np.ndarray, expand: bool = True):
    if expand:
        y, x = np.expand_dims(y, 0), np.expand_dims(x, 0)
    return np.sqrt(np.mean((y-x)**2, axis=0))


def tp(x: np.ndarray, y: np.ndarray):
    return np.sum(np.logical_and(x == 1, y == 1))


def fp(x: np.ndarray, y: np.ndarray):
    return np.sum(np.logical_and(x == 1, y == 0))


def fn(x: np.ndarray, y: np.ndarray):
    return np.sum(np.logical_and(x == 0, y == 1))


def precision(x: np.ndarray, y: np.ndarray):
    TP = tp(x, y)
    return non_zero_div(TP, TP + fp(x, y))


def recall(x: np.ndarray, y: np.ndarray):
    TP = tp(x, y)
    return non_zero_div(TP, TP + fn(x, y))


def f1_score(p: float, r: float):
    return 2*non_zero_div(p*r, p+r)


def true_positives(target_idx: Set[int], pred_idx: Set[int], margin=5):
    pred_idx = set(list(pred_idx))
    TP = set()
    for t_i in target_idx:
        close = [(abs(t_i - x), x) for x in pred_idx if abs(t_i - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(t_i)
        pred_idx.remove(xstar)
    return TP


def f_measure(annotations: Union[Dict[str, Union[List[int], Set[int]]], Union[List[int], Set[int]]], predictions: Union[List[int], Set[int]], margin: int = 5, alpha: float = .5, include_zero: bool = True):
    # include index 0
    X = {0, *set(predictions)} if include_zero else set(predictions)
    if isinstance(annotations, dict):
        # make sure indices are unique and include 0
        Y_dict = {k: {0, *v} for k, v in annotations.items()}
        Y_vals = [v for _, v in Y_dict.items()]
        Y = {v for vals in Y_vals for v in vals}
        P = len(true_positives(Y, X, margin=margin)) / len(X)
        dict_size = len(Y_dict)
        TP_dict = {k: true_positives(
            Y_dict[k], X, margin=margin) for k in Y_dict}
        R = 1 / dict_size * \
            sum(len(TP_dict[k]) / len(Y_dict[k]) for k in Y_dict)
    else:
        # include index 0 again
        Y = {0, *set(annotations)} if include_zero else set(annotations)
        P = len(true_positives(Y, X, margin=margin)) / len(X)
        R = len(true_positives(Y, X, margin=margin)) / len(Y)
    F = P * R / (alpha * R + (1 - alpha) * P + 1e-6)
    return F, P, R


def overlap(A: Union[List[int], Set[int]], B: Union[List[int], Set[int]]):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        A = set(B)
    assert isinstance(A, set) and isinstance(B, set)
    return len(A.intersection(B)) / len(A.union(B))


def segment_cover(y_segments: List[Set[int]], x_segments: List[Set[int]]):
    x_seg_len_sum = sum(map(len, x_segments))
    assert x_seg_len_sum == sum(map(len, y_segments))
    cover = [len(y_seg)*max(overlap(y_seg, x_seg) for x_seg in x_segments)
             for y_seg in y_segments]
    return sum(cover) / x_seg_len_sum


def cover_score(annotations, predictions, n_obs):
    y_partitions = partition_from_cps(annotations, n_obs)
    x_partitions = partition_from_cps(predictions, n_obs)
    return segment_cover(x_partitions, y_partitions)


def cover_single(S, Sprime):
    """Compute the covering of a segmentation S by a segmentation Sprime.
    This follows equation (8) in Arbaleaz, 2010.
    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2, 3}, {4, 5}, {6}])
    0.8333333333333334
    >>> cover_single([{1, 2, 3, 4, 5, 6}], [{1, 2, 3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2}, {3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1}, {2}, {3}, {4, 5, 6}], [{1, 2, 3, 4, 5, 6}])
    0.3333333333333333
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    """Compute the average segmentation covering against the human annotations.
    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted Cp locations
    n_obs : number of observations in the series
    >>> covering({1: [10, 20], 2: [10], 3: [0, 5]}, [10, 20], 45)
    0.7962962962962963
    >>> covering({1: [], 2: [10], 3: [40]}, [10], 45)
    0.7954144620811286
    >>> covering({1: [], 2: [10], 3: [40]}, [], 45)
    0.8189300411522634
    """
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)


def mean_detection_time_single(annotations: Union[List[int], Set[int]], predictions: Union[List[int], Set[int]], n_obs: int):
    if not len(annotations):
        return 0
    if not len(predictions):
        return n_obs
    d = []
    X = sorted(set(predictions))
    Y = sorted(set(annotations))
    for y in Y:
        X_after = [(i, x) for i, x in enumerate(X) if x >= y]
        if len(X_after):
            i, x = X_after[0]
            d.append(abs(y-x))
            del X[i]
        # else:
        #     d.append(n_obs)
    return sum(d) / len(d) if len(d) else 0


def mean_detection_time(annotations: Dict[str, Union[List[int], Set[int]]], predictions: Union[List[int], Set[int]], n_obs: int):
    assert isinstance(annotations, dict)
    d = []
    for _, v in annotations.items():
        mdt = mean_detection_time_single(v, predictions, n_obs)
        if mdt > 0:
            d.append(mdt)
    return sum(d) / len(d) if len(d) else None
