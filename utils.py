from typing import Any, Dict, Iterable, List


def non_zero_div(x, y):
    try:
        return x/(y)
    except ZeroDivisionError:
        return 0


def fill_(x: Dict, keys: List[str], fill: Any = .0):
    for k in keys:
        if k not in x:
            x[k] = fill


def evenize_(*dicts: Dict, fill: Any = .0):
    keys = set(dicts[0])
    u = keys
    i = keys
    for d in dicts[1:]:
        k = set(d)
        u = u | k
        i = i & k
    keys = u - i
    for d in dicts:
        fill_(d, keys, fill)


def filter_(x: Dict, keys: Iterable, keep: bool = True):
    if not isinstance(keys, set):
        keys = set(keys)
    unwanted_keys = set(x) - keys if keep else keys
    for k in unwanted_keys:
        del x[k]


def filter(x: Dict, keys: Iterable, keep: bool = True):
    keys = keys if keep else set(x) - keys
    assert all(k in x for k in keys)
    return {k: x[k] for k in keys}


def dicts_to_items(*dicts: Dict):
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    return [(k, *(d[k] for d in dicts)) for k in keys]


def partition_from_cps(locations: List[int], n_obs: int):
    """Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.
    >>> partition_from_cps([], 5)
    [{0, 1, 2, 3, 4}]
    >>> partition_from_cps([3, 5], 8)
    [{0, 1, 2}, {3, 4}, {5, 6, 7}]
    >>> partition_from_cps([1,2,7], 8)
    [{0}, {1}, {2, 3, 4, 5, 6}, {7}]
    >>> partition_from_cps([0, 4], 6)
    [{0, 1, 2, 3}, {4, 5}]
    """
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition
