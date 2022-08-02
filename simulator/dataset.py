import random
from collections import deque
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .generator import DistributionGenerator
from .normal import ChangingNormalDistribution, NormalDistribution


class Dataset:
    '''
    Multivariate dataset.
    '''

    def __init__(self, cols: Union[int, List[str]], dist: List[DistributionGenerator] = [], size: int = 100):
        '''
        Constructs the necessary attributes for the Multivariate Dataset.

        Parameters
        ----------
        cols: int or array of int \\
            Number of columns or their names.

        dist: array of `DistributionGenerator`\\
            Distributions of the time series.

        size: int \\
            Number of samples to generate.
        '''
        if isinstance(cols, int):
            cols = [f'feature_{i}' for i in range(cols)]
        elif not isinstance(cols, (list, tuple)):
            cols = [cols]
        assert len(cols) == len(dist), \
            f'Columns and distribution array must have the same size [{len(cols)}/{len(dist)}].'
        self.cols = cols
        self._cols = cols + ['target_cols',
                             'mean_gradient', 'change']
        self.size = size
        items = zip(self.cols, dist)
        self._dist = {c: d for c, d in items}
        self._data = None

    def generate(self, progress: bool = False):
        for dist in self._dist.values():  # reset distributions
            dist.reset()
        records = np.zeros((self.size, len(self._cols)), dtype=object)
        size_iter = range(self.size)
        if progress:
            size_iter = tqdm(size_iter)
        for i in size_iter:
            instance = np.asarray([[j, *self._dist[col].get()]
                                  for j, col in enumerate(self.cols)])
            x = instance[:, 1]
            col_grad = instance[:, [0, 2]]
            grad = col_grad[:, -1]
            change = int(np.any(grad >= 1, axis=0))
            mean_grad = np.mean(grad)
            target_cols = [self.cols[int(j)] for j in col_grad[grad > 0][:, 0]]
            target_cols = ','.join(target_cols) if target_cols else None
            records[i, :] = np.asarray([*x, target_cols,
                                        mean_grad, change], dtype=object)
        self._data = records
        return self._data

    @property
    def cpt_idx(self):
        if self._data is None:
            raise RuntimeError('No generated data for this distribution.')
        idx = np.argwhere(self._data[:, -1] == 1)
        return idx.reshape(idx.shape[0])

    @property
    def rows(self):
        if self._data is None:
            raise RuntimeError('No generated data for this distribution.')
        return self._data[:, :(len(self.cols)-len(self._cols))]

    def to_df(self, features_only: bool = False):
        if self._data is None:
            raise RuntimeError('No generated data for this distribution.')
        data = self._data
        cols = self._cols
        if features_only:
            data = self.rows
            cols = self.cols
        return pd.DataFrame.from_records(data, columns=cols)

    def plot(self, legend: bool = False, save_dir: str = None):
        if self._data is None:
            raise RuntimeError('No generated data for this distribution.')
        col_vals = self._data[:, :(len(self.cols) - len(self._cols))].T
        for i, vals in enumerate(col_vals):
            col = self.cols[i]
            plt.plot(vals, label=col)
        if legend:
            plt.legend()
        if save_dir is not None:
            plt.savefig(save_dir)


class RandomDataset(Dataset):
    def __init__(self,
                 cols: Union[int, List[str]],
                 changing_cols: Union[bool, int, List[str]],
                 change_start: int = 0,
                 num_changepoints: Union[int, List[int]] = 1,
                 drift_steps: Union[int, List[Union[int, List[int]]]] = 1,
                 init_mean: Union[float, List[float]] = 0.,
                 init_std: Union[float, List[float]] = 1.,
                 mean_ball: Union[float, List[Union[float, List[float]]]] = 10,
                 std_ratio: Union[float, List[Union[float, List[float]]]] = 1,
                 size: int = 100):
        # build column array
        if isinstance(cols, (float, int)):
            cols = [f'col_{i}' for i in range(int(cols))]
        elif not isinstance(cols, (list, tuple)):
            cols = [cols]
        # build changing columns array
        if isinstance(changing_cols, (float, int)):
            idx = random.sample(range(len(cols)), int(changing_cols))
            changing_cols = [cols[i] for i in sorted(idx)]
        elif isinstance(changing_cols, bool):
            changing_cols = [*cols] if changing_cols else []
        else:
            if not isinstance(changing_cols, (list, tuple)):
                changing_cols = [changing_cols]
            assert len(changing_cols) <= len(cols)
            assert all(c_col in cols for c_col in changing_cols)
        # preprocess number of changepoints
        tmp_num_changepoints = [1]*len(changing_cols)
        if isinstance(num_changepoints, (float, int)):
            tmp_num_changepoints = [int(num_changepoints)]*len(changing_cols)
        else:  # should be a list
            for i, n in enumerate(num_changepoints[:len(changing_cols)]):
                tmp_num_changepoints[i] = n
        num_changepoints = tmp_num_changepoints
        # preprocess drift
        tmp_drift_steps = None
        if isinstance(drift_steps, (float, int)):
            tmp_drift_steps = [
                [int(drift_steps)]*n for n in num_changepoints][:len(changing_cols)]
        else:  # should be a list
            tmp_drift_steps = [
                [int(1)]*n for n in num_changepoints][:len(changing_cols)]
            for i, d in enumerate(drift_steps[:len(changing_cols)]):
                n = num_changepoints[i]
                if isinstance(d, (float, int)):
                    tmp_drift_steps[i] = [int(d)]*n
                else:  # should be a list
                    tmp_drift_steps[i] = (d+[1]*(max(0, n-len(d))))[:n]
        drift_steps = tmp_drift_steps
        # generate changepoints
        changepoints = [[np.random.randint(change_start, max(0, size-steps)) for steps in drift]
                        for _, drift in zip(num_changepoints, drift_steps)]
        # preprocess mean, std
        if isinstance(init_mean, (float, int)):
            init_mean = [init_mean]*len(cols)
        init_mean = init_mean[:len(cols)]
        if isinstance(init_std, (float, int)):
            init_std = [init_std]*len(cols)
        init_std = init_std[:len(cols)]
        # compute changing mean, std
        if isinstance(mean_ball, (float, int)):
            mean_ball = [mean_ball]*len(changing_cols)
        elif isinstance(mean_ball, (tuple, list)):
            mean_ball = (mean_ball + [0] *
                         len(changing_cols))[:len(changing_cols)]
        if isinstance(std_ratio, (float, int)):
            std_ratio = [std_ratio]*len(changing_cols)
        elif isinstance(std_ratio, (tuple, list)):
            std_ratio = (std_ratio + [0] *
                         len(changing_cols))[:len(changing_cols)]
        for i, (ball, ratio) in enumerate(zip(mean_ball[:len(changepoints)], std_ratio[:len(changepoints)])):
            cpt = changepoints[i]
            # check for mean_ball input type
            if isinstance(ball, (float, int)):
                mean_ball[i] = [ball]*len(cpt)
            else:  # should be a list
                mean_ball[i] = (ball+[0]*len(cpt))[:len(cpt)]
            # check for std_ratio input type
            if isinstance(ratio, (float, int)):
                std_ratio[i] = [ratio]*len(cpt)
            else:  # should be a list
                std_ratio[i] = (ratio+[1]*len(cpt))[:len(cpt)]
        # generate distributions
        dist = []
        changing_dist = []
        changepoints = deque(zip(changepoints, drift_steps))
        for i, col in enumerate(cols):
            mean, std = init_mean[i], init_std[i]
            if col in changing_cols:
                i_ = i-len(dist)
                cpt = changepoints.popleft()
                stats, w = [], 1
                for j, (pts, drift) in enumerate(zip(*cpt)):
                    stats.append((pts, mean+w*mean_ball[i_][j],
                                  std*std_ratio[i_][j], drift))
                    w = -w
                dist.append(
                    ChangingNormalDistribution(mean, std, stats))
            else:
                dist.append(NormalDistribution(mean, std))
        # init class attributes
        self.changing_cols = changing_cols
        super().__init__(cols, [*dist, *changing_dist], size)
