import argparse
import json
import os
from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from change_graph import ChangeGraph
from feature.scaler import StandardScaler
from simulator.dataset import RandomDataset


def eval():
    if not os.path.exists('times_res'):
        os.mkdir('times_res')
    # vars
    feature_count = [10, 100, 200]
    sample_size = [10000, 50000]
    change_ratio = [.1, .3]
    top_n = [1, 3, 5]
    times = 50
    cols = ['sample_size', 'feature_count',
            'change_ratio', *[f'top_{n}' for n in top_n]]
    loop_results = []
    # process
    for t in range(times):
        t_bar = tqdm(total=(len(feature_count) *
                     len(sample_size) * len(change_ratio)))
        print(f'\nIteration {t}')
        for size in sample_size:
            for f_count in feature_count:
                for cr in change_ratio:
                    t_bar.update(1)
                    changing_cols_count = 0 if not cr else max(
                        1, int(f_count * cr))
                    num_changepoints = 0 if not changing_cols_count else np.random.randint(
                        1, 6)
                    dataset = RandomDataset(cols=f_count,
                                            changing_cols=changing_cols_count,
                                            change_start=100,
                                            num_changepoints=num_changepoints,
                                            drift_steps=50,
                                            init_mean=list(np.random.randint(
                                                1, 11, size=f_count)),
                                            init_std=list(np.random.randint(
                                                1, 11, size=f_count)),
                                            mean_ball=list(np.random.randint(
                                                5, 21, size=num_changepoints)),
                                            size=size)
                    dataset.generate()
                    df = dataset.to_df(features_only=False)
                    # plt.plot(df)
                    # plt.show()
                    # return
                    model = ChangeGraph(num_features=dataset.cols,  # if not dataset.changing_cols else dataset.changing_cols,
                                        cat_features=[],
                                        window_size=100,
                                        num_scaler=StandardScaler,
                                        custom_start=100,
                                        threshold=.6)
                    # accumulators
                    top_n_score_list = []
                    # average_scores = []
                    # all_feature_scores = defaultdict(list)
                    for i, x in df.iterrows():
                        x = x.to_dict()
                        triggered, _, _, num_feature_scores = model.learn_one(
                            i, x, None)
                        num_avg_score, feature_list, feature_score_list = num_feature_scores
                        # HERE store average score
                        # ...
                        # store scores per feature in a dict
                        # for j, f_name in enumerate(feature_list):
                        #     all_feature_scores[f_name].append(
                        #         feature_score_list[j])
                        target_cols_str: str = x['target_cols']
                        top_n_iter_scores = []
                        if target_cols_str is not None:
                            target_features = set(target_cols_str.split(','))
                            feature_size = len(target_features)
                            for n in top_n:
                                n = max(feature_size, n)
                                pred_features = set(feature_list[:n])
                                # print(pred_features, target_features)
                                inter = target_features & pred_features
                                top_n_score = len(
                                    inter) / len(target_features) if len(target_features) != 0 else 0
                                top_n_iter_scores.append(top_n_score)
                            top_n_score_list.append(top_n_iter_scores)
                    # feature_scores = np.asarray(
                    #     list(zip(*all_feature_scores.values())))
                    # feature_names = all_feature_scores.keys()
                    # print(top_n_score_list)
                    res_mean = np.mean(top_n_score_list, axis=0)
                    loop_results.append(
                        [size, f_count, cr, *res_mean])
                    # print(loop_results)
                    # print(feature_scores)
                    # split = np.asarray(np.split(np.asarray(feature_scores), 20))
                    # maxes = np.max(split, axis=1)

        # print(loop_results)
        res_df = pd.DataFrame(loop_results, columns=cols)
        res_df.to_csv(f'times_res/res_iter_{t}.csv', index=False)
        loop_results = []
        return


if __name__ == '__main__':
    eval()
    # print('test')
