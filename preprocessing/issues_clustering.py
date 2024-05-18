import enum

import numpy as np
import pandas as pd
from feature_engine.discretisation import ArbitraryDiscretiser
from sklearn.preprocessing import KBinsDiscretizer


class DiscretizeResults:
    def __init__(self, count, labels, borders, discrete_values):
        self.count = count
        self.labels = []
        self.labels.extend(labels)
        self.borders = borders
        self.discrete_values = discrete_values


def show_bins(result: DiscretizeResults, divider=1):
    """Shows the ranges of values the bins comprises"""
    for i in range(len(result.borders) - 1):
        print(f'Bin {i}: {str(int(result.borders[i] / divider))} -> {str(int(result.borders[i + 1] / divider))}')


def kmean_discretize(issues_df: pd.DataFrame):
    clusters = 6
    kbins = KBinsDiscretizer(n_bins=clusters, strategy='kmeans', encode='ordinal')
    wf_total_time_pins = kbins.fit_transform(np.array(issues_df['wf_total_time']).reshape(-1, 1))
    return DiscretizeResults(count=clusters,
                             labels=["cluster" + str(i) for i in range(0, clusters)],
                             borders=kbins.bin_edges_[0],
                             discrete_values=wf_total_time_pins)


def bins_discretize(issues_df: pd.DataFrame):
    hour = 60 * 60
    day = hour * 24
    bins = [0, day + 1, (day * 7) + 1, (day * 31), day * 60, int(day * (365 / 2)), issues_df['wf_total_time'].max()]
    bins_border = {'wf_total_time': bins}
    ad = ArbitraryDiscretiser(binning_dict=bins_border)
    temp_df = issues_df[['wf_total_time']]
    temp_df.fillna(0)
    wf_bins = ad.fit_transform(temp_df[['wf_total_time']])
    return DiscretizeResults(count=len(bins) - 1,
                             labels=["Within a Day", "Within a Week", "Within 30 Days", "Within 60 Days",
                                     "Within six months", "More"],
                             borders=bins,
                             discrete_values=wf_bins['wf_total_time'].to_numpy().reshape(-1, 1))

class DiscretizeMode(enum.Enum):
    BINS = 1
    KMEANS = 2

class TotalTimeClusteringPreProcess:

    def pre_process(self, issues_df: pd.DataFrame, mode: DiscretizeMode):
        discretizer = bins_discretize if mode == DiscretizeMode.BINS else kmean_discretize
        results = discretizer(issues_df)
        issues_df['wf_total_time_bin'] = pd.Series(results.discrete_values[:, 0], index=issues_df.index)
        show_bins(results, divider=60 * 60 * 24)
        return results, issues_df
