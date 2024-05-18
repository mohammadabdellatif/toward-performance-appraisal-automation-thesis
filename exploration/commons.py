import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from pandas import DataFrame
from scipy.stats import gaussian_kde


def plot_line(ax: Axes, x, y, xlabel: str, ylabel: str):
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_density_scatter(ax, x, y, label: str = None, size: int = 10):
    if len(x) <= 1 or len(y) <= 1:
        ax.scatter(y=y, x=x)
        return
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(y=y, x=x, c=z, s=size, label=label)


def plot_hist(ax: Axes,
              values,
              bins: int,
              xlabel: str,
              ylabel: str,
              grid: str = None,
              max_y: int = None,
              xsteps: int = 50,
              ysteps: int = 50,
              xrotation: int = 0):
    x_values = range(0, values.max(), xsteps)

    ax.hist(values, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values, rotation=xrotation)

    if max_y is not None:
        ax.set_yticks(range(0, max_y, ysteps))
    if grid is not None:
        ax.grid(True, axis=grid)


def plot_box(ax: Axes,
             values,
             label: str,
             vertical: bool = True,
             steps: int = 50,
             rotation: int = 0,
             grid: str = None):
    ax.boxplot(values, vert=vertical)
    xticks = range(0, values.max(), steps)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=rotation)
    ax.set_xlabel(label)
    ax.set_yticklabels([''])
    if grid is not None:
        ax.grid(True, axis=grid)


def pair_plot(df_pp: DataFrame, cat_col: str = None):
    if cat_col is None:
        sns.pairplot(df_pp,corner=True)
        return
    labels = []
    cat = df_pp[cat_col].drop_duplicates().sort_values()
    for c in cat:
        c_len = len(df_pp[df_pp[cat_col] == c])
        labels.append(f'{c_len} issues')

    pair_plot = sns.pairplot(df_pp, hue=cat_col, palette=sns.color_palette("tab10")[0:(len(cat))], corner=True)

    lgnd = pair_plot.legend
    for i, l in enumerate(labels):
        lgnd.texts[i].set_text(l)
