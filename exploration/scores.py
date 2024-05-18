import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.pyplot import figure
from pandas import DataFrame

from exploration.commons import plot_hist


class Correlation:

    def correlation(self, scored_issues_df: DataFrame):
        scored_issues_df = scored_issues_df[
            (scored_issues_df['Q1'] > 0) & (scored_issues_df['Q2'] > 0) & (scored_issues_df['Q3'] > 0)]
        scored_issues_df = scored_issues_df.drop(
            columns=["id", "started", "ended", "issue_num", "issue_proj", "issue_reporter", "issue_assignee", "turn"])
        scored_issues_df = scored_issues_df.drop(
            columns=[c for c in scored_issues_df.columns if c != 'wf_total_time' and c.startswith('wf')])
        return scored_issues_df.corr(numeric_only=True)


if __name__ == '__main__':
    merged_df = pd.read_csv('../temp_data/scored_issues_snapshots.csv', index_col='idx')
    frequencies = merged_df['Q1'].value_counts();
    print(frequencies)
    fig = figure(figsize=(6, 6))
    ax: Axes = fig.add_subplot(111)
    ax.hist(merged_df['Q1'])
    # plot_hist(ax, values=merged_df['Q1'], bins=6, xlabel="Scores", xsteps=1, ysteps=1, ylabel="Frequency")

    merged_df = merged_df[merged_df['Q1'] != 0]

    # fig = figure(figsize=(5, 20))
    # ax = fig.add_subplot(411)
    # ax.set_xlabel('comments')
    # plot_density_scatter(ax, y=merged_df['Q1'].to_numpy(),
    #                      x=merged_df['issue_comments_count'].to_numpy())
    # ax = fig.add_subplot(412)
    # ax.set_xlabel('Assignee utterances')
    # plot_density_scatter(ax, y=merged_df['Q1'].to_numpy(),
    #                      x=merged_df['assignee_utterances_count'].to_numpy())
    # ax = fig.add_subplot(413)
    # ax.set_xlabel('Reporter comments')
    # plot_density_scatter(ax, y=merged_df['Q1'].to_numpy(),
    #                      x=merged_df['reporter_comments_count'].to_numpy())
    #
    # ax = fig.add_subplot(414)
    # ax.set_xlabel('Others Comments')
    # plot_density_scatter(ax, y=merged_df['Q1'].to_numpy(), x=merged_df['others_comments_count'].to_numpy())

    # plt.scatter(merged_df['reporter_comments'],merged_df['Q1'])

    # plot_issues_cluster(merged_df, category_col='Q3')
    # pair_plot(merged_df[['wf_total_time',
    #                      'issue_comments_count',
    #                      'processing_steps',
    #                      'assignee_comments',
    #                      'reporter_comments',
    #                      'others_comments',
    #                      'Q1']])
    plt.show()
