import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandas import DataFrame

from exploration.commons import plot_density_scatter
from preprocessing.issues_clustering import DiscretizeResults


def wf_steps_summary(issues_df: DataFrame, divider: int = 60):
    wf_columns = __list_wf_steps_time_columns(issues_df)
    subset_df = issues_df.iloc[:, wf_columns]

    frames = []
    for c in subset_df.columns:
        single_c = __single_wf_step_df(subset_df, c, divider)
        if single_c is not None:
            c_summary = single_c.describe()
            int_count = c_summary.loc['count', c]
            c_summary[c] = c_summary[c].map('{:,.2f}'.format)
            c_summary.loc['int_count', c] = int_count
            c_summary = c_summary.rename(columns={c: c.replace('wf_', '').replace('_', ' ')})
            frames.append(c_summary.transpose())
    steps_statistics = pd.concat(frames).sort_values('int_count', ascending=False)
    return steps_statistics.drop(columns='int_count')


def __list_wf_steps_time_columns(issues_df):
    return [i for i, c in enumerate(issues_df.columns) if c.startswith('wf_') and not c.startswith('wf_total_time')]


def __single_wf_step_df(subset_df, c, divider):
    single_c = subset_df[[c]]
    single_c = single_c[(single_c[c] != 0) & pd.notna(single_c[c])]
    if len(single_c) == 0:
        return None
    return np.ceil(single_c / divider)


def plot_wf_spent_summary(issues_df: DataFrame, ax: Axes, divider: int = 60):
    wf_columns = __list_wf_steps_time_columns(issues_df)
    # Creating axes instance
    subset_df = issues_df.iloc[:, wf_columns]

    data = []
    m = 0
    for c in subset_df.columns:
        single_c = __single_wf_step_df(subset_df, c, divider)
        if single_c is None:
            continue
        values = single_c.to_numpy().reshape(-1)
        t_m = max(values)
        m = max(m, t_m)
        l = f'{single_c.columns[0]} [n: {len(values):,}]'
        data.append((l, values))

    data.sort(key=lambda e: len(e[1]))
    ax.boxplot([d for l, d in data], vert=False)
    ax.set_yticklabels([l for l, d in data])
    # ax.set_xticks(range(0, math.ceil(m), math.ceil(m / 10)))
    ax.grid(True)
    # for i in range(0, len(data)):
    #     ax.text(s='n=' + str(len(data[i][1])), y=i + 1, x=max(data[i][1]) + 20)


def plot_total_time_bin(issues_df: DataFrame, result: DiscretizeResults, ax: Axes):
    frequencies = issues_df['wf_total_time_bin'].value_counts()
    x = range(0, result.count)
    ax.bar(x, frequencies, width=0.5)
    ax.grid(True, axis="y")
    # ax.set_title("Issues Resolution time")
    ax.set_xticks(x)
    ax.set_xticklabels(result.labels, rotation=90)
    ax.set_xlabel("Group")
    ax.set_ylabel("Frequency")


def plot_box_by_issue_category(issues_df: DataFrame, result: DiscretizeResults, ax: Axes):
    bins_df = issues_df[['wf_total_time', 'wf_total_time_bin']]
    labels = [l for l in result.labels]
    boxes = []
    y_ticks = [result.borders[0]]
    for i in range(len(result.borders) - 1):
        y_ticks.append(result.borders[i + 1])
        boxes.append(bins_df[bins_df['wf_total_time'].between(result.borders[i], result.borders[i + 1] - 1)][
                         'wf_total_time'].to_numpy())

    ax.boxplot(boxes)
    ax.grid(True, axis="y")
    ax.set_xlabel('issue spent time category')
    ax.set_ylabel('Workflow time in days')
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels([str(int(l / (60 * 60 * 24))) for l in y_ticks])


def plot_issues_by_year(issues_df: DataFrame, ax: Axes):
    ax.grid(True, axis="y")
    ax.set_xlabel("Year")
    ax.set_ylabel("Issues count")
    # ax.set_title("Reported Issues by Year")

    by_month = issues_df[['issue_created', 'issue_priority']].copy()
    by_month['issue_year'] = by_month['issue_created'].transform(
        lambda x: f'%i' % (x.year))

    by_month_grouped = by_month.groupby('issue_year').count().sort_values('issue_year')

    ax.bar(by_month_grouped.index, by_month_grouped['issue_created'])
    ax.set_xticklabels(by_month_grouped.index, rotation=90)


def plot_issues_by_month(issues_df: DataFrame, ax: Axes):
    pr_color = {
        'Blocker': '#5c1010',
        'Highest': '#940000',
        'High': '#c30101',
        'Medium': '#f97300',
        'Low': '#ffb21b',
        'Lowest': 'green'
    }
    ax.grid(True, axis="y")
    ax.set_xlabel("Month/Year")
    ax.set_ylabel("Issues count")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['' if i == 0 else ('%i' % i) for i in range(1, 13)])
    # ax.set_title("Reported Issues by Month of Year")

    by_month = issues_df[['issue_created', 'issue_priority']].copy()
    by_month['issue_month'] = by_month['issue_created'].transform(lambda x: x.month)

    offset = -0.4
    for p in pr_color.keys():
        p_by_month = by_month[by_month['issue_priority'] == p]
        by_month_grouped = p_by_month.groupby('issue_month').count().sort_values('issue_month')
        ax.bar(by_month_grouped.index + offset, by_month_grouped['issue_created'], color=pr_color[p], width=0.17,
               label=p)
        offset = offset + 0.16

    ax.legend(loc='upper center', ncol=3)

    ax.set_ymargin(0.1)


def plot_issues_by_priority(issues_df: DataFrame, ax: Axes):
    count_by_priority = issues_df[['issue_created', 'issue_priority']].groupby('issue_priority').count()
    # count_by_priority = count_by_priority.reset_index()
    ordered_list = ['Lowest', 'Low', 'Medium', 'High', 'Highest', 'Blocker']
    ax.bar(ordered_list, [count_by_priority.loc[x, 'issue_created'] for x in ordered_list])
    ax.grid(True, axis="y")
    # ax.set_title("Issues Count By Priority")
    ax.set_yticks(count_by_priority['issue_created'])
    ax.set_ylabel("Count")
    ax.set_xlabel("Priorities")


def plot_comments_count_frequency(issues_df: DataFrame, ax: Axes):
    comments = issues_df[['issue_comments_count']].fillna(0)
    freq = comments['issue_comments_count'].value_counts()
    freq_df = freq.to_frame()

    ax.bar(freq_df.index, freq_df['issue_comments_count'], width=1)
    ax.set_xlabel("Comments count")
    ax.set_ylabel("Frequency")
    # ax.set_title("Comments Count Frequency")
    ax.set_yticks(range(0, int(freq_df['issue_comments_count'].max()), 100))
    ax.set_xticks(range(0, int(freq_df.index.max()), 10))
    ax.grid(True)


def plot_relation_between_processing_steps_and_time(issues_df: DataFrame,
                                                    ax: Axes,
                                                    by_field=None,
                                                    divider: int = None,
                                                    max_x=None,
                                                    max_y=None):
    __plot_time_vs_field(issues_df, 'processing_steps', ax, by_field, divider, yticks_span=5, max_x=max_x, max_y=max_y)
    ax.set_ylabel("Processing steps")


def plot_comments_count_time_spent(issues_df: DataFrame,
                                   ax: Axes,
                                   by_field=None,
                                   divider: int = None,
                                   max_x=None,
                                   max_y=None):
    __plot_time_vs_field(issues_df, 'issue_comments_count', ax, by_field, divider, max_x=max_x, max_y=max_y)
    ax.set_ylabel("Comments count")


def __plot_time_vs_field(issues_df: DataFrame,
                         other_field: str,
                         ax: Axes,
                         by_field: str,
                         divider: int,
                         yticks_span: int = 50,
                         max_x: int = None,
                         max_y: int = None):
    divider = (60 * 60) if divider is None else divider
    total_time = issues_df['wf_total_time'] / divider
    if by_field is not None:
        field_values = issues_df[by_field].drop_duplicates()
        for b in field_values:
            df = issues_df[issues_df[by_field] == b]
            ax.scatter(y=df[other_field], x=df['wf_total_time'] / divider, s=10,
                       label=f"{len(issues_df)} for {len(issues_df['issue_proj'].drop_duplicates())} projects")
    else:
        total_time = issues_df['wf_total_time'] / divider
        x = total_time.to_numpy()
        y = issues_df[other_field].to_numpy()
        plot_density_scatter(ax, x, y,
                             label=f"{len(issues_df)} for {len(issues_df['issue_proj'].drop_duplicates())} projects")
    ax.set_xlabel("Total process time (in Hours)")

    max_x = int(max_x / divider) if max_x is not None else int(total_time.max())
    ax.set_xticks(range(0, max_x, 1000))
    ax.set_yticks(range(0, max_y if max_y is not None else int(issues_df[other_field].max()), yticks_span))

    # ax.set_xticklabels(range(0, max_x, 500), rotation=90)
    ax.legend()
    ax.grid(True)


def plot_comments_count_summary(issues_df: DataFrame, ax: Axes):
    ax.boxplot(issues_df.loc[:, 'issue_comments_count'].to_numpy(), vert=False)
    ax.set_xlabel("Comments Count")
    ax.set_xticks(range(0, int(issues_df.loc[:, 'issue_comments_count'].max()), 10))
    ax.set_ylabel("")
    ax.set_yticklabels([""])
    # ax.set_title("Comments Count Summary")
    ax.grid(True)


def plot_wf_total_time_summary(issues_df: DataFrame, ax: Axes):
    ax.grid(True)
    values = np.ceil(issues_df['wf_total_time'] / (60 * 60)).to_numpy()
    ax.boxplot(values, vert=False)
    last = int(max(values))
    ax.set_xticks(range(0, last, 1000))
    # ax.set_title("Total process time Summary")
    # ax.set_xlabel("Processing Time (Hours)")
    ax.set_yticklabels([''])


def plot_issues_processing_steps(issues_df: DataFrame, ax: Axes):
    ax.grid(True)
    ax.boxplot(issues_df.loc[:, 'processing_steps'], vert=False)
    # ax.set_title('Processing steps summary')
    # ax.set_xlabel('Steps counts')
    ax.set_yticklabels([''])


def plot_processing_steps_frequency(issues_df: DataFrame, ax: Axes):
    freq = issues_df['processing_steps'].value_counts()
    ax.bar(freq.index, freq)
    ax.grid(True)
    ax.set_xlabel("processing steps")
    ax.set_ylabel("Issues count")
    ax.set_yticks(range(0, max(freq), 100))
    # ax.set_title("Issues count by processing steps")


def plot_assignees_participation(df: DataFrame, ax: Axes):
    assignee_work_df = df.groupby('newvalue').count()
    ax.grid(True)
    ax.boxplot(assignee_work_df['issueid'])
    ax.set_yticks(range(0, assignee_work_df['issueid'].max(), 50))
    # ax.set_title('Issues done by assignee summary')
    ax.set_ylabel('number of issues')
    ax.set_xticklabels([''])


def plot_issue_contributors_summary(contr_df: DataFrame, ax: Axes):
    ax.boxplot(contr_df, vert=False)
    ax.grid(True)
    ax.set_yticklabels([''])
    ax.set_ylabel("Number of Contributors")


def plot_issue_contributors_frequency(contr_df: DataFrame, ax: Axes):
    df = contr_df['issue_contr_count'].value_counts().to_frame()
    ax.bar(df.index, df['issue_contr_count'])
    # ax.grid(True)
    # ax.set_title("Issue Contributors Frequencies")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Number of Contributors")
    ax.set_yticks(df['issue_contr_count'])
    ax.set_yticklabels(['' if v < 50 else str(v) for v in df['issue_contr_count']])


def plot_issue_contributors_to_total_spent_time(issues_df: DataFrame, ax: Axes):
    x = issues_df['issue_contr_count'].to_numpy()
    y = (issues_df['wf_total_time'] / (60 * 60)).to_numpy()
    plot_density_scatter(ax, x, y)
    # ax.set_title("Contributors effect to Total spent time")
    ax.set_ylabel("Total Spent Time (in hours)")
    ax.set_xlabel("Number of Contributors")


def plot_issue_contributors_to_total_comments(issues_df: DataFrame, ax: Axes):
    plot_density_scatter(y=issues_df['issue_comments_count'].to_numpy(),
                         x=issues_df['issue_contr_count'].to_numpy(),
                         ax=ax)
    ax.grid(True)
    ax.set_ylabel("Comments count")
    ax.set_xlabel("Number of Contributors")
