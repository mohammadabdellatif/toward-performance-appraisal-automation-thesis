import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from clustering.issues import IssuesClustering

mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer


def cluster_then_3dplot():
    issues_df = pd.read_csv('../temp_data/issues.csv')
    types = ['Ticket', 'Deployment', 'HD Service']

    issues_df = issues_df[(issues_df['issue_proj'].str.match('\w{2}\d{2}\w{1,}'))
                          & (issues_df['issue_type'].isin(types))
                          & (issues_df['issue_created'] >= '2022-01-01')
                          & (issues_df['issue_created'] <= '2022-12-31')
                          & pd.notna(issues_df['issue_resolution_date'])]
    features = ['processing_steps', 'issue_comments_count', 'issue_contr_count', 'wf_total_time']
    issues_df['category'] = IssuesClustering().cluster(4, issues_df.copy(), features)
    for f in features:
        print(f'{f}: mean= {issues_df[f].mean()}, std={issues_df[f].std()}')
    plot_issues_cluster(issues_df)


def plot_issues_cluster(issues_df,category_col: str = 'category'):
    c_max = int(issues_df['issue_contr_count'].drop_duplicates().max())
    contr_size = [1 for c in range(0, c_max)]
    legends = []
    for i in range(0, c_max):
        contr_size[i] = 10 + i * 10

    legends.append(Line2D([], [], color="white", marker='o', markerfacecolor='black', markersize=6,
                          label=f'One contributor'))
    legends.append(Line2D([], [], color="white", marker='o', markerfacecolor='black', markersize=10,
                          label=f'{c_max} contributors'))
    fig1 = plt.figure(figsize=(8, 8))
    # fig2 = plt.figure(figsize=(8, 8))
    fig1_ax: Axes = fig1.add_subplot(projection='3d')
    colors = sns.color_palette("tab10")
    categories = issues_df[category_col].drop_duplicates().sort_values()

    x = ('Total time (Days)', 'wf_total_time')
    y = ('Comments Count', 'issue_comments_count')
    z = ('Processing Steps', 'processing_steps')

    for i, c in enumerate(categories):
        # Plot the 3D surface
        df_c = issues_df[issues_df[category_col] == c]
        x_v = df_c[x[1]]/(60*60*24)
        y_v = df_c[y[1]]
        z_v = df_c[z[1]]
        p_size = [contr_size[int(v) - 1] for v in df_c['issue_contr_count']]

        legends.append(Line2D([], [], color="white", marker='o', markerfacecolor=colors[i],
                              label=f'{len(df_c)} items in cluster {c}'))
        fig1_ax.scatter(x_v, y_v, z_v, s=p_size, color=colors[i])

    __set_ticks_and_labels(fig1_ax, x, y, z)
    fig1_ax.legend(handles=legends)
    plt.show()


def __set_ticks_and_labels(ax, x, y, z):
    if len(z) > 2:
        ax.set_zticks(z[2])
    if len(x) > 2:
        ax.set_xticks(x[2])
    if len(y) > 2:
        ax.set_yticks(y[2])
    # Plot the 3D surface
    ax.set_xlabel(x[0])
    ax.set_ylabel(y[0])
    ax.set_zlabel(z[0])


if __name__ == '__main__':
    cluster_then_3dplot()
