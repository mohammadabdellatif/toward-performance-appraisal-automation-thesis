import pandas as pd
from pandas import DataFrame

from clustering.issues import IssuesClustering
from exploration.issues_clustering import plot_issues_cluster
from sampling.issues import IssuesSampler

if __name__ == '__main__':
    sampler = IssuesSampler()
    issues_df = pd.read_csv('../temp_data/issues.csv')
    types = ['Ticket', 'Deployment', 'HD Service']

    issues_df = issues_df[(issues_df['issue_proj'].str.match('\w{2}\d{2}\w{1,}'))
                          & (issues_df['issue_type'].isin(types))
                          & (issues_df['issue_created'] >= '2022-01-01')
                          & (issues_df['issue_created'] <= '2022-12-31')
                          & pd.notna(issues_df['issue_resolution_date'])]
    features = ['processing_steps', 'issue_comments_count', 'issue_contr_count', 'wf_total_time']
    issues_df.loc[:, 'category'] = IssuesClustering().cluster(4, issues_df, features)
    sample = sampler.sample(360, issues_df)
    sample.to_csv('../temp_data/issues_sample.csv')

    issues_snapshots_df: DataFrame = pd.read_csv("../temp_data/issues_snapshot.csv")
    issues_snapshots_df = issues_snapshots_df[issues_snapshots_df['id'].isin(sample['id'])]
    issues_snapshots_df = issues_snapshots_df[
        ['id', 'issue_num', 'issue_proj', 'issue_reporter', 'issue_contr_count', 'issue_type', 'issue_priority',
         'turn', 'issue_assignee', 'started', 'ended', 'wf_total_time', 'processing_steps',
         'issue_comments_count']].sort_values([
        'id', 'started']).reset_index(drop=True)
    issues_snapshots_df['wf_total_time'] = round(issues_snapshots_df['wf_total_time'] / (60 * 60), 3)

    issues_snapshots_df.rename(inplace=True, columns={
        'issue_num': 'no',
        'issue_type': 'type',
        'issue_priority': 'priority',
        'issue_proj': 'project',
        'issue_reporter': 'reporter',
        'issue_assignee': 'assignee',
        'wf_total_time': 'spent hours',
        'issue_contr_count': 'contributors',
        'processing_steps': 'steps',
        'issue_comments_count': 'comments count',
        'turn': 'turn_no'
    })
    issues_snapshots_df['valid'] = True
    issues_snapshots_df['Q1'] = 0
    issues_snapshots_df['Q2'] = 0
    issues_snapshots_df['Q3'] = 0
    issues_snapshots_df['Notes'] = None

    issues_snapshots_df = issues_snapshots_df.set_index(
        keys=['id', 'no', 'project', 'reporter', 'type', 'priority', 'contributors', 'turn_no'])
    issues_snapshots_df.to_excel('../temp_data/issues_snapshot_sample.xlsx')
    issues_snapshots_df.to_csv('../temp_data/issues_snapshot_sample.csv')

    plot_issues_cluster(sample)
    print('done')
