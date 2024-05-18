import numpy as np
import pandas as pd
from pandas import DataFrame

from clustering.projects import ProjectsClustering


class ProjectsPreProcess:

    def __init__(self):
        pass

    def pre_process(self, issues_df: DataFrame,
                    features: list = ['count', 'wf_total_time_mean', 'issue_comments_count_mean',
                                      'issue_contr_count_mean', 'processing_steps_mean'],
                    clusters: int = 5):
        summary = ProjectsSummaryExtractor().summary_from_issues(issues_df)
        features.insert(0, 'issue_proj')
        labels = ProjectsClustering().cluster_projects(summary[features], n=clusters)
        summary['category'] = labels
        return summary

    def merge(self, issues_df: DataFrame, labeled_projects: DataFrame):
        for i, r in issues_df.iterrows():
            issues_df.loc[i, 'proj_category'] = \
                labeled_projects[labeled_projects['issue_proj'] == issues_df.loc[i, 'issue_proj']]['category'].values[0]

#
# ppp = ProjectsPreProcess()
# issues_df = pd.read_csv('../temp_data/issues_preprocessed.csv')
# labeled_proj = ppp.pre_process(issues_df)


class ProjectsSummaryExtractor:
    __features = {
        "count": ('wf_total_time', len),
        "wf_total_time_mean": ('wf_total_time', np.mean),
        "issue_comments_count_mean": ('issue_comments_count', np.mean),
        "issue_contr_count_mean": ('issue_contr_count', np.mean),
        "processing_steps_mean": ('processing_steps', np.mean),
    }

    def summary_from_issues(self, issues_df):
        aggregated = issues_df.groupby(['issue_proj']).aggregate(**ProjectsSummaryExtractor.__features)
        aggregated = aggregated.round(0)
        aggregated = aggregated.reset_index(names=['issue_proj'])
        aggregated.fillna(inplace=True, value={key: 0 for key in ProjectsSummaryExtractor.__features.keys()})
        return aggregated
