import math

import pandas as pd
from pandas import DataFrame

from clustering.issues import IssuesClustering


class IssuesSampler:

    def sample(self, n, issues_df: DataFrame):
        features = ['processing_steps', 'issue_comments_count', 'issue_contr_count', 'wf_total_time']
        clustering = IssuesClustering()
        df = issues_df.copy()
        df['category'] = clustering.cluster(4, df, features)
        categories = df['category'].drop_duplicates()
        c_share = math.ceil(n / len(categories))
        samples = []
        for c in categories:
            df_c: DataFrame = df[df['category'] == c].copy()
            samples.append(df_c.sample(c_share, random_state=42))
        return pd.concat(samples)


class AnnotatorScoresMerger:

    def merge(self, issues_df: DataFrame, annotated_scores_df: DataFrame) -> DataFrame:
        annotated_scores_df = annotated_scores_df.reset_index()
        issues_df = issues_df[issues_df['id'].isin(annotated_scores_df['id'])].copy()
        issues_df['Q1'] = 0
        issues_df['Q2'] = 0
        issues_df['Q3'] = 0
        for idx, row in annotated_scores_df.iterrows():
            issue_id = row['id']
            turn = row['turn_no']
            snapshot_id = (issues_df['id'] == issue_id) & (issues_df['turn'] == turn)
            issues_df.loc[snapshot_id, 'Q1'] = row['Q1']
            issues_df.loc[snapshot_id, 'Q2'] = row['Q2']
            issues_df.loc[snapshot_id, 'Q3'] = row['Q3']

        return issues_df
