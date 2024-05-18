import pandas as pd
from pandas import DataFrame


class CommentsCountPreProcess:

    def pre_process(self, issues_df: DataFrame):
        issues_df.loc[pd.isna(issues_df['issue_comments_count']), 'issue_comments_count'] = 0
        return issues_df


class ContributorsPreProcess:

    def pre_process(self, issues_df: DataFrame):
        return issues_df.fillna(value={'issue_contr_count': 1})


class TotalProcessingStepsPreProcess:

    def pre_process(self, issues_df: pd.DataFrame):
        wf_entries_columns = [i for i, c in enumerate(issues_df.columns) if 'wfe_' in c]

        issues_df['processing_steps'] = 0
        for i, row in issues_df.iterrows():
            for c in issues_df.columns[wf_entries_columns]:
                if row[c] > 0:
                    issues_df.at[i, 'processing_steps'] = issues_df.at[i, 'processing_steps'] + row[c]

        return issues_df


class TotalTimePreProcess:

    def pre_process(self, issues_df: pd.DataFrame):
        wf_columns = [i for i, c in enumerate(issues_df.columns) if c.startswith('wf_')]
        # TODO this fails for snapshot issues
        zeros = pd.DataFrame(columns=issues_df.columns[wf_columns], data=[[0 for c in wf_columns]])
        wf_zeros = issues_df.fillna(zeros)
        issues_df['wf_total_time'] = wf_zeros.iloc[:, wf_columns].sum(axis=1)
        issues_df.fillna({'wf_total_time': 0},inplace=True)
        return issues_df
