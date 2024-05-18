import pandas as pd
from pandas import DataFrame

import preprocessing.merge as merge
from ds_extractor.issues import extract_issues_in_study_scope
from sampling.issues import AnnotatorScoresMerger


def classify_utterances_by_clustering(utterances_df: DataFrame) -> DataFrame:
    utr_cls = merge.UtterancesClassifier(assignee_clusters=10,
                                         reporter_clusters=12,
                                         others_clusters=10,
                                         vector_size=300)
    return utr_cls.classify_utterances(utterances_df)


def prepare_ml_dataset(issues_df: DataFrame):
    annotated_scores_df = pd.read_excel('../temp_data/issues_snapshot_sample.xlsx',
                                        index_col=[i for i in range(0, 8)],
                                        usecols=[i for i in range(0, 19)])
    merged_df = AnnotatorScoresMerger().merge(issues_df, annotated_scores_df)
    merged_df.to_csv('../temp_data/scored_issues_snapshots_w2v_cls.csv')
    print('done')


def extract_issues():
    snapshots = extract_issues_in_study_scope(pd.read_csv('../temp_data/issues_snapshot.csv', index_col='idx'))
    snapshots.to_csv('../temp_data/issues_snapshots_2022.csv')
    return snapshots


if __name__ == '__main__':
    issues_df = extract_issues()
    print('issues extracted')
    utterances_df = pd.read_csv('../temp_data/pp_utterances.csv')
    # extract the utterances that are not an automated comment
    utterances_df = utterances_df[~((utterances_df['author'].isin(['admin', 'u003']))
                                    & (utterances_df['comment_seq'] < 3)
                                    & (utterances_df['author_role'] == 'others'))]
    issues_df = merge.CommentsStatisticsAggregator().merge(issues_snapshots_df=issues_df,
                                                           utterances_df=utterances_df.copy().reset_index())
    print('merged with comments statistics')
    classified_issue_utterances = classify_utterances_by_clustering(utterances_df)
    print('utterances are classified')
    issues_df = merge.IssuesUtterancesClassesAggregator().merge_with_utterances_classes(issues_df,
                                                                                        classified_issue_utterances)
    print('issues dataframe enriched with utterances classes')
    prepare_ml_dataset(issues_df)
