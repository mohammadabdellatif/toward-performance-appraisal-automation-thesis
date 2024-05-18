import pandas as pd
from pandas import DataFrame

from ds_extractor.issues import extract_issues_in_study_scope
from preprocessing.merge import IssuesSnapshotsTfidf, SnapshotsUtterancesMerger, CommentsStatisticsAggregator
from sampling.issues import AnnotatorScoresMerger


def merge_issues_with_comments():
    issues_snapshots_df = pd.read_csv('../temp_data/issues_snapshots_2022.csv', index_col='idx')
    utterances_df = pd.read_csv('../temp_data/pp_utterances.csv', index_col=['issueid', 'id', 'utr_seq'])
    merged_snapshots_df = SnapshotsUtterancesMerger() \
        .combine_snapshot_comments_by_role(issues_snapshots_df, utterances_df.copy())
    merged_df = CommentsStatisticsAggregator().merge(issues_snapshots_df=merged_snapshots_df,
                                                     utterances_df=utterances_df.copy().reset_index())
    merged_df.to_csv('../temp_data/issues_snapshots_2022_comments.csv')


def build_tfidf_features() -> DataFrame:
    df = pd.read_csv('../temp_data/issues_snapshots_2022_comments.csv', index_col='idx')
    ml_df = IssuesSnapshotsTfidf(use_idf=False).vectorize_issues_comments(issues_snapshots_df=df)
    ml_df.to_csv('../temp_data/issues_snapshots_2022_ml.csv')
    print('TF-IDF dataset completed')
    return ml_df


def prepare_ml_dataset(tfidf_df: DataFrame):
    merger = AnnotatorScoresMerger()
    annotated_scores_df = pd.read_excel('../temp_data/issues_snapshot_sample.xlsx',
                                        index_col=[i for i in range(0, 8)],
                                        usecols=[i for i in range(0, 19)])
    merged_df = merger.merge(tfidf_df, annotated_scores_df)
    merged_df.to_csv('../temp_data/scored_issues_snapshots_tfidf.csv')
    print('done')


def extract_issues():
    snapshots = extract_issues_in_study_scope(pd.read_csv('../temp_data/issues_snapshot.csv', index_col='idx'))
    snapshots.to_csv('../temp_data/issues_snapshots_2022.csv')


if __name__ == '__main__':
    extract_issues()
    merge_issues_with_comments()
    tfidf_features = build_tfidf_features()
    prepare_ml_dataset(tfidf_features)
