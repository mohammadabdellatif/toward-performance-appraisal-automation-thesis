import sys

import nltk
import pandas as pd
from pandas import DataFrame

from ds_extractor.comments import CommentsDatasetExtractor, CommentsFilterByIssuesIDs
from ds_extractor.issues import extract_issues_in_study_scope
from preprocessing.comments import CommentsPreProcessor


def extract_utterances(extra_filter: object):
    print('extract comments from db')
    comments_extractor = CommentsDatasetExtractor('postgresql+psycopg2://admin:sami@127.0.0.1:5455/supportdb1',
                                                  "../temp_data",
                                                  thread_count=5,
                                                  extra_filter=extra_filter)
    comments_extractor.process()
    return pd.read_csv('../temp_data/utterances.csv', index_col='id', na_filter=False)


def generate_pre_processed_utterances(utterances_df: DataFrame):
    print('pre-process utterances')
    in_scope_issues_df = pd.read_csv('../temp_data/issues_snapshot.csv', index_col='idx')
    in_scope_issues_df = extract_issues_in_study_scope(in_scope_issues_df)
    comments_processor = CommentsPreProcessor()
    pp_utterances_df = comments_processor.preprocess(utterances_df, in_scope_issues_df)
    pp_utterances_df.to_csv('../temp_data/pp_utterances.csv', na_rep='')


if __name__ == '__main__':
    nltk.download('punkt')
    extra_filter = None
    if len(sys.argv) > 1 and sys.argv[1] == 'sample_only':
        ids = [1004285]
        sample_df = pd.read_excel('../temp_data/issues_snapshot_sample.xlsx',
                                  index_col=[i for i in range(0, 8)],
                                  usecols=[i for i in range(0, 19)])
        sample_df.reset_index(inplace=True)
        ids = sample_df['id'].drop_duplicates().to_list()
        print(f'number of sampled issues {len(ids)}')
        extra_filter = CommentsFilterByIssuesIDs(issue_ids=ids).filter
    utterances_df = extract_utterances(extra_filter)
    generate_pre_processed_utterances(utterances_df)
