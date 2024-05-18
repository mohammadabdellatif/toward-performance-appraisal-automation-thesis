import pandas as pd
from pandas import DataFrame

from ds_extractor.comments import CommentsDatasetExtractor
from ds_extractor.issues import extract_issues_in_study_scope
from preprocessing.comments import CommentsPreProcessor


def extract_utterances():
    print('extract comments from db')
    comments_extractor = CommentsDatasetExtractor('postgresql+psycopg2://admin:sami@127.0.0.1:5455/supportdb1',
                                                  "../temp_data",
                                                  thread_count=5)
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
    utterances_df = extract_utterances()
    generate_pre_processed_utterances(utterances_df)
