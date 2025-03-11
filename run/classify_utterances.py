from pandas import DataFrame
import pandas as pd
import preprocessing.merge as merge


def classify_utterances_by_clustering(utterances_df: DataFrame) -> DataFrame:
    utr_cls = merge.UtterancesClassifier(assignee_clusters=10,
                                         reporter_clusters=12,
                                         others_clusters=10,
                                         vector_size=300)
    return utr_cls.classify_utterances(utterances_df)


if __name__ == '__main__':
    utterances_df = pd.read_csv('../temp_data/pp_utterances.csv')
    classify_utterances_by_clustering(utterances_df)
    print("done")
