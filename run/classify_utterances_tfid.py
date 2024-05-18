import pandas as pd

from preprocessing.merge import TFIDFUtterancesClassifier

if __name__ == '__main__':
    utterances_df = pd.read_csv('../temp_data/pp_utterances.csv', index_col=['issueid', 'id', 'utr_seq'])
    classifier = TFIDFUtterancesClassifier(assignee_clusters=10,others_clusters=10,reporter_clusters=12)
    classifier.classify_utterances(utterances_df)
    print('done')
