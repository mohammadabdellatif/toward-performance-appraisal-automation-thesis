import pandas as pd
from pandas import DataFrame, CategoricalDtype
from sklearn.model_selection import train_test_split


def rename_utterances_class_features(X):
    r_cols = {
        ('assignee', 0): 'open_close',
        ('assignee', 1): 'inform',
        ('assignee', 2): 'user_mention',
        ('assignee', 3): 'resolution',
        ('assignee', 4): 'technical',
        ('assignee', 5): 'investigation',
        ('assignee', 6): 'assignment_update',
        ('assignee', 7): 'reminder',
        ('assignee', 8): 'status_update',
        ('assignee', 9): 'support_session',

        ('others', 0): 'open_close',
        ('others', 1): 'user_mention',
        ('others', 2): 'investigation',
        ('others', 3): 'reminder',
        ('others', 4): 'assignment_update',
        ('others', 5): 'technical',
        ('others', 6): 'request',
        ('others', 7): 'resolution_update',
        ('others', 8): 'update_request',
        ('others', 9): 'resolution',

        ('reporter', 0): 'user_mention',
        ('reporter', 1): 'open_close',
        ('reporter', 2): 'attach_info',
        ('reporter', 3): 'inform_1',
        ('reporter', 4): 'inform_2',
        ('reporter', 5): 'technical_1',
        ('reporter', 6): 'resolution',
        ('reporter', 7): 'technical_2',
        ('reporter', 8): 'technical_3',
        ('reporter', 9): 'technical_4',
        ('reporter', 10): 'support_session',
        ('reporter', 11): 'request',
    }

    rename = {}
    for c in r_cols:
        rename[f'utr_{c[0]}_{c[1]}'] = f'utr_{c[0]}_{r_cols[c]}'
    X.rename(columns=rename, inplace=True)


def combine_utterances_class_features(X):
    combines = {
        ('utr_reporter_inform_1', 'utr_reporter_inform_2'): 'utr_reporter_inform',
        ('utr_reporter_technical_1', 'utr_reporter_technical_2', 'utr_reporter_technical_3',
         'utr_reporter_technical_4'): 'utr_reporter_technical'
    }
    for cs in combines:
        X[combines[cs]] = 0
        for c in cs:
            X[combines[cs]] = X[combines[cs]] + X[c]
            X.drop(columns=c, inplace=True)


def calculate_utterances_percentage(X):
    utr_f = [f for f in X.columns if f.startswith('utr_')]
    for idx, row in X.iterrows():
        for f in utr_f:
            utr_count = X.loc[idx, f"{f.split('_')[1]}_utterances_count"]
            X.loc[idx, f] = round(X.loc[idx, f] / (1 if utr_count == 0 else utr_count), 2)
    # X.drop(columns=['assignee_utterances_count', 'others_utterances_count', 'reporter_utterances_count'],
    #        inplace=True)


def build_dataset(learning_df,
                  class_to_predict=None,
                  add_dummies: bool = True,
                  utterances_as_percentage: bool = True,
                  drop_categories: bool = True):
    learning_df = learning_df[learning_df['Q1'] > 0].copy()
    modify_class_labels(class_to_predict, learning_df)

    # learning_df.loc[:,'Q1'] = learning_df['Q1'].astype(str)

    x = learning_df.drop(columns=['Q1', 'Q2', 'Q3', 'started', 'ended', 'issue_created',
                                  'issue_resolution_date', 'issue_assignee', 'issue_reporter', 'issue_num', 'id',
                                  'issue_resolution', 'issue_status', 'last_change_date', 'issue_contr_count'])
    # Remove comments and utterances count as the words count should be able
    # to reflect those fields, need to verify the assumption
    x = x.drop(columns=['assignee_comments_count', 'others_comments_count', 'reporter_comments_count'])
    x = x.drop(columns=['issue_proj'])

    if add_dummies:
        x = pd.concat([x, pd.get_dummies(x[['issue_type', 'issue_priority']])], axis=1)
    else:
        priority_category = CategoricalDtype(categories=['Blocker', 'High', 'Highest', 'Low', 'Medium'], ordered=False)
        type_category = CategoricalDtype(categories=['Deployment', 'HD Service', 'Ticket'], ordered=False)
        x['issue_type'] = x['issue_type'].astype(type_category)
        x['issue_priority'] = x['issue_priority'].astype(priority_category)

    if drop_categories:
        x = x.drop(columns=['issue_priority', 'issue_type'])

    x = x.fillna(0)
    y = learning_df[['Q1']]

    rename_utterances_class_features(x)
    combine_utterances_class_features(x)
    if utterances_as_percentage:
        calculate_utterances_percentage(x)

    return x, y


def modify_class_labels(class_to_predict, learning_df: DataFrame):
    if class_to_predict is None:
        return
    if type(class_to_predict) == int:
        learning_df.loc[learning_df['Q1'] != class_to_predict, 'Q1'] = 0
        return
    learning_df['Q1_temp'] = learning_df['Q1']
    for c in class_to_predict:
        learning_df.loc[learning_df['Q1'] == c, 'Q1_temp'] = class_to_predict[c]
    learning_df['Q1'] = learning_df['Q1_temp']
    learning_df.drop(columns='Q1_temp', inplace=True)


def drop_utterances_classification_features(df: DataFrame):
    return df.drop(columns=[c for c in df.columns if c.startswith('utr_')])


def split(X, y, train_size: float = 0.7):
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=train_size)
    # print(f'X train size {len(x_train)}, X test size {len(x_test)}')
    y_train.value_counts()
    return x_train, x_test, y_train, y_test
