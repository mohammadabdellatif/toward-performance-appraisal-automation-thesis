import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from pandas import DataFrame
from textblob import Word

from ds_extractor.comments import RegexReplacement, SentenceReplacement
from sys_utils.systemlogger import SystemLogger, progress, new_line

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words("english")
# print(stop_words)
stop_words.remove('no')
stop_words.remove('once')
stop_words.remove('y')

porter_stemmer = EnglishStemmer()
punctuation = string.punctuation.replace('?', '').replace('#', '').replace('_', '')
logger = SystemLogger.logger("comments_pre_processor")


class Counter:

    def __init__(self, count: int) -> None:
        self.end = count
        self.start = 0

    def increment(self):
        self.start += 1
        return self.start

    def __str__(self) -> str:
        return f'{str(self.start)} out of {str(self.end)}'


common_words_replacers = [
    # common words replacements
    RegexReplacement("communication protocol",
                     # br = Best Regards, kr = Kind Regards :(
                     r"(\b)(((best\s+|kind\s+)?)regards|thank(s?)|dear(s?)|greeting(s?)|hi|hello|br|kr|kindly|please)(\b)",
                     r'\1ph_com_protocol\8'),
    RegexReplacement("unify SQL", r"ph_[\w_]*sql", 'ph_sql'),
    RegexReplacement("stack trace as logs", r"ph_[\w_]*trace", 'ph_logs'),
    # RegexReplacement("Products",
    #                  r"(\b)(ecc|rdc|echeque|cheque|pps|payhub|cliq|pdc|corpay|qatch|ebpp|mpay|pssig|wps|bankpay|mmsp|mmsc|aml|ach|)(\b)",
    #                  r'\1ph_a_product\3'),
    # RegexReplacement("Numbers", r"(\b)\d+(.\d+)?(\b)", r'\1ph_number\3'),
    RegexReplacement("Separate ? from end of words", r"(\w)\?", r'\1 ?'),
    SentenceReplacement("Replace single Y with Yes", r"y(\.)?", 'yes'),
    SentenceReplacement("Replace single N with no", r"n(\.)?", 'no')
]


class CommentsPreProcessor:

    def preprocess(self, utterances_df: DataFrame, issues_snapshots_df: DataFrame):
        self.__convert_actionbody_to_str(utterances_df)
        logger.info("text cleanup")
        utterances_df = self.__impute_author_role(utterances_df, issues_snapshots_df)
        counter = Counter(len(utterances_df))
        utterances_df = utterances_df.apply(lambda x: self.__pre_process(x, counter), axis=1)
        logger.info("done")
        return utterances_df

    def __pre_process(self, row, counter):
        row['pp_actionbody'] = self.__text_cleanup(row['actionbody'], counter)
        row['words_count'] = self.__count_words(row['actionbody'])
        row['pp_words_count'] = self.__count_words(row['pp_actionbody'])
        # row['pp_actionbody'] = 'ar_' + row['author_role'] + ' ' + row['pp_actionbody']
        return row

    def __convert_actionbody_to_str(self, utterances_df):
        logger.info("convert actionbody to string")
        utterances_df['actionbody'] = utterances_df['actionbody'].astype(str)

    def __count_utterances_words(self, utterances_df):
        logger.info("count utterances words")
        utterances_df['words_count'] = utterances_df['actionbody'].apply(self.__count_words)

    def __count_pre_processed_utterances_words(self, utterances_df):
        logger.info("count preprocessed utterances words")
        utterances_df['pp_words_count'] = utterances_df['pp_actionbody'].apply(self.__count_words)

    def __count_words(self, x: str):
        return len(x.split()) if pd.notna(x) else 0

    def __text_cleanup(self, action_body: str, counter):
        counter.increment()
        progress(f"{counter}")
        if pd.isna([action_body])[0] or action_body.strip() == '':
            return ''
        action_body = self.__remove_punctuation(action_body)
        action_body = self.__stop_words_removal(action_body)

        action_body = self.__common_words_replacement(action_body)

        action_body = self.__lemmatization(action_body)
        action_body = self.__stemming(action_body)
        return action_body

    def pre_process_text(self, text):
        return self.__text_cleanup(text, Counter(1))

    def __remove_punctuation(self, x):
        return "".join(i for i in x if i not in punctuation)

    def __stop_words_removal(self, x):
        return " ".join(i for i in x.split() if i not in stop_words)

    def __stemming(self, x):
        return " ".join([porter_stemmer.stem(word) for word in x.split()])

    def __lemmatization(self, x):
        return " ".join([Word(word).lemmatize() for word in x.split()])

    def __impute_author_role(self, utterances_df: DataFrame, issues_snapshots_df: DataFrame):
        logger.info("impute author role")
        logger.info(f"utterances {len(utterances_df)}")
        utterances_df = utterances_df[utterances_df['issueid'].isin(issues_snapshots_df['id'].drop_duplicates())].copy()
        logger.info(f'filtered utterances {len(utterances_df)}')
        utterances_df['author_role'] = 'others'
        counter = Counter(len(issues_snapshots_df))
        for idx, row in issues_snapshots_df.iterrows():
            assignee = row['issue_assignee']
            reporter = row['issue_reporter']
            base_filter = self.__utterances_filter(row, utterances_df)

            utterances_df.loc[base_filter, 'author_role'] = 'others'
            # it is important to set the reporter before the assignee, since there are some issues reported
            # and assigned by the SID himself
            utterances_df.loc[base_filter & (utterances_df['author'] == reporter), 'author_role'] = 'reporter'
            utterances_df.loc[(base_filter & (utterances_df['author'] == assignee)), 'author_role'] = 'assignee'
            counter.increment()
            progress(f'{counter}')
        new_line()
        return utterances_df

    def __utterances_filter(self, row, utterances_df: DataFrame):
        return ((utterances_df['issueid'] == row['id'])
                & (utterances_df['created'] >= row['started'])
                & (utterances_df['created'] <= row['ended']))

    def __common_words_replacement(self, action_body):
        for rep in common_words_replacers:
            action_body = rep.process(action_body)
        return action_body


class UtterancesMerger:

    def merge_by_comment(self, utterances_df: DataFrame):
        """
        Merges comments utterances, setting back the comment as a one single block of text

        :param utterances_df:  dataset of utterances
        :return: comments dataset
        """
        utterances_df = utterances_df.set_index(['issueid', 'id', 'utr_seq'])
        utterances_df['actionbody'] = utterances_df['actionbody'].astype(str)
        utterances_df['pp_actionbody'] = utterances_df['pp_actionbody'].astype(str)

        comment_record = self.__new_comment_record(utterances_df.index[0], utterances_df.iloc[0])
        logger.info(f"combine {len(utterances_df)} utterances by issueid and id")
        count = 1
        records = []
        for idx, row in utterances_df.iterrows():
            if self.__is_new_comment(comment_record, idx):
                records.append(comment_record)
                comment_record = self.__new_comment_record(idx, row)

            self.__merge_utterance_to_comment(comment_record, row)
            progress(f'processed {count} out of {len(utterances_df)}')
            count += 1

        comments_df = self.__prepare_comments_df(utterances_df, records)
        return comments_df

    def __is_new_comment(self, comment_record, idx):
        return comment_record['id'] != idx[1]

    def __merge_utterance_to_comment(self, comment, utterance):
        comment['actionbody'] = ' '.join([comment['actionbody'], utterance['actionbody']])
        comment['words_count'] += utterance['words_count']

        if pd.notna(utterance['pp_actionbody']) or utterance['pp_actionbody'].strip() != '':
            comment['pp_actionbody'] = ' '.join([comment['pp_actionbody'], utterance['pp_actionbody']])
            comment['pp_words_count'] += utterance['pp_words_count']

    def __record_as_df(self, comments_df, record):
        return pd.DataFrame(index=[(record['issueid'], record['id'])], columns=comments_df.columns, data=record)

    def __prepare_comments_df(self, utterances_df, records):
        comments_df = pd.DataFrame(columns=[c for c in ['issueid', 'id'] + utterances_df.columns.to_list() if
                                            c != 'utr_seq'],
                                   data=records)
        comments_df.set_index(['issueid', 'id'], inplace=True)
        return comments_df

    def __new_comment_record(self, idx, row):
        return {
            'id': idx[1],
            'issueid': idx[0],
            'created': row['created'],
            'author': row['author'],
            'author_role': row['author_role'],
            'is_private': row['is_private'],
            'actionbody': '',
            'pp_actionbody': '',
            'pp_words_count': 0,
            'words_count': 0
        }


if __name__ == '__main__':
    cp = CommentsPreProcessor()
    print(cp.pre_process_text('helpdesk team.'))
    print('done')
