from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from sys_utils import systemlogger
from sys_utils.systemlogger import SystemLogger, progress

logger: Logger = SystemLogger.logger('ds_merger')


class CommentsStatisticsAggregator:
    """
    For each issue snapshot, this class will calculate the number of comments, utterances, and words (terms)
    for each role and adds them as a feature to the final dataset.
    """

    class __ByAuthorAggregation:

        def __init__(self, features: list[str], aggregator: object) -> None:
            self.aggregator = aggregator
            self.features = features

        def aggregate(self, idx, row, snapshots_df: DataFrame, utterances_df: DataFrame):
            aggregated_utterances = self.aggregator(utterances_df)
            self.__fill_counts(idx, row, snapshots_df, aggregated_utterances,
                               self.features[0], self.features[1], self.features[2])

        def __fill_counts(self, idx, row,
                          snapshots_df: DataFrame,
                          by_author_df: DataFrame,
                          f_assign: str,
                          f_reporter: str,
                          f_others: str):
            assignee = row['issue_assignee']
            reporter = row['issue_reporter']
            for auth, v in by_author_df.iterrows():
                _count = v['count']
                if type(auth) == tuple:
                    auth = auth[0]
                if auth == assignee:
                    snapshots_df.loc[idx, f_assign] = snapshots_df.loc[idx, f_assign] + _count
                    continue
                if auth == reporter:
                    snapshots_df.loc[idx, f_reporter] = snapshots_df.loc[idx, f_reporter] + _count
                    continue
                snapshots_df.loc[idx, f_others] = snapshots_df.loc[idx, f_others] + _count

    def __init__(self) -> None:
        self.aggregators = [
            CommentsStatisticsAggregator.__ByAuthorAggregation(
                ['assignee_comments_count', 'reporter_comments_count', 'others_comments_count'],
                self.__aggregate_comments),
            CommentsStatisticsAggregator.__ByAuthorAggregation(
                ['assignee_utterances_count', 'reporter_utterances_count', 'others_utterances_count'],
                self.__aggregate_utterances),
            CommentsStatisticsAggregator.__ByAuthorAggregation(['assignee_terms_count', 'reporter_terms_count',
                                                                'others_terms_count'],
                                                               self.__aggregate_terms)]

    def merge(self, issues_snapshots_df: DataFrame, utterances_df: DataFrame) -> DataFrame:
        logger.info("create default fields")
        issues_snapshots_df = self.__fill_features_init_values(issues_snapshots_df)
        c = 1
        for i, row in issues_snapshots_df.iterrows():
            issue_utterances_df = utterances_df[self.__utterances_filter(row, utterances_df)]
            systemlogger.progress(f'processed {c} of {len(issues_snapshots_df)}')
            self.__merge_dfs(i, row, issues_snapshots_df, issue_utterances_df)
            c += 1
        systemlogger.new_line()
        logger.info("done")
        return issues_snapshots_df

    def __utterances_filter(self, row, utterances_df: DataFrame):
        return ((utterances_df['issueid'] == row['id'])
                & (utterances_df['created'] >= row['started'])
                & (utterances_df['created'] < row['ended'])
                & (utterances_df['pp_words_count'] > 0))

    def __fill_features_init_values(self, issues_snapshots_df):
        issues_snapshots_df = issues_snapshots_df.copy()

        for a in self.aggregators:
            for f in a.features:
                issues_snapshots_df[f] = 0

        return issues_snapshots_df

    def __merge_dfs(self, idx, row, snapshots_df: DataFrame, utterances_df: DataFrame):
        for a in self.aggregators:
            a.aggregate(idx, row, snapshots_df, utterances_df)

    def __aggregate_comments(self, utterances_df):
        by_author_df = utterances_df[['author', 'id']].drop_duplicates().groupby('author').count()
        by_author_df.rename(columns={'id': 'count'}, inplace=True)
        return by_author_df

    def __aggregate_utterances(self, utterances_df):
        by_author_df = utterances_df[['author', 'utr_seq']].groupby('author').count()
        by_author_df.rename(columns={'utr_seq': 'count'}, inplace=True)
        return by_author_df

    def __aggregate_terms(self, utterances_df):
        by_author_df = utterances_df[['author', 'pp_words_count']].groupby('author').sum()
        by_author_df.rename(columns={'pp_words_count': 'count'}, inplace=True)
        return by_author_df


class SnapshotsUtterancesMerger:
    """
    Combines issues snapshots utterances by author role and adds them as columns to the dataset
    """

    def combine_snapshot_comments_by_role(self, issues_snapshots_df: DataFrame, utterances_df: DataFrame):
        issues_snapshots_df = self.__clone_snapshot_df(issues_snapshots_df)
        utterances_df = self.__remove_automated_comments(utterances_df)

        utterances_df = utterances_df.reset_index().sort_values(by=['issueid', 'id', 'utr_seq'])
        logger.info(f"combine {len(issues_snapshots_df)} issues snapshots")
        count = 1
        for idx, row in issues_snapshots_df.iterrows():
            snapshot_utterances = self.__filter_snapshot_utterances(row, utterances_df)
            self.__combine_snapshot_utterances(idx, issues_snapshots_df, snapshot_utterances)
            progress(f'processed {count} of {len(issues_snapshots_df)}')
            count += 1

        return issues_snapshots_df

    def __combine_snapshot_utterances(self, idx, issues_snapshots_df, snapshot_utterances):
        comments = {
            'assignee': '',
            'reporter': '',
            'others': ''
        }
        for u_idx, u_row in snapshot_utterances.iterrows():
            comments[u_row['author_role']] = comments[u_row['author_role']] + ' ' + u_row['pp_actionbody']
        issues_snapshots_df.loc[idx, 'assignee_comments'] = comments['assignee']
        issues_snapshots_df.loc[idx, 'reporter_comments'] = comments['reporter']
        issues_snapshots_df.loc[idx, 'others_comments'] = comments['others']

    def __filter_snapshot_utterances(self, row, utterances_df):
        snapshot_utterances = utterances_df[(utterances_df['issueid'] == row['id'])]
        snapshot_utterances = snapshot_utterances[
            snapshot_utterances['created'].between(row['started'], row['ended'], inclusive='left')]
        return snapshot_utterances

    def __clone_snapshot_df(self, issues_snapshots_df):
        issues_snapshots_df = issues_snapshots_df.copy()
        issues_snapshots_df['reporter_comments'] = ''
        issues_snapshots_df['assignee_comments'] = ''
        issues_snapshots_df['others_comments'] = ''
        return issues_snapshots_df

    def __remove_automated_comments(self, utterances_df):
        utterances_df = utterances_df[~((utterances_df['author'].isin(['admin', 'u003']))
                                        & (utterances_df['comment_seq'] < 3)
                                        & (utterances_df['author_role'] == 'others'))]
        utterances_df = utterances_df[~pd.isna(utterances_df['pp_actionbody'])]
        return utterances_df


class IssuesSnapshotsTfidf:

    def __init__(self,
                 max_df: float = 1.0,
                 min_df: float = 0.001,
                 use_idf: bool = False):
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf

    def vectorize_issues_comments(self, issues_snapshots_df: DataFrame):
        issues_snapshots_df = issues_snapshots_df.copy()
        issues_snapshots_df.reset_index(inplace=True)
        for idx, row in issues_snapshots_df.iterrows():
            if pd.isna(row['assignee_comments']):
                issues_snapshots_df.loc[idx, 'assignee_comments'] = 'ph_empty_comment'
            if pd.isna(row['reporter_comments']):
                issues_snapshots_df.loc[idx, 'reporter_comments'] = 'ph_empty_comment'
            if pd.isna(row['others_comments']):
                issues_snapshots_df.loc[idx, 'others_comments'] = 'ph_empty_comment'
        logger.info("fit TF IDF for each assignee comments")
        assignee_tfidf = self.__tfidf('assignee_comments', issues_snapshots_df)

        logger.info("fit TF IDF for each reporter comments")
        reporter_tfidf = self.__tfidf('reporter_comments', issues_snapshots_df)

        logger.info("fit TF IDF for each other comments")
        other_tfidf = self.__tfidf('others_comments', issues_snapshots_df)

        issues_snapshots_df = self.__append_tfidf_to_dataframe('ac_', assignee_tfidf, issues_snapshots_df)
        issues_snapshots_df = self.__append_tfidf_to_dataframe('rc_', reporter_tfidf, issues_snapshots_df)
        issues_snapshots_df = self.__append_tfidf_to_dataframe('oc_', other_tfidf, issues_snapshots_df)

        issues_snapshots_df = issues_snapshots_df.drop(
            columns=['assignee_comments', 'others_comments', 'reporter_comments'])
        issues_snapshots_df.set_index('idx', inplace=True, verify_integrity=True)

        return issues_snapshots_df

    def __append_tfidf_to_dataframe(self, prefix, tfidf, issues_snapshots_df: DataFrame):
        vocabulary = tfidf[0]
        df_columns = [''] * len(vocabulary)
        for c, idx in vocabulary.items():
            df_columns[idx] = prefix + c

        tfidf_values = tfidf[1].toarray()
        indexes = [i for i in range(0, len(tfidf_values))]
        tfidf_df = pd.DataFrame(data=tfidf_values, columns=df_columns, index=indexes)
        return pd.concat([issues_snapshots_df, tfidf_df], axis=1)

    def __tfidf(self, comments_col, issues_snapshots_df) -> TfidfVectorizer:
        tfidf_vectorizer = TfidfVectorizer(
            max_df=self.max_df,
            min_df=self.min_df,
            use_idf=self.use_idf,
            max_features=None,
            stop_words=None,
            token_pattern=r'(?u)(\b\w\w+\b|#\b\w\w+\b#)')
        # TF-IDF feature matrix
        fitted = tfidf_vectorizer.fit_transform(issues_snapshots_df[comments_col])
        return tfidf_vectorizer.vocabulary_, fitted


class UtterancesClassifier:

    def __init__(self,
                 assignee_clusters: int,
                 reporter_clusters: int,
                 others_clusters: int,
                 vector_size: int = 100):
        self.__assignee_clusters = assignee_clusters
        self.__reporter_clusters = reporter_clusters
        self.__others_clusters = others_clusters
        self.__vector_size = vector_size

    def classify_utterances(self, utterances_df: DataFrame):
        logger.info("classify utterances")
        au_df = self.__cluster_then_classify(utterances_df, 'assignee', self.__assignee_clusters)
        ru_df = self.__cluster_then_classify(utterances_df, 'reporter', self.__reporter_clusters)
        ou_df = self.__cluster_then_classify(utterances_df, 'others', self.__others_clusters)

        utterances_df['label'] = ''
        logger.info("merge with labels")
        self.__update_utterances_labels(au_df, utterances_df)
        self.__update_utterances_labels(ru_df, utterances_df)
        self.__update_utterances_labels(ou_df, utterances_df)

        return utterances_df

    def __vectorize(self, sentence, w2v_model):
        words = sentence.split()
        words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if len(words_vecs) == 0:
            return np.zeros(self.__vector_size)
        words_vecs = np.array(words_vecs)
        return words_vecs.mean(axis=0)

    def __update_utterances_labels(self, au_df, utterances_df):
        for idx, u in au_df.iterrows():
            utterances_df.loc[idx, 'label'] = str(u['label'])

    def __cluster_then_classify(self, utterances_df, author_role, clusters):
        others_utterances = self.__filter_utterances_by_author_role(author_role, utterances_df)
        text, w2v_model = self.__build_word2vec_model(others_utterances)
        text_as_w2v = self.__build_text_word2vec_model(text, w2v_model)
        self.__cluster_text(clusters, text_as_w2v, others_utterances)
        self.__save_classified_as_dataset(author_role, others_utterances)
        return others_utterances

    def __filter_utterances_by_author_role(self, author_role, utterances_df):
        u_df = utterances_df[utterances_df['author_role'] == author_role].copy()
        return u_df

    def __cluster_text(self, clusters, df, u_df):
        km = KMeans(n_clusters=clusters, n_init='auto', random_state=42)
        km.fit(df)
        u_df['label'] = km.labels_

    def __build_text_word2vec_model(self, text, w2v_model):
        f_names = [f'f{i}' for i in range(0, self.__vector_size)]
        rows = []
        for t in text:
            rows.append(self.__vectorize(str(t), w2v_model))
        df = pd.DataFrame(columns=f_names, data=rows)
        return df

    def __build_word2vec_model(self, u_df):
        text = u_df['pp_actionbody']
        sentences = [str(sentence).split() for sentence in text]
        logger.info('train model')
        w2v_model = Word2Vec(sentences, vector_size=self.__vector_size, window=5, min_count=5, workers=1, seed=42)
        return text, w2v_model

    def __save_classified_as_dataset(self, author_role, u_df):
        for label in u_df['label'].drop_duplicates():
            l_df = u_df[u_df['label'] == label]
            Path(f'../temp_data/utterances-{author_role}').mkdir(exist_ok=True, parents=True)
            l_df.to_csv(f'../temp_data/utterances-{author_role}/utterances_{label}.csv')


class TFIDFUtterancesClassifier:

    def __init__(self,
                 assignee_clusters: int,
                 reporter_clusters: int,
                 others_clusters: int):
        self.__assignee_clusters = assignee_clusters
        self.__reporter_clusters = reporter_clusters
        self.__others_clusters = others_clusters

    def classify_utterances(self, utterances_df: DataFrame):
        logger.info("classify utterances")
        utterances_df = self.__remove_automated_comments(utterances_df)
        au_df = self.__cluster_then_classify(utterances_df, 'assignee', self.__assignee_clusters)
        ru_df = self.__cluster_then_classify(utterances_df, 'reporter', self.__reporter_clusters)
        ou_df = self.__cluster_then_classify(utterances_df, 'others', self.__others_clusters)

        utterances_df['label'] = ''
        logger.info("merge with labels")
        self.__update_utterances_labels(au_df, utterances_df)
        self.__update_utterances_labels(ru_df, utterances_df)
        self.__update_utterances_labels(ou_df, utterances_df)

        return utterances_df

    def __remove_automated_comments(self, utterances_df):
        utterances_df = utterances_df[~((utterances_df['author'].isin(['admin', 'u003']))
                                        & (utterances_df['comment_seq'] < 3)
                                        & (utterances_df['author_role'] == 'others'))]
        utterances_df = utterances_df[~pd.isna(utterances_df['pp_actionbody'])]
        return utterances_df

    def __tfidf(self, utterances_df: DataFrame) -> TfidfVectorizer:
        tfidf_vectorizer = TfidfVectorizer(
            max_df=1.0,
            min_df=0.001,
            max_features=None,
            stop_words=None,
            token_pattern=r'(?u)(\b\w\w+\b|#\b\w\w+\b#)')
        # TF-IDF feature matrix
        fitted = tfidf_vectorizer.fit_transform(utterances_df['pp_actionbody'])
        return tfidf_vectorizer.vocabulary_, fitted

    def __update_utterances_labels(self, au_df, utterances_df):
        for idx, u in au_df.iterrows():
            utterances_df.loc[idx, 'label'] = str(u['label'])

    def __cluster_then_classify(self, utterances_df, author_role, clusters):
        others_utterances = self.__filter_utterances_by_author_role(author_role, utterances_df)
        vocab, tf_idf = self.__tfidf(others_utterances)
        text_as_w2v = self.__build_text_by_model(utterances_df['pp_actionbody'], vocab, tf_idf)
        self.__cluster_text(clusters, text_as_w2v, others_utterances)
        self.__save_classified_as_dataset(author_role, others_utterances)
        return others_utterances

    def __filter_utterances_by_author_role(self, author_role, utterances_df):
        u_df = utterances_df[utterances_df['author_role'] == author_role].copy()
        return u_df

    def __cluster_text(self, clusters, df, u_df):
        km = KMeans(n_clusters=clusters, n_init='auto', random_state=42)
        km.fit(df)
        u_df['label'] = km.labels_

    def __build_text_by_model(self, text, vocab, tf_idf):
        df_columns = [''] * len(vocab)
        for c, idx in vocab.items():
            df_columns[idx] = c
        return pd.DataFrame(data=tf_idf.toarray(), columns=df_columns)

    def __save_classified_as_dataset(self, author_role, u_df):
        for label in u_df['label'].drop_duplicates():
            l_df = u_df[u_df['label'] == label]
            Path(f'../temp_data/utterances-{author_role}').mkdir(exist_ok=True, parents=True)
            l_df.to_csv(f'../temp_data/utterances-{author_role}/utterances_{label}.csv')


class IssuesUtterancesClassesAggregator:

    def merge_with_utterances_classes(self, issues_df: DataFrame, utterances_df: DataFrame) -> DataFrame:
        logger.info("merge issues with utterances labels")
        self.__append_utterances_labels_features('utr_assignee_', 'assignee', issues_df, utterances_df)
        self.__append_utterances_labels_features('utr_reporter_', 'reporter', issues_df, utterances_df)
        self.__append_utterances_labels_features('utr_others_', 'others', issues_df, utterances_df)

        count = 1
        for idx, issue in issues_df.iterrows():
            progress(f'process {count} from {len(issues_df)}')
            issue_utterances = utterances_df[(utterances_df['issueid'] == issue['id'])
                                             & (utterances_df['created']
                                                .between(issue['started'], issue['ended'], inclusive='left'))]
            for i, utr in issue_utterances.iterrows():
                f = f"utr_{utr['author_role']}_{utr['label']}"
                issues_df.loc[idx, f] = 1 + issues_df.loc[idx, f]
            count = count + 1
        return issues_df

    def __append_utterances_labels_features(self, prefix, author_role, issues_df, utterances_df):
        utr_classes = sorted(utterances_df[utterances_df['author_role'] == author_role]['label'].drop_duplicates())
        features = [f'{prefix}{l}' for l in utr_classes]
        for f in features:
            issues_df[f] = 0
