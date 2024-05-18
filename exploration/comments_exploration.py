import gensim
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
from gensim import corpora
from gensim.models import CoherenceModel
from matplotlib.axes import Axes
from pandas import DataFrame
from wordcloud import WordCloud

from sys_utils.systemlogger import progress


def tokenize_sentence(sentence: str) -> list[str]:
    return sentence.split()


def cloud(utterances_df: DataFrame,
          ax: Axes,
          max_words: int = None,
          width: int = 2000,
          height: int = 2000):
    all_words = []
    unique_words = set()
    for s in utterances_df['pp_actionbody']:
        if pd.isna(s):
            continue
        word = str(s).strip()
        if word == '':
            continue
        all_words.append(word)
        unique_words.add(word)

    wordcloud: WordCloud = WordCloud(width=width,
                                     height=height,
                                     collocations=False,
                                     background_color='black',
                                     normalize_plurals=False,
                                     colormap=plt.colormaps['tab20c'],
                                     max_words=max_words).generate(" ".join(all_words))
    ax.imshow(wordcloud, interpolation='bilinear')
    return len(unique_words), len(all_words)


class LDAModel:

    def __init__(self, id2word: corpora.Dictionary,
                 corpus: list,
                 lda_model: gensim.models.LdaMulticore,
                 coherence: float,
                 lda_visualize) -> None:
        super().__init__()
        self.__coherence = coherence
        self.__id2word = id2word
        self.__corpus = corpus
        self.__lda_model = lda_model
        self.lda_visualize = lda_visualize

    def document_topics(self, sentence: str):
        tokens = tokenize_sentence(sentence)
        bow = self.__id2word.doc2bow(tokens)
        return self.__lda_model.get_document_topics(bow)

    def coherence(self):
        return self.__coherence

    def topics_count(self):
        return len(self.__lda_model.get_topics())

    def topics(self):
        return self.__lda_model.get_topics()

    def show_topics(self):
        return self.__lda_model.show_topics()


def utterances_lda(utterances_d: DataFrame, num_topics: int, random_state: int = 42) -> LDAModel:
    sentences = utterances_as_text(utterances_d)
    return text_lda(sentences, num_topics, random_state)


def text_lda(sentences: list[str], num_topics: int, random_state: int = 42):
    # Create Dictionary
    corpus, id2word = sentences_as_corpus(sentences)
    return lda(corpus, id2word, num_topics, random_state, sentences)


def lda(corpus, id2word, num_topics, random_state, sentences):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=random_state)
    lda_vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=sentences, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return LDAModel(id2word, corpus, lda_model, coherence_lda, lda_vis)


def sentences_as_corpus(sentences):
    id2word = corpora.Dictionary(sentences)
    corpus = [id2word.doc2bow(text) for text in sentences]
    return corpus, id2word


def utterances_as_text(utterances_d: DataFrame):
    utterances_d = utterances_d[['author_role', 'pp_actionbody']]
    utterances_d = utterances_d.drop_duplicates()
    sentences = []
    for s in utterances_d['pp_actionbody']:
        if pd.isna(s) or str(s).strip() == '':
            continue
        sentences.append(tokenize_sentence(str(s)))
    return sentences


def classify(utterances_df, out_file: str):
    c = 0
    for i, row in utterances_df.iterrows():
        c += 1
        progress(f'process {c} out of {len(utterances_df)}')
        sent = row['pp_actionbody']
        topics = lda_model.document_topics(sent)
        h_topic_v = 0
        for t in topics:
            if h_topic_v < t[1]:
                h_topic_v = t[1]
                utterances_df.loc[i, 'h_topic'] = t[0]
            utterances_df.loc[i, f'topic_{t[0]}'] = t[1]
    utterances_df.to_csv(f'../temp_data/{out_file}.csv')


if __name__ == '__main__':
    file = 'pp_utterances'
    utterances_df = pd.read_csv(f'../temp_data/{file}.csv')
    print(len(utterances_df))
    utterances_df = utterances_df[~pd.isna(utterances_df['pp_actionbody'])]
    utterances_df['pp_actionbody~'] = utterances_df['pp_actionbody'].astype(str)
    # fig: Figure = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # print(cloud(utterances_df, ax))
    # fig.show()
    utterances_df = utterances_df[~((utterances_df['author'].isin(['admin','u003']))
                        & (utterances_df['comment_seq'] < 3)
                        & (utterances_df['author_role'] == 'others'))]
    print('start lda modeling')
    lda_model = utterances_lda(utterances_df, num_topics=8, random_state=42)
    topics_count = lda_model.topics_count()
    print(f'model completed {topics_count}')
    for t in range(topics_count):
        utterances_df[f'topic_{t}'] = 0
    utterances_df['h_topic'] = 0

    classify(utterances_df, out_file=f'{file}_lda')
    print('done')
