import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def vectorize(sentence, w2v_model):
    words = sentence.split()
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(100)
    words_vecs = np.array(words_vecs)
    print(words_vecs)
    return words_vecs.mean(axis=0)


if __name__ == "__main__":
    df = pd.read_csv('../temp_data/pp_utterances.csv')
    text = df['pp_actionbody'][:100]

    sentences = [str(sentence).split() for sentence in text]
    print(max([len(s) for s in sentences]))
    print('train model')
    w2v_model = Word2Vec(sentences, vector_size=10, window=5, min_count=5, workers=4)
    print('done')
    print(text[4])
    vect = vectorize(text[4], w2v_model)
    print(len(vect))
    print(vect)
