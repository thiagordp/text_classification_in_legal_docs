"""

@author Thiago
"""

from gensim.models import Word2Vec

from utils.constants import *
from utils.embeddings_utils import visualize_embeddings


def generate_word2vec_skipgram(data):
    print("Training Word Embeddings w/ Word2Vec Skipgram")
    model = Word2Vec(
        sentences=data,
        size=EMBEDDINGS_LEN,
        workers=4,
        sg=1,
        min_count=2,
        iter=EMBEDDINGS_ITER)

    words = list(model.wv.vocab)
    model_name = EMBEDDINGS_PATH + "model_word2vec_skipgram_" + EMBEDDINGS_LEN + ".txt"
    model.wv.save_word2vec_format(model_name, binary=False)

    visualize_embeddings(model)

    result = model.most_similar("procedente", topn=20)
    print(result)


def generate_word2vec_cbow(data):
    print("Training Word Embeddings w/ Word2Vec C-BOW")
    model = Word2Vec(
        sentences=data,
        size=EMBEDDINGS_LEN,
        workers=4,
        sg=0,
        min_count=2,
        iter=EMBEDDINGS_ITER)

    model_name = EMBEDDINGS_PATH + "model_word2vec_cbow_" + EMBEDDINGS_LEN + ".txt"
    model.wv.save_word2vec_format(model_name, binary=False)

    visualize_embeddings(model)

    result = model.most_similar("procedente", topn=20)
    print(result)
