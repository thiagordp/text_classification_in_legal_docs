"""

@author Thiago
"""

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def visualize_embeddings(model):
    """
    Visualize Embeddings vocabulary in two dimensions with PCA.

    :param model: Trained Model
    :return: None
    """

    x = model[model.wv.vocab]

    pca = PCA(n_components=2)
    result = pca.fit_transform(x)

    plt.figure(figsize=(30, 30))
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)

    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.show()
