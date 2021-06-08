"""

@author Thiago
"""
from gensim.models import KeyedVectors
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from evaluation.classifier_eval import full_evaluation
from utils.data_utils import split_train_test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAX_NB_WORDS = 12000
MAX_LEN_SEQ = 400
EMBEDDING_DIM = 100


def lstm_training(train_data, test_data, path_emb):
    print("LSTM Training")

    x_train, y_train, x_test, y_test = split_train_test(train_data, test_data)

    tokenizer = Tokenizer(
        num_words=MAX_NB_WORDS,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
        lower=True
    )

    tokenizer.fit_on_texts(x_train)
    word_index = tokenizer.word_index
    print('\tFound %s unique tokens.' % len(word_index))

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN_SEQ)
    print('\tShape of data tensor:', x_train.shape)

    y_train = pd.get_dummies(y_train).values

    word_vectors = KeyedVectors.load_word2vec_format(path_emb, binary=False)

    vocabulary_size = min(len(word_index) + 1, MAX_NB_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

    vec = np.random.rand(EMBEDDING_DIM)
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            # vec = np.random.rand(EMBEDDING_DIM)
            embedding_matrix[i] = vec

    model = Sequential()
    model.add(
        Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1], weights=[embedding_matrix],
                  trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 100
    batch_size = 30

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3)
    plt.suptitle('Optimizer : Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.plot(history.history['loss'], color='b', label='Training Loss')
    plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.show()

    print("\tEvaluating....")
    sequences_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(sequences_test, maxlen=MAX_LEN_SEQ)
    y_pred = model.predict(x_test)
    y_pred = [np.argmax(y) for y in y_pred]
    full_evaluation(y_test, y_pred)
