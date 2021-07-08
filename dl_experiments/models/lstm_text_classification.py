"""

@author Thiago
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from keras import Input, Model, metrics, regularizers
from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Conv2D, MaxPooling2D, concatenate, Bidirectional
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras_self_attention import SeqSelfAttention
from tensorflow.python.keras.layers import Reshape
from attention import Attention

from evaluation.classifier_eval import full_evaluation
from utils.constants import EMBEDDINGS_LEN
from utils.data_utils import split_train_test

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


def lstm_model(input_length, embedding_matrix):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=input_length, weights=[embedding_matrix], trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])

    return model


def bilstm_model(input_length, embedding_matrix):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=input_length, weights=[embedding_matrix], trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Attention(10))
    model.add(Dense(4, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])

    logging.info(model.summary())

    return model


def cnn_lstm_model(input_length, embedding_matrix):
    tf.keras.backend.clear_session()

    sequence_length = input_length
    filter_sizes = [2, 3, 4, 5]
    num_filters = 10
    drop = 0.5

    embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDINGS_LEN, weights=[embedding_matrix], trainable=True)
    inputs = Input(shape=(sequence_length,))
    embedding = embedding_layer(inputs)
    reshape = Reshape((sequence_length, EMBEDDINGS_LEN, 1))(embedding)

    convs = []
    maxpools = []

    for filter_size in filter_sizes:
        conv = Conv2D(num_filters, (filter_size, EMBEDDINGS_LEN), activation='relu',
                      kernel_regularizer=regularizers.l2(0.01))(reshape)

        maxpool = MaxPooling2D(
            (sequence_length - filter_size + 1, 1), strides=(1, 1))(conv)

        maxpools.append(maxpool)
        convs.append(conv)

    merged_tensor = concatenate(maxpools, axis=1)

    # flatten = Flatten()(merged_tensor)
    # reshape = Reshape((3 * num_filters,))(flatten)
    # dropout = Dropout(drop)(flatten)
    lstm = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(merged_tensor)
    # conc = Dense(40)(dropout)
    output = Dense(4, activation='sigmoid',
                   kernel_regularizer=regularizers.l2(0.01))(lstm)

    # this creates a model that includes
    cnn_model = Model(inputs, output)
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])

    return cnn_model
