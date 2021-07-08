"""

@author Thiago
"""
from gensim.models import KeyedVectors
from keras import Input, Model, metrics, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Reshape, Conv2D, MaxPooling2D, concatenate, Flatten, Dropout, Dense
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from evaluation.classifier_eval import full_evaluation
from models.lstm_text_classification import MAX_NB_WORDS
from utils.constants import EMBEDDINGS_LEN
from utils.data_utils import split_train_test

NUM_WORDS = 20000


def cnn_training(train_data, test_data, embeddings_path):
    print("=========  CNN TRAINING  ==========")

    x_train, y_train, x_test, y_test = split_train_test(train_data, test_data)

    tokenizer = Tokenizer(
        num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
    tokenizer.fit_on_texts(x_train)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Convert train and val to sequence
    sequences_train = tokenizer.texts_to_sequences(x_train)

    x_train = pad_sequences(sequences_train, maxlen=100)
    y_train = np.asarray(y_train)
    y_train = to_categorical(y_train)

    print('Shape of X train and X validation tensor:', x_train.shape)
    print('Shape of label train and validation tensor:', y_train.shape)

    word_vectors = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)

    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDINGS_LEN))

    vec = np.random.rand(EMBEDDINGS_LEN)
    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            vec = np.random.rand(EMBEDDINGS_LEN)
            embedding_matrix[i] = vec

    # physical_devices = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Define Embedding function using the embedding_matrix
    embedding_layer = Embedding(vocabulary_size, EMBEDDINGS_LEN, weights=[embedding_matrix], trainable=True)

    sequence_length = x_train.shape[1]
    filter_sizes = [2, 3, 4, 5]
    num_filters = 10
    drop = 0.5

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

    flatten = Flatten()(merged_tensor)
    # reshape = Reshape((3 * num_filters,))(flatten)
    dropout = Dropout(drop)(flatten)
    conc = Dense(40)(dropout)
    output = Dense(4, activation='sigmoid',
                   kernel_regularizer=regularizers.l2(0.01))(conc)

    # this creates a model that includes
    model = Model(inputs, output)

    opt = SGD(lr=1e-3)
    #opt = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=[metrics.binary_crossentropy, metrics.mae, metrics.categorical_accuracy])

    # Fitting Model to the data
    callbacks = [EarlyStopping(monitor='val_loss')]

    print("Training...")
    hist_adam = model.fit(x_train, y_train, batch_size=30, epochs=100, verbose=0, validation_split=0.3)

    # plotting Loss
    plt.suptitle('Optimizer : Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.plot(hist_adam.history['loss'], color='b', label='Training Loss')
    plt.plot(hist_adam.history['val_loss'], color='r', label='Validation Loss')
    plt.legend(loc='upper right')
    # plt.show()

    print("Evaluating....")
    sequences_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(sequences_test, maxlen=x_train.shape[1])
    y_pred = model.predict(x_test)
    y_pred = [np.argmax(y) for y in y_pred]
    full_evaluation(y_test, y_pred)


def cnn_model(input_length, embedding_matrix):
    tf.keras.backend.clear_session()
    embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDINGS_LEN, weights=[embedding_matrix], trainable=True)

    sequence_length = input_length
    filter_sizes = [2, 3, 4, 5]
    num_filters = 10
    drop = 0.5

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

    flatten = Flatten()(merged_tensor)
    # reshape = Reshape((3 * num_filters,))(flatten)
    dropout = Dropout(drop)(flatten)
    # conc = Dense(40)(dropout)
    output = Dense(4, activation='sigmoid',
                   kernel_regularizer=regularizers.l2(0.01))(dropout)

    # this creates a model that includes
    model = Model(inputs, output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
    # embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDINGS_LEN, weights=[embedding_matrix], trainable=True)
    #
    # sequence_length = input_length
    # filter_sizes = [1, 2, 3, 4, 5]
    # num_filters = 100
    # drop = 0.5
    #
    # inputs = Input(shape=(sequence_length,))
    # embedding = embedding_layer(inputs)
    # reshape = Reshape((sequence_length, EMBEDDINGS_LEN, 1))(embedding)
    #
    # convs = []
    # maxpools = []
    #
    # for filter_size in filter_sizes:
    #     conv = Conv2D(num_filters, (filter_size, EMBEDDINGS_LEN), activation='relu',
    #                   kernel_regularizer=regularizers.l2(0.01))(reshape)
    #
    #     maxpool = MaxPooling2D(
    #         (sequence_length - filter_size + 1, 1), strides=(1, 1))(conv)
    #
    #     maxpools.append(maxpool)
    #     convs.append(conv)
    #
    # merged_tensor = concatenate(maxpools, axis=1)
    #
    # flatten = Flatten()(merged_tensor)
    # # reshape = Reshape((3 * num_filters,))(flatten)
    # dropout = Dropout(drop)(flatten)
    # conc = Dense(40)(dropout)
    # output = Dense(4, activation='sigmoid',
    #                kernel_regularizer=regularizers.l2(0.01))(conc)
    #
    # # this creates a model that includes
    # model = Model(inputs, output)
    #
    # opt = SGD(lr=1e-3)
    # model.compile(loss='binary_crossentropy', optimizer=opt,
    #               metrics=[metrics.binary_crossentropy, metrics.mae, metrics.categorical_accuracy])

    # callbacks = [EarlyStopping(monitor='val_loss')]

    return model
