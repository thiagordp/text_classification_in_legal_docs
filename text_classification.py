"""

@author Thiago R. Dal Pont
"""
import json
import warnings

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras.utils.vis_utils import plot_model

from evaluation.classifier_eval import evaluate_classifier
from models.cnn_text_classification import cnn_model

with warnings.catch_warnings():
    import glob
    import time

    import pandas as pd
    import tensorflow as tf

    import random
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore", category=FutureWarning)
    tf.get_logger().setLevel('ERROR')

    from sklearn.model_selection import StratifiedKFold, train_test_split
    import logging
    import sys
    import numpy as np
    import tqdm

    from models.lstm_text_classification import lstm_training, EMBEDDING_DIM, MAX_NB_WORDS, MAX_LEN_SEQ, lstm_model, bilstm_model
    from utils.constants import *
    from pre_processing import text_preprocessing

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


def get_data():
    # Load "procedentes"
    logging.info("Loading text files...")
    data = []

    dict_classes = {
        PATH_LAB_PROC: PROCEDENTE,
        PATH_LAB_INPROC: IMPROCEDENTE,
        PATH_LAB_EXT: EXTINCAO,
        PATH_LAB_PARC_PROC: PARCIALMENTE_PROCEDENTE
    }

    for path_class in dict_classes.keys():

        folder = DATASET_PATH + PATH_LAB_JEC_UFSC + path_class
        # print(folder)
        file_paths = glob.glob(folder + "*.txt")

        for file_path in file_paths:
            with open(file_path) as f:
                raw_content = f.read()
                file_name = file_path.replace(folder, "")
                data.append([file_name, raw_content, dict_classes[path_class]])

    logging.info("Pre-processing...")
    processed_data = []

    for file_name, content, label in data:
        clean_text = text_preprocessing.clear_text(content)
        processed_text = text_preprocessing.pre_process(clean_text)
        processed_data.append([file_name, processed_text, label])

    df = pd.DataFrame(data=processed_data, columns=[
        "file_name", "content", "label"])
    df.to_csv(PROJECT_PATH + DEST_PATH_DATA +
              DEST_PATH_FINAL + "jec_ufsc_dataset.csv")

    x = df["content"]
    y = df["label"]

    # x_train, x_test, y_train, y_test = train_test_split(x,
    #                                                     y,
    #                                                     test_size=0.2,
    #                                                     random_state=(
    #                                                             int(time.time()) % 2 ** 32),
    #                                                     stratify=y)
    #
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    #
    # _path = PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_FINAL + "jec_ufsc_train.csv"
    # x = pd.DataFrame(data=x_train, columns=["file_name", "content"])
    # y = pd.DataFrame(data=y_train, columns=["label"])
    # train_set = np.concatenate((x, y), axis=1)
    # _train_df = pd.DataFrame(train_set, columns=["file_name", "content", "label"])
    # _train_df.to_csv(_path, encoding="utf-8")
    #
    # _path = PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_FINAL + "jec_ufsc_test.csv"
    # x = pd.DataFrame(data=x_test, columns=["file_name", "content"])
    # y = pd.DataFrame(data=y_test, columns=["label"])
    # test_set = np.concatenate((x, y), axis=1)
    # _test_df = pd.DataFrame(test_set, columns=["file_name", "content", "label"])
    # _test_df.to_csv(_path, encoding="utf-8")

    return x, y


def load_embeddings():
    ignore_list = [
        # "pre-trained_general/model_word2vec_ptr_cbow_100",
        # "pre-trained_legal/fasttext_sg_3500000000_100",
        # "pre-trained_air_transport/word2vec_sg_100000000_100",
        # "pre-trained_legal/word2vec_cbow_3500000000_100",
        # "pre-trained_air_transport/word2vec_cbow_100000000_100",
        # "pre-trained_air_transport/fasttext_cbow_100000000_100",
        # "pre-trained_general/model_fasttext_ptr_skipgram_100",
        # "pre-trained_air_transport/glove_100000000_100",
        # "pre-trained_general/model_gove_ptr_100",
        # "pre-trained_general/model_fasttext_ptr_cbow_100",
        # "pre-trained_legal/word2vec_sg_3500000000_100",
        # "pre-trained_legal/fasttext_cbow_3500000000_100",
        # "pre-trained_general/model_wang2vec_ptr_skipgram_100",
        # "pre-trained_legal/glove_3500000000_100"
    ]
    matches = []
    logging.info("Listing Embeddings")
    for root, dirnames, filenames in os.walk(EMBEDDINGS_PATH):
        for filename in filenames:
            if filename.endswith('.txt'):
                path_emb = os.path.join(root, filename)

                # found = False
                # for item in ignore_list:
                #     if path_emb.find(item) != -1:
                #         found = True
                #         break
                #
                # if found:
                #     continue
                matches.append(os.path.join(root, filename))

    logging.info("Loading embeddings")
    dict_list_embeddings = {}
    random.shuffle(matches)

    for path_emb in tqdm.tqdm(matches):
        if path_emb.find("glove") != -1:

            temp = get_tmpfile("glove2word2vec.txt")
            glove2word2vec(path_emb, temp)

            word_vectors = KeyedVectors.load_word2vec_format(temp, binary=False)
        else:
            word_vectors = KeyedVectors.load_word2vec_format(path_emb, binary=False)

        key = path_emb.replace(EMBEDDINGS_PATH, "").replace(".txt", "")
        dict_list_embeddings[key] = word_vectors

        # return dict_list_embeddings # TODO: remove

    logging.info(dict_list_embeddings)

    return dict_list_embeddings


def train_dl():
    dict_results = {
    }

    dict_load_embeddings = load_embeddings()

    # Create new
    x, y = get_data()

    k_splits = 10
    random_state = int(str(int((random.random() * random.random() * time.time())))[::-1]) % 2 ** 32
    kfold = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=random_state)

    final_set = [[train_ix, test_ix] for train_ix, test_ix in kfold.split(x, y)]

    for emb_model_key in dict_load_embeddings.keys():
        logging.info("Training models for %s" % emb_model_key)

        word_vectors = dict_load_embeddings[emb_model_key]

        cross_val_dict = {
            "cnn": {
                "acc": [],
                "f1": [],
                "conf_mat": []
            },
            "lstm": {
                "acc": [],
                "f1": [],
                "conf_mat": []
            },
            "cnn_lstm": {
                "acc": [],
                "f1": [],
                "conf_mat": []
            }
        }

        cross_val_i = 0
        for train_ix, test_ix in final_set:
            cross_val_i += 1
            logging.info("Cross val - %d of %d" % (cross_val_i, k_splits))

            tf.keras.backend.clear_session()

            x_train = np.array(x)[train_ix.astype(int)]
            x_test = np.array(x)[test_ix.astype(int)]
            y_train = np.array(y)[train_ix.astype(int)]
            y_test = np.array(y)[test_ix.astype(int)]

            tokenizer = Tokenizer(
                num_words=MAX_NB_WORDS,
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                lower=True
            )

            tokenizer.fit_on_texts(x_train)
            word_index = tokenizer.word_index
            logging.info('Found %s unique tokens.' % len(word_index))

            x_train = tokenizer.texts_to_sequences(x_train)
            x_train = pad_sequences(x_train, maxlen=MAX_LEN_SEQ)
            logging.info('Shape of data tensor: ' + str(x_train.shape))

            y_train = pd.get_dummies(y_train).values

            vocabulary_size = min(len(word_index) + 1, MAX_NB_WORDS)
            embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

            vec = np.random.rand(EMBEDDING_DIM)
            for word, index_w in word_index.items():
                if index_w >= MAX_NB_WORDS:
                    continue
                try:
                    embedding_vector = word_vectors[word]
                    embedding_matrix[index_w] = embedding_vector
                except KeyError:
                    # vec = np.random.rand(EMBEDDING_DIM)
                    embedding_matrix[index_w] = vec

            epochs = 20
            batch_size = 16

            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                              stratify=y_train)

            sequences_test = tokenizer.texts_to_sequences(x_test)
            x_test = pad_sequences(sequences_test, maxlen=MAX_LEN_SEQ)

            logging.info("-" * 100)
            logging.info("Training LSTM")
            tf.keras.backend.clear_session()
            model = lstm_model(x_train.shape[1], embedding_matrix)
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val))

            logging.info("\tEvaluating....")

            y_pred = model.predict(x_test)
            y_pred = [np.argmax(y) for y in y_pred]
            evaluate_classifier(y_test, y_pred, cross_val_dict["lstm"])

            logging.info("-" * 100)
            logging.info("Training CNN")
            tf.keras.backend.clear_session()
            model = cnn_model(x_train.shape[1], embedding_matrix)
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val))

            logging.info("\tEvaluating....")
            # sequences_test = tokenizer.texts_to_sequences(x_test)
            # x_test = pad_sequences(sequences_test, maxlen=MAX_LEN_SEQ)
            y_pred = model.predict(x_test)
            y_pred = [int(np.argmax(y)) for y in y_pred]
            evaluate_classifier(y_test, y_pred, cross_val_dict["cnn"])

            logging.info("-" * 100)
            logging.info("Training Bi-LSTM")
            tf.keras.backend.clear_session()
            model = bilstm_model(x_train.shape[1], embedding_matrix)
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val))

            logging.info("\tEvaluating....")
            # sequences_test = tokenizer.texts_to_sequences(x_test)

            y_pred = model.predict(x_test)
            y_pred = [int(np.argmax(y)) for y in y_pred]
            evaluate_classifier(y_test, y_pred, cross_val_dict["cnn_lstm"])

        logging.info("cross_val dict:")
        logging.info(cross_val_dict)
        dict_results[emb_model_key] = cross_val_dict

        with open("data/results_com_disp.json", "w+") as fp:
            json.dump(dict_results, fp, indent=4)


def export_model_image():
    #model = bilstm_model(1000, np.random.rand(12000, 100))
    #model = cnn_model(1000, np.random.rand(12000, 100))
    model = lstm_model(1000, np.random.rand(12000, 100))
    plot_model(model, to_file='model.png')


def old_classif():
    get_data()

    train_df = pd.read_csv(
        DEST_PATH_DATA + DEST_PATH_FINAL + "jec_ufsc_train.csv")
    test_df = pd.read_csv(
        DEST_PATH_DATA + DEST_PATH_FINAL + "jec_ufsc_test.csv")

    embeddings_path = glob.glob(EMBEDDINGS_PATH + "*.txt")

    random.shuffle(embeddings_path)

    for path_emb in embeddings_path:
        print("-------------------------------------------------------------------------------")
        print("Using embeddings: ", path_emb)

        # cnn_training(train_df, test_df, path_emb)
        lstm_training(train_df, test_df, path_emb)

    # TODO: Call classifiers (CNN, LSTM, RNN, CNN-LSTM, etc.)

    # TODO: Evaluate classifiers


if __name__ == "__main__":
    # print("Text Classification")
    # print("==========================")
    # train_dl()
    export_model_image()
    # process_json_results("data/results_prod_1.json")
