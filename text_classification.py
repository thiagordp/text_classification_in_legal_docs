"""

@author Thiago R. Dal Pont
"""

import warnings

from models.cnn_text_classification import cnn_training
from models.lstm_text_classification import lstm_training

with warnings.catch_warnings():
    import glob
    import time

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import random
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.preprocessing.text import Tokenizer

    from gensim.models import KeyedVectors
    from imblearn.over_sampling import RandomOverSampler
    from keras import Input, regularizers, Model, metrics
    from keras.callbacks import EarlyStopping
    from keras.layers import Embedding, Reshape, Conv2D, MaxPooling2D, concatenate, Flatten, Dropout, Dense
    from keras.optimizers import Adam
    from keras_preprocessing.sequence import pad_sequences
    from keras_preprocessing.text import Tokenizer
    from sklearn.model_selection import train_test_split
    from keras.utils import to_categorical

    from evaluation.classifier_eval import full_evaluation
    from utils.constants import *
    from pre_processing import text_preprocessing


def get_data():
    # Load "procedentes"
    print("Loading text files...")
    data = []

    dict_classes = {
        PATH_LAB_PROC: PROCEDENTE,
        PATH_LAB_INPROC: IMPROCEDENTE,
        PATH_LAB_EXT: EXTINCAO,
        PATH_LAB_PARC_PROC: PARCIALMENTE_PROCEDENTE
    }

    for path_class in dict_classes.keys():

        folder = DATASET_PATH + PATH_LAB_JEC_UFSC + path_class
        file_paths = glob.glob(folder + "*.txt")

        for file_path in file_paths:
            with open(file_path) as f:
                raw_content = f.read()
                file_name = file_path.replace(folder, "")
                data.append([file_name, raw_content, dict_classes[path_class]])

    print("Pre-processing...")
    processed_data = []

    for file_name, content, label in data:
        clean_text = text_preprocessing.clear_text(content)
        processed_text = text_preprocessing.pre_process(clean_text)
        processed_data.append([file_name, processed_text, label])

    df = pd.DataFrame(data=processed_data, columns=[
        "file_name", "content", "label"])
    df.to_csv(PROJECT_PATH + DEST_PATH_DATA +
              DEST_PATH_FINAL + "jec_ufsc_dataset.csv")

    x = df[["file_name", "content"]]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=(
                                                                int(time.time()) % 2 ** 32),
                                                        stratify=y)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    _path = PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_FINAL + "jec_ufsc_train.csv"
    x = pd.DataFrame(data=x_train, columns=["file_name", "content"])
    y = pd.DataFrame(data=y_train, columns=["label"])
    train_set = np.concatenate((x, y), axis=1)
    _train_df = pd.DataFrame(train_set, columns=["file_name", "content", "label"])
    _train_df.to_csv(_path, encoding="utf-8")

    _path = PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_FINAL + "jec_ufsc_test.csv"
    x = pd.DataFrame(data=x_test, columns=["file_name", "content"])
    y = pd.DataFrame(data=y_test, columns=["label"])
    test_set = np.concatenate((x, y), axis=1)
    _test_df = pd.DataFrame(test_set, columns=["file_name", "content", "label"])
    _test_df.to_csv(_path, encoding="utf-8")


if __name__ == "__main__":
    print("Text Classification")
    print("==========================")

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
