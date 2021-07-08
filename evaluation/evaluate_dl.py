"""

"""
import json
import math

import numpy as np
import pandas as pd
import tqdm


def precision_calc(label, confusion_matrix):
    col = confusion_matrix[:, label]
    prec = confusion_matrix[label, label] / col.sum()
    if math.isnan(prec):
        prec = 0
    return prec


def recall_calc(label, confusion_matrix):
    row = confusion_matrix[label, :]
    recall = confusion_matrix[label, label] / row.sum()
    if math.isnan(recall):
        recall = 0
    return recall


def f1_calc(recall, precision):
    f1 = 2 * ((precision * recall) / (precision + recall))
    if math.isnan(f1):
        f1 = 0

    return f1


def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision_calc(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall_calc(label, confusion_matrix)
    return sum_of_recalls / columns


def get_class_metrics_from_conf_matrix(conf_matrix_list):
    f1_repetitions = []
    acc_repetitions = []

    for conf_matrix in conf_matrix_list:
        df_conf_matrix = pd.DataFrame(conf_matrix).to_numpy()

        f1_class = []
        acc_class = []

        for label in range(4):
            prec = precision_calc(label, df_conf_matrix)
            recall = recall_calc(label, df_conf_matrix)

            f1_class.append(f1_calc(recall, prec))

            # True negatives are all the samples that are not our current GT class (not the current row)
            # and were not predicted as the current class (not the current column)
            true_negatives = np.sum(np.delete(np.delete(df_conf_matrix, label, axis=0), label, axis=1))

            # True positives are all the samples of our current GT class that were predicted as such
            true_positives = df_conf_matrix[label, label]

            # The accuracy for the current class is ratio between correct predictions to all predictions
            acc_class.append((true_positives + true_negatives) / np.sum(df_conf_matrix))

        f1_repetitions.append(f1_class)
        acc_repetitions.append(acc_class)

    return f1_repetitions, acc_repetitions


def process_json_results(path_results):
    results_list_full = []
    results_list = []

    EMBEDDING_IGNORE = [
        "pre-trained_general/model_wang2vec_ptr_skipgram_100",
        "pre-trained_general/model_wang2vec_ptr_cbow_s100"
    ]

    with open(path_results) as fp:
        results_dict = json.load(fp)

        for embedding in tqdm.tqdm(results_dict.keys()):
            # logging.info(embedding)

            if embedding in EMBEDDING_IGNORE:
                continue

            for class_tech in results_dict[embedding].keys():

                conf_matrix_list = results_dict[embedding][class_tech]["conf_mat"]
                class_f1_score_list, class_acc_list = get_class_metrics_from_conf_matrix(conf_matrix_list)

                class_f1_df = pd.DataFrame(class_f1_score_list, columns=["P", "I", "E", "PP"])
                class_acc_df = pd.DataFrame(class_acc_list, columns=["P", "I", "E", "PP"])

                for acc_value, f1_value, class_f1, class_acc in zip(results_dict[embedding][class_tech]["acc"],
                                                                    results_dict[embedding][class_tech]["f1"],
                                                                    class_f1_score_list, class_acc_list):
                    results_row = [embedding, class_tech, acc_value, f1_value]
                    results_row.extend(class_f1)
                    results_row.extend(class_acc)
                    results_list_full.append(results_row)
                acc_results_mean = np.mean(results_dict[embedding][class_tech]["acc"])
                acc_stddev = np.std(results_dict[embedding][class_tech]["acc"])
                f1_results_mean = np.mean(results_dict[embedding][class_tech]["f1"])
                f1_stddev = np.std(results_dict[embedding][class_tech]["f1"])
                f1_p = np.mean(class_f1_df["P"])
                f1_i = np.mean(class_f1_df["I"])
                f1_e = np.mean(class_f1_df["E"])
                f1_pp = np.mean(class_f1_df["PP"])
                acc_p = np.mean(class_acc_df["P"])
                acc_i = np.mean(class_acc_df["I"])
                acc_e = np.mean(class_acc_df["E"])
                acc_pp = np.mean(class_acc_df["PP"])
                results_list.append([embedding, class_tech, acc_results_mean, acc_stddev, f1_results_mean, f1_stddev,
                                     f1_p, f1_i, f1_e, f1_pp, acc_p, acc_i, acc_e, acc_pp])

    df = pd.DataFrame(results_list_full,
                      columns=["Embedding", "Technique", "Accuracy", "F1-Score",
                               "F1-P", "F1-I", "F1-E", "F1-PP",
                               "Acc-P", "Acc-I", "Acc-E", "Acc-PP"])
    df.to_excel("data/results_full_table_com_disp.xlsx", index=False)

    df = pd.DataFrame(results_list, columns=["Embedding", "Technique", "Accuracy",
                                             "Acc Std Dev", "F1-Score", "F1 Std Dev",
                                             "F1-P", "F1-I", "F1-E", "F1-PP",
                                             "Acc-P", "Acc-I", "Acc-E", "Acc-PP"
                                             ])
    df.to_excel("data/results_table_com_disp.xlsx", index=False)
