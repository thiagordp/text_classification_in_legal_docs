"""

"""
import time

from imblearn.over_sampling import RandomOverSampler


def split_train_test(train_data, test_data):
    x_train = train_data[["file_name", "content"]]

    y_train = train_data["label"]

    ros = RandomOverSampler(random_state=(int(time.time()) % 2 ** 32))
    x_train, y_train = ros.fit_resample(x_train, y_train)

    x_train = x_train["content"]
    x_test = test_data["content"]
    y_test = test_data["label"]

    return x_train, y_train, x_test, y_test
