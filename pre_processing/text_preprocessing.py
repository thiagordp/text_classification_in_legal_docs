"""

@author Thiago Dal Pont
"""
import glob
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords

from utils.constants import *


def clear_text(text):
    text = str(text)

    text = text.replace("\n", " ").replace("\t", "")

    url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    text = re.sub(url_pattern, ' ', text)

    text = text.replace("http://www ", "")

    text = re.sub("-+", "", text)
    text = re.sub("\.+", "", text)

    # Symbols
    for symb in "/,-":
        text = text.replace(symb + " ", " ")

    for symb in "()[]{}!?\"§_“”‘’–'º•—|<>$#*@:;":
        text = text.replace(symb, " ")

    for symb in ".§ºª":
        text = text.replace(symb, " " + symb + " ")

    text = text.replace("⁄", "/")

    misspellings = {
        " r io ": " rio ",
        "ministrorelator": "ministro relator"
    }

    for key in misspellings:
        text = text.replace(key, misspellings[key])

    for letter in "bcdfghjklmnpqrstvwxyz":
        text = text.replace(" " + letter + " ", " ")

    text = re.sub(" +", " ", text, 0)

    return text


def pre_process(text):
    text = str(text)

    # Normalize
    text = text.lower()

    # Stop Words
    tokens = text.split()
    stop_words = set(stopwords.words("portuguese"))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    # stemmer = nltk.stem.RSLPStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]

    text = " ".join(tokens)

    return text


def merge_text_files(src_folders, destiny_folder):
    print("Merge text files")

    contents = []

    raw_contents = ""

    for folder in src_folders:
        print("\tGetting Texts from: ", folder)
        file_paths = glob.glob(folder + "*.txt")

        for file_path in file_paths:
            with open(file_path, "r") as file:
                raw_content = file.read()

                # Pré-processing
                clean_text = clear_text(raw_content)
                content = pre_process(clean_text)

                raw_contents += "\n " + content

                src = folder.split("/")[2]
                f = file_path.split("/")[3]
                contents.append([src, f, content])

    vocab = {}
    for sentence in contents:
        text = sentence[2].split()
        for word in text:
            try:
                vocab[word.lower()] += 1
            except KeyError:
                vocab[word.lower()] = 1

    # Create a list of tuples sorted by index 1 i.e. value field
    list_of_tuples = sorted(vocab.items(), key=lambda x: x[1])

    count = 0
    # Iterate over the sorted sequence
    for elem in list_of_tuples:
        print(elem[0], "\t::\t", elem[1])
        count += elem[1]

    print("Vocab size ->", len(vocab))
    print("Count -> ", count)

    final_dataset = pd.DataFrame(columns=["source", "file_name", "content"], data=contents)
    final_dataset.to_csv(destiny_folder + "/final_dataset.csv", encoding="utf-8", sep=";", decimal=",")
