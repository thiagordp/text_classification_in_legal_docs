"""

@author Thiago Raulino Dal Pont
"""
import os
import shutil
from os import listdir
from os.path import isfile, join


def list_files_in_dir(path, ext=""):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    print("Files in: ", path)
    for file in files:
        print("\t", file)


def clear_dir(path):
    file_list = [f for f in os.listdir(path) if f.endswith(".jpg")]
    for f in file_list:
        os.remove(os.path.join(path, f))


def write_to_file(file, content):
    with open(file, "w+") as file_w:
        file_w.write(content)


def copy_file(file, dest):
    shutil.copy2(file, dest)
