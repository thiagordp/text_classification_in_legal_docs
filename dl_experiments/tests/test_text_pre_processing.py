"""

@author Thiago R. Dal Pont
"""

from utils.constants import *
from pre_processing import text_preprocessing

if __name__ == "__main__":
    print("Testing Pre-processing")

    print("Merge Database")

    src_paths = [PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_STF,
                 PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_STJ,
                 PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_TJSC,
                 PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_JEC_UFSC,
                 PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_OTHER]
    dest_path = PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_FINAL

    text_preprocessing.merge_text_files(src_folders=src_paths, destiny_folder=dest_path)
