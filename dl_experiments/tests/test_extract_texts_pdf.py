"""

"""
import glob

from utils.constants import *
from utils.files_utils import *
from pre_processing import extract_texts_pdf
from progressbar import ProgressBar

if __name__ == '__main__':
    # unittest.main()
    print("================================================")
    print("Test extract texts from PDF")

    src_paths = [DATASET_PATH + SRC_PATH_STF,
                 DATASET_PATH + SRC_PATH_STJ,
                 DATASET_PATH + SRC_PATH_TJSC,
                 DATASET_PATH + SRC_PATH_JEC_UFSC,
                 DATASET_PATH + SRC_PATH_OTHERS]
    dest_paths = [PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_STF,
                  PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_STJ,
                  PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_TJSC,
                  PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_JEC_UFSC,
                  PROJECT_PATH + DEST_PATH_DATA + DEST_PATH_OTHER]

    pbar = ProgressBar()
    for i in pbar(range(len(src_paths))):
        src_path = src_paths[i]
        dest_path = dest_paths[i]

        print("Search in path:\t", src_path)
        # list_files_in_dir(src_path)

        # Digitalized PDF files
        files = glob.glob(src_path + "*d.pdf")
        for file in files:
            print("Applying OCR in: ", file)
            text = extract_texts_pdf.ocr_pdf_file(file)
            dest_file = dest_path + file.replace(src_path, "").replace(".pdf", ".txt")
            print("\tWriting to file")
            write_to_file(dest_file, text)

        # Remaining PDF files
        remaining_files = glob.glob(src_path + "*.pdf")

        for file in remaining_files:

            # Skip PDFs which OCR was applied
            if file in files:
                continue

            print("Applying Digital Text Extraction in: ", file)
            text = extract_texts_pdf.digital_pdf_file(file)
            dest_file = dest_path + file.replace(src_path, "").replace(".pdf", ".txt")
            print("\tWriting to file")
            write_to_file(dest_file, text)

        # Text Files
        text_files = glob.glob(src_path + "*.txt")

        for text_file in text_files:
            dest_file = dest_path + text_file.replace(src_path, "")
            copy_file(text_file, dest_file)
