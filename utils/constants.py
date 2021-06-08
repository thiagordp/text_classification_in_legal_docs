"""
In this file, alls the constants and files are defined

@author Thiago R. Dal Pont
"""

# Base Paths
# Paths for Ideapad320
PROJECT_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Projects/text-classification_main-proj/"
# PROJECT_PATH = "/home/egov/Documentos/Experiments/Projects/text-classification_main-proj/"
DATASET_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/"
# DATASET_PATH = "/home/egov/Documentos/Experiments/Datasets/"

# Paths for EGOV-PC
# PROJECT_PATH = ""
# DATASET_PATH = ""

# Source Paths
SRC_PATH_STF = "corpus-direito/acordaos/STF/"
SRC_PATH_STJ = "corpus-direito/acordaos/STJ/"
SRC_PATH_TJSC = "corpus-direito/acordaos/TJSC/"
SRC_PATH_OTHERS = "corpus-direito/others/"
SRC_PATH_JEC_UFSC = "corpus-direito/jec_ufsc/"

# Destiny paths
DEST_PATH_DATA = "data/"
DEST_PATH_STF = "stf/"
DEST_PATH_STJ = "stj/"
DEST_PATH_TJSC = "tjsc/"
DEST_PATH_OTHER = "other/"
DEST_PATH_FINAL = "final_dataset/"
DEST_PATH_OCR = "ocr_files/"
DEST_PATH_JEC_UFSC = "jec_ufsc/"

# Path to Labeled Sentences
PATH_LAB_JEC_UFSC = "processos_transp_aereo/merge_step_03/"
# Path for EGOV Desktop
# PATH_LAB_JEC_UFSC = ""
PATH_LAB_PROC = "procedente/"
PATH_LAB_INPROC = "improcedente/"
PATH_LAB_PARC_PROC = "parcialmente_procedente/"
PATH_LAB_EXT = "extincao/"

# Sentence labels
PROCEDENTE = 0
IMPROCEDENTE = 1
EXTINCAO = 2
PARCIALMENTE_PROCEDENTE = 3

# Embeddings Constants
EMBEDDINGS_PATH = "data/embeddings/"
EMBEDDINGS_LEN = 100
EMBEDDINGS_ITER = 200
