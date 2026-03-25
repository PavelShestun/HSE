import os

# Paths
DATA_DIR = '../data'
GENOME_FASTA_PATH = os.path.join(DATA_DIR, 'U00096.3.fasta')
CSV_FILE_PATH = os.path.join(DATA_DIR, 'mmc4.csv')

# Params
UPSTREAM_WINDOW_SIZE = 500
RANDOM_STATE = 42
N_SPLITS_CV = 5

# Encoding
BASES = ['A', 'C', 'G', 'T', 'N']
NUM_BASES = len(BASES)

# CNN
EPOCHS = 15
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2