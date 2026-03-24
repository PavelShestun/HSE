import numpy as np
from config import BASES, NUM_BASES, UPSTREAM_WINDOW_SIZE

def one_hot_encode_3d(sequence):
    encoding_map = {base: i for i, base in enumerate(BASES)}
    matrix = np.zeros((UPSTREAM_WINDOW_SIZE, NUM_BASES), dtype=np.int8)

    for i, char in enumerate(sequence):
        idx = encoding_map.get(char, encoding_map['N'])
        matrix[i, idx] = 1

    return matrix


def build_3d_tensor(df):
    X = np.array([one_hot_encode_3d(seq) for seq in df['upstream_context']])
    return X