import numpy as np
from scipy.sparse import csr_matrix, vstack
from config import BASES

def one_hot_encode_sparse(sequence):
    encoding_map = {base: i for i, base in enumerate(BASES)}

    indices = []
    for i, base in enumerate(sequence):
        idx = encoding_map.get(base, encoding_map['N'])
        indices.append(i * len(BASES) + idx)

    data = np.ones(len(indices), dtype=np.int8)
    row_idx = np.zeros(len(indices), dtype=np.int32)
    col_idx = np.array(indices, dtype=np.int32)

    return csr_matrix((data, (row_idx, col_idx)),
                      shape=(1, len(sequence) * len(BASES)))


def build_sparse_matrix(df):
    encoded = [one_hot_encode_sparse(seq) for seq in df['upstream_context']]
    X = vstack(encoded)
    return X