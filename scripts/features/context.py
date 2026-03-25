def get_upstream_context(row, genome_seq, window_size):
    pos = row['pos'] - 1
    strand = row['strand']

    if strand == '+':
        seq = genome_seq[max(0, pos - window_size):pos]
        return seq.rjust(window_size, 'N')

    elif strand == '-':
        seq = genome_seq[pos:min(len(genome_seq), pos + window_size)]
        seq = seq.ljust(window_size, 'N')

        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return "".join([complement.get(base, 'N') for base in reversed(seq)])

    return 'N' * window_size


def add_context_to_df(df, genome_seq, window_size):
    df['upstream_context'] = df.apply(
        lambda row: get_upstream_context(row, genome_seq, window_size),
        axis=1
    )
    return df