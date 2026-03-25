from sklearn.model_selection import train_test_split

import config
from data.loader import load_dataframe
from data.genome import download_genome_if_needed, load_genome_sequence
from features.context import add_context_to_df
from features.encoding_sparse import build_sparse_matrix
from features.encoding_cnn import build_3d_tensor
from models.sklearn_models import train_sklearn_models
from models.cnn_model import build_cnn, train_cnn, evaluate_cnn

def run_pipeline():
    # Data
    df = load_dataframe(config.CSV_FILE_PATH)

    # Genome
    download_genome_if_needed(config.GENOME_FASTA_PATH)
    genome_seq = load_genome_sequence(config.GENOME_FASTA_PATH)

    # Context
    df = add_context_to_df(df, genome_seq, config.UPSTREAM_WINDOW_SIZE)

    y = df['Avrg_500_med_win_propensity'].values

    # --- SKLEARN ---
    X_sparse = build_sparse_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sparse, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    sklearn_results = train_sklearn_models(
        X_train, X_test, y_train, y_test, config.RANDOM_STATE
    )

    # --- CNN ---
    X_3d = build_3d_tensor(df)
    X_train_3d, X_test_3d, y_train, y_test = train_test_split(
        X_3d, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    model = build_cnn()
    train_cnn(model, X_train_3d, y_train, config)
    cnn_results = evaluate_cnn(model, X_test_3d, y_test)

    return {
        "sklearn": sklearn_results,
        "cnn": cnn_results
    }