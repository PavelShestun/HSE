import pandas as pd

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'original_index'})

    df.dropna(subset=['Avrg_500_med_win_propensity'], inplace=True)

    columns_to_drop = [
        'raw_propensity1', 'raw_propensity2',
        '500_med_win_propensity1', '500_med_win_propensity2'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    df.reset_index(drop=True, inplace=True)

    return df