import pandas as pd

def load_dataset(dataset_path: str) -> pd.DataFrame:
    file_path = f"{dataset_path}/main_data.tsv"
    # Detect delimiter automatically
    with open(file_path, 'r') as f:
        sample = f.read(1024)
    sep = ',' if sample.count(',') > sample.count('\t') else '\t'
    df = pd.read_csv(file_path, sep=sep)
    df.columns = df.columns.astype(str).str.strip()
    return df
