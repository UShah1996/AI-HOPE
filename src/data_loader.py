import pandas as pd

def load_dataset(dataset_path: str):
    """Load main data table from a folder following AI-HOPE format."""
    data_path = f"{dataset_path}/main_data.tsv"
    df = pd.read_csv(data_path, sep="\t")
    return df