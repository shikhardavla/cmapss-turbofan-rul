from pathlib import Path
import pandas as pd


def load_cmapss(
    dataset_name: str,
    dataset_directory: str,
):
    
    """
    Loads CMAPSS data - train/test/RUL files for each experiment.

    Parameters: 
    -----------
    dataset_name: a string indicating which dataset or experiment
    dataset_directory: the directory where the dataset is stored

    Returns:
    --------
    (train_df, test_df, RUL_df) : a tuple of the pandas dataframes for the dataset,
    containing all the necessary files 
    """

    valid_datasets = {
        "FD001",
        "FD002",
        "FD003",
        "FD004",
    }


    # A quick error message if the experiment name is invalid
    # prior to creating the directory

    if dataset_name not in valid_datasets:
        raise ValueError(
            f"Invalid dataset: '{dataset_name}'."
            f"Expected one of: '{', '.join(sorted(valid_datasets))}'."
        )
    
    # Access the datasets

    dataset_directory = Path(dataset_directory)

    train_set_path = dataset_directory / f"train_{dataset_name}.txt"
    test_set_path = dataset_directory / f"test_{dataset_name}.txt"
    rul_path = dataset_directory / f"RUL_{dataset_name}.txt"

    # A quick error message if the path does not exist
    
    for path in (train_set_path, test_set_path, rul_path):
        if not path.exists():
            raise FileNotFoundError(path)
        
    # Set column names, since text files are values only

    column_names = (
        ["unit_number", "time_cycles"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    # Create pd dataframes for the purpose of 

    train_df = pd.read_csv(
        train_set_path,
        sep=r"\s+",
        header=None,
        names=column_names,
    )

    test_df = pd.read_csv(
        test_set_path,
        sep=r"\s+",
        header=None,
        names=column_names,
    )

    rul_df = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
    )

    rul_df = rul_df.dropna(axis=1, how="all")

    if rul_df.shape[1] != 1:
        raise ValueError(
            f"Expected exactly 1 column in {rul_path}, got {rul_df.shape[1]}."
        )

    rul_df.columns = ["RUL"]

    train_df = train_df.dropna(axis=1, how="all")
    test_df = test_df.dropna(axis=1, how="all")

    return train_df, test_df, rul_df