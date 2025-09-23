import pandas as pd
from pathlib import Path


class PARAMS:
    N_SPLITS = 7
    DATA = Path("D:/.projects/uit-ds-2025/dataset/vihallu-train.csv")
    BASE_PATH = Path("D:/.projects/uit-ds-2025/dataset")
    BASE_FILENAME = "vihallu-train"


def split_into_n_parts(data: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Splits the DataFrame into n approximately equal parts."""
    return [data[i::n].reset_index(drop=True) for i in range(n)]


def save_splits(
    splits: list[pd.DataFrame], base_path: Path, base_filename: str
) -> None:
    """Saves each split DataFrame to a CSV file."""
    base_path.mkdir(parents=True, exist_ok=True)
    for i, split in enumerate(splits):
        split.to_csv(base_path / f"{base_filename}_part_{i+1}.csv", index=False)


def main():
    # Example usage
    data = pd.read_csv(PARAMS.DATA)
    n_splits = PARAMS.N_SPLITS
    splits = split_into_n_parts(data, n_splits)
    save_splits(splits, PARAMS.BASE_PATH, PARAMS.BASE_FILENAME)


if __name__ == "__main__":
    main()
