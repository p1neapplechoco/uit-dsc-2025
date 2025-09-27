import pandas as pd
import requests
import json
import time
from typing import List, Tuple, Optional
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import gc

from src.models.ResponseNormalizer import ResponseNormalizer
from tqdm import tqdm


def process_csv(
    normalizer: ResponseNormalizer,
    file_path: str,
    output_path: str,
    batch_size: int = 1,
) -> Optional[pd.DataFrame]:
    """Process CSV file and normalize responses"""
    try:
        # Read CSV
        df = pd.read_csv(file_path)

        # Check required columns
        required_columns = ["context", "prompt", "response"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        print(f"Processing {len(df)} rows...")

        df["normalized_response"] = ""

        for i in tqdm(
            range(0, len(df), batch_size),
            total=(len(df) + batch_size - 1) // batch_size,
        ):
            batch_end = min(i + batch_size, len(df))
            print(f"Processing rows {i+1}-{batch_end}...")

            for idx in range(i, batch_end):
                row = df.iloc[idx]

                try:
                    normalized = normalizer(
                        str(row["context"]),
                        str(row["prompt"]),
                        str(row["response"]),
                    )

                    df.at[idx, "normalized_response"] = normalized

                except Exception as e:
                    print(f"Error processing row {idx}: {str(e)}")
                    df.at[idx, "normalized_response"] = str(row["response"])

        if output_path:
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"Normalized data saved to: {output_path}")

        return df

    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return None


def main(input, output, batch_size):
    normalizer = ResponseNormalizer(
        model_name="Qwen/Qwen3-4B-Instruct-2507", device="auto"
    )
    process_csv(normalizer, input, output, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.batch_size)
