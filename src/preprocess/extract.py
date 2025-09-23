try:
    from ..models import COREF, EE, RE
except Exception:
    import os, sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models import COREF, EE, RE  # type: ignore

from tqdm import tqdm
import argparse
import logging
import json
import pandas as pd
import torch
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PARAMS:
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def run_preprocessor(data_path, output_path):
    logger.info(f"Running preprocessor on data: {data_path}")

    results = []

    data = pd.read_csv(data_path)

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        id = row["id"]

        context = row["context"]
        response = row["response"]

        coref = COREF(model_name=PARAMS.MODEL_NAME)

        context = coref(context)
        response = coref(response)

        del coref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ee = EE(model_name=PARAMS.MODEL_NAME)

        context_entities = ee(context)
        response_entities = ee(response)

        del ee
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        re = RE(model_name=PARAMS.MODEL_NAME)

        context_relations = re(context, context_entities)
        response_relations = re(response, response_entities)

        del re
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results.append(
            {
                "id": id,
                "context": context,
                "response": response,
                "context_entities": context_entities,
                "response_entities": response_entities,
                "context_relations": context_relations,
                "response_relations": response_relations,
            }
        )

    if not output_path.endswith(".json"):
        output_path += ".json"

    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specified model.")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the input data file.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output data file.",
    )

    args = parser.parse_args()
    run_preprocessor(args.data_path, args.output_path)
