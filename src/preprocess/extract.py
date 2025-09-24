"""Preprocessing pipeline with optional multi-GPU (2x) support.

Kaggle Dual GPU Usage:
----------------------
1. In Kaggle Notebook settings, enable "Two GPUs" (if available) or choose a
    GPU type with multiple devices.
2. (Usually not required) You can restrict visible GPUs via environment var:
         import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
3. Run this script; it will automatically set device_map="auto" when it detects
    more than one CUDA device. This shards the model across both GPUs.
4. If you want single-GPU behavior, pass --device cuda:0 (then device_map will be None).

Examples:
    python -m src.preprocess.extract --data_path dataset/vihallu-train.csv \
         --output_path build/processed/train.json --model_name Qwen/Qwen3-4B-Instruct-2507

    # Limit rows & disable reuse (to reduce peak VRAM but slower)
    python -m src.preprocess.extract --data_path dataset/vihallu-train.csv \
         --output_path build/processed/train_head50.json --max_rows 50 --no_reuse
"""

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
import os
from typing import Optional, Any, cast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PARAMS:
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    DEVICE = "auto"  # auto | cpu | cuda | cuda:0 etc.
    MAX_ROWS: Optional[int] = None  # limit rows for debugging
    REUSE_MODELS = True  # load once then reuse (recommended)


def _resolve_device_map(device: str):
    """Return device_map argument for model loading.

    If multiple GPUs available and user picked cuda/auto -> use device_map="auto" for
    model sharding across 2 GPUs (e.g. Kaggle P100 / T4 pair or dual L4).
    """
    if device == "cpu":
        return None
    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and device.startswith("cuda")
    ):
        return "auto"
    if device == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "auto"
    return None


def run_preprocessor(
    data_path,
    output_path,
    model_name=None,
    device=None,
    max_rows=None,
    reuse_models=True,
):
    model_name = model_name or PARAMS.MODEL_NAME
    device = device or PARAMS.DEVICE
    max_rows = max_rows if max_rows is not None else PARAMS.MAX_ROWS
    reuse_models = reuse_models if reuse_models is not None else PARAMS.REUSE_MODELS

    logger.info(
        f"Running preprocessor | data={data_path} | model={model_name} | device={device} | multi_gpu={torch.cuda.device_count() if torch.cuda.is_available() else 0} GPUs | reuse_models={reuse_models}"
    )
    if torch.cuda.is_available():
        logger.info(
            f"CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
        )

    device_map = _resolve_device_map(device)
    if device_map == "auto":
        logger.info("Using HuggingFace device_map=auto for multi-GPU sharding")

    # Optional single construction of models
    coref_model: Any = None
    ee_model: Any = None
    re_model: Any = None
    if reuse_models:
        coref_model = COREF(model_name=model_name, device=device)
        ee_model = EE(model_name=model_name, device=device, device_map=device_map)
        re_model = RE(model_name=model_name, device=device)

    results = []
    data = pd.read_csv(data_path)
    if max_rows:
        data = data.head(max_rows)

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        rid = row.get("id", idx)
        context = row["context"]
        response = row["response"]

        # COREF
        coref = (
            cast(Any, coref_model)
            if reuse_models
            else COREF(model_name=model_name, device=device)
        )
        context = coref(context)  # type: ignore[operator]
        response = coref(response)  # type: ignore[operator]
        if not reuse_models:
            del coref

        # EE (entity extraction)
        ee = (
            cast(Any, ee_model)
            if reuse_models
            else EE(model_name=model_name, device=device, device_map=device_map)
        )
        context_entities = ee(context)  # type: ignore[operator]
        response_entities = ee(response)  # type: ignore[operator]
        if not reuse_models:
            del ee

        # RE (relation extraction)
        rel = (
            cast(Any, re_model)
            if reuse_models
            else RE(model_name=model_name, device=device)
        )
        context_relations = rel(context, context_entities)  # type: ignore[operator]
        response_relations = rel(response, response_entities)  # type: ignore[operator]
        if not reuse_models:
            del rel

        if not reuse_models:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(
            {
                "id": rid,
                "context": context,
                "response": response,
                "context_entities": context_entities,
                "response_entities": response_entities,
                "context_relations": context_relations,
                "response_relations": response_relations,
            }
        )

    if reuse_models:
        # cleanup large models explicitly at end
        del coref_model, ee_model, re_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not output_path.endswith(".json"):
        output_path += ".json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, ensure_ascii=False, indent=4)
    logger.info(f"Wrote {len(results)} processed rows -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline with multi-GPU support."
    )
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
    parser.add_argument(
        "--model_name", type=str, default=PARAMS.MODEL_NAME, help="HF model id"
    )
    parser.add_argument(
        "--device", type=str, default=PARAMS.DEVICE, help="Device: auto|cpu|cuda|cuda:0"
    )
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Limit rows for debug"
    )
    parser.add_argument(
        "--no_reuse",
        action="store_true",
        help="Do not reuse models (loads every row; slower, lower VRAM peaks).",
    )

    args = parser.parse_args()
    run_preprocessor(
        args.data_path,
        args.output_path,
        model_name=args.model_name,
        device=args.device,
        max_rows=args.max_rows,
        reuse_models=not args.no_reuse,
    )
