"""Data pipeline: dataset registry, loaders, image utils, and tokenization."""

from vlm_ocr_bench.data.datasets import (
    DATASET_REGISTRY,
    DatasetLoader,
    DatasetSpec,
    get_dataset,
    get_loader,
    list_datasets,
)
from vlm_ocr_bench.data.image_utils import get_image_stats, preprocess_image
from vlm_ocr_bench.data.omnidocbench import DocSample, GroundTruth, GTElement
from vlm_ocr_bench.data.tokenization import PROMPT_TEMPLATES, build_inference_input

__all__ = [
    "DATASET_REGISTRY",
    "PROMPT_TEMPLATES",
    "DatasetLoader",
    "DatasetSpec",
    "DocSample",
    "GTElement",
    "GroundTruth",
    "build_inference_input",
    "get_dataset",
    "get_image_stats",
    "get_loader",
    "list_datasets",
    "preprocess_image",
]
