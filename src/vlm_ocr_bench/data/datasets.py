"""Dataset registry and loader factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from vlm_ocr_bench.data.omnidocbench import DocSample

logger = structlog.get_logger()


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a benchmark or training dataset."""

    name: str
    loader_cls: str
    num_samples: int | None = None
    hf_id: str | None = None
    source: str = "huggingface"
    version: str | None = None
    split: str | None = None
    doc_types: list[str] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    path: str | None = None
    sample_size: int | None = None
    stratify_by: str | None = None


class DatasetLoader(ABC):
    """Base class for all dataset loaders."""

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self._samples: list[DocSample] | None = None

    @abstractmethod
    def load(self) -> list[DocSample]:
        """Load and return all samples from the dataset."""

    @abstractmethod
    def get_samples(
        self,
        doc_types: list[str] | None = None,
        max_samples: int | None = None,
    ) -> list[DocSample]:
        """Return filtered samples."""


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "omnidocbench_v1.5": DatasetSpec(
        name="omnidocbench_v1.5",
        hf_id="opendatalab/OmniDocBench",
        version="v1.5",
        split="test",
        num_samples=1355,
        doc_types=[
            "academic",
            "textbook",
            "slide",
            "financial",
            "newspaper",
            "exam",
            "note",
            "magazine",
            "colorful_textbook",
        ],
        loader_cls="OmniDocBenchLoader",
    ),
    "olmocr_bench": DatasetSpec(
        name="olmocr_bench",
        source="github:allenai/olmocr",
        loader_cls="OlmOCRBenchLoader",
    ),
    "real5_omnidocbench": DatasetSpec(
        name="real5_omnidocbench",
        hf_id="PaddlePaddle/Real5-OmniDocBench",
        num_samples=6775,
        scenarios=["scanning", "warping", "screen_photo", "illumination", "skew"],
        loader_cls="Real5OmniDocLoader",
    ),
    "thai_docs_500": DatasetSpec(
        name="thai_docs_500",
        source="local",
        path="data/thai_fin_legal/",
        num_samples=500,
        loader_cls="ThaiDocsLoader",
    ),
    "docmatix_50k": DatasetSpec(
        name="docmatix_50k",
        hf_id="HuggingFaceM4/Docmatix",
        split="train",
        sample_size=50000,
        stratify_by="doc_type",
        loader_cls="DocmatixLoader",
    ),
}

_LOADER_MAP: dict[str, str] = {
    "OmniDocBenchLoader": "vlm_ocr_bench.data.omnidocbench",
    "OlmOCRBenchLoader": "vlm_ocr_bench.data.olmocr_bench",
    "Real5OmniDocLoader": "vlm_ocr_bench.data.real5_omnidoc",
    "ThaiDocsLoader": "vlm_ocr_bench.data.thai_docs",
    "DocmatixLoader": "vlm_ocr_bench.data.docmatix",
}


def get_dataset(name: str) -> DatasetSpec:
    """Get dataset spec by name.

    Raises KeyError if the dataset name is not in the registry.
    """
    if name not in DATASET_REGISTRY:
        msg = f"Unknown dataset: {name!r}. Available: {list(DATASET_REGISTRY)}"
        raise KeyError(msg)
    return DATASET_REGISTRY[name]


def list_datasets() -> list[str]:
    """Return all registered dataset names."""
    return list(DATASET_REGISTRY.keys())


def get_loader(name: str) -> DatasetLoader:
    """Create a loader instance for the given dataset name.

    Uses importlib to lazily import the loader module, avoiding
    heavy dependencies at import time.

    Raises KeyError if dataset or loader class is not found.
    """
    import importlib

    spec = get_dataset(name)
    module_path = _LOADER_MAP.get(spec.loader_cls)
    if module_path is None:
        msg = f"Unknown loader class: {spec.loader_cls!r} for dataset {name!r}"
        raise KeyError(msg)

    module = importlib.import_module(module_path)
    loader_cls: type[DatasetLoader] = getattr(module, spec.loader_cls)
    logger.debug("loader_created", dataset=name, loader=spec.loader_cls)
    return loader_cls(spec)
