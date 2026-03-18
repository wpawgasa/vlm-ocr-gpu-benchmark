"""OmniDocBench v1.5 dataset loader."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from vlm_ocr_bench.data.datasets import DatasetLoader, DatasetSpec

logger = structlog.get_logger()


BBox = tuple[float, float, float, float]


@dataclass
class GTElement:
    """A ground-truth document element with bounding box."""

    element_id: int
    type: str  # "title", "text", "table", "formula", etc.
    content: str  # text / LaTeX / HTML
    bbox: BBox
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruth:
    """Structured ground truth for a document sample."""

    markdown: str
    elements: list[GTElement] = field(default_factory=list)
    reading_order: list[int] = field(default_factory=list)
    bboxes: list[BBox] = field(default_factory=list)


@dataclass
class DocSample:
    """A single document sample from OmniDocBench."""

    sample_id: str
    image_path: Path
    doc_type: str  # e.g., "academic", "financial"
    language: str = "en"
    layout_type: str = "single_column"
    has_tables: bool = False
    has_formulas: bool = False
    has_figures: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)
    _ground_truth: GroundTruth | None = field(default=None, repr=False)

    @property
    def image(self) -> Any:
        """Lazily load PIL Image to avoid memory blow-up."""
        from PIL import Image

        return Image.open(self.image_path)


def _normalize_text(text: str) -> str:
    """Unicode-normalize text to NFKC for consistent comparison."""
    return unicodedata.normalize("NFKC", text)


class OmniDocBenchLoader(DatasetLoader):
    """Load and preprocess OmniDocBench v1.5 for evaluation."""

    def __init__(self, spec: DatasetSpec) -> None:
        super().__init__(spec)
        self._gt_map: dict[str, GroundTruth] = {}

    def load(self) -> list[DocSample]:
        """Download from HuggingFace and parse annotation JSON."""
        from datasets import load_dataset

        logger.info("loading_dataset", name=self.spec.name, hf_id=self.spec.hf_id)

        ds = load_dataset(
            self.spec.hf_id,
            split=self.spec.split or "test",
            trust_remote_code=True,
        )

        samples: list[DocSample] = []
        for idx, row in enumerate(ds):
            sample_id = str(row.get("id", f"omnidoc_{idx}"))
            image = row.get("image")

            # Save image to temp path for lazy loading
            image_path = Path(f"/tmp/omnidocbench/{sample_id}.png")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if image is not None:
                image.save(str(image_path))

            doc_type = str(row.get("doc_type", "unknown"))
            attrs = dict(row) if isinstance(row, dict) else {}

            sample = DocSample(
                sample_id=sample_id,
                image_path=image_path,
                doc_type=doc_type,
                language=str(row.get("language", "en")),
                layout_type=str(row.get("layout_type", "single_column")),
                has_tables=bool(row.get("has_tables", False)),
                has_formulas=bool(row.get("has_formulas", False)),
                has_figures=bool(row.get("has_figures", False)),
                attributes=attrs,
            )
            samples.append(sample)

            # Store ground truth if available
            md_text = row.get("markdown", row.get("ground_truth", ""))
            if md_text:
                self._gt_map[sample_id] = GroundTruth(
                    markdown=_normalize_text(str(md_text)),
                )

        self._samples = samples
        logger.info("dataset_loaded", name=self.spec.name, num_samples=len(samples))
        return samples

    def get_samples(
        self,
        doc_types: list[str] | None = None,
        max_samples: int | None = None,
    ) -> list[DocSample]:
        """Filter samples by document type."""
        if self._samples is None:
            self.load()
        assert self._samples is not None

        result = self._samples
        if doc_types:
            result = [s for s in result if s.doc_type in doc_types]
        if max_samples is not None:
            result = result[:max_samples]
        return result

    def get_ground_truth(self, sample_id: str) -> GroundTruth:
        """Return structured ground truth for a sample."""
        if sample_id not in self._gt_map:
            msg = f"No ground truth for sample: {sample_id!r}"
            raise KeyError(msg)
        return self._gt_map[sample_id]
