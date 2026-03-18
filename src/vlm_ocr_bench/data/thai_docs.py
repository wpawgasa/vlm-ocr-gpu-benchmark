"""Thai financial/legal document dataset loader."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import structlog

from vlm_ocr_bench.data.datasets import DatasetLoader, DatasetSpec
from vlm_ocr_bench.data.omnidocbench import DocSample, GroundTruth

logger = structlog.get_logger()


def _normalize_text(text: str) -> str:
    """Unicode-normalize text to NFKC for consistent comparison."""
    return unicodedata.normalize("NFKC", text)


class ThaiDocsLoader(DatasetLoader):
    """Load local Thai financial/legal document dataset."""

    def __init__(self, spec: DatasetSpec) -> None:
        super().__init__(spec)
        self._gt_map: dict[str, GroundTruth] = {}

    def load(self) -> list[DocSample]:
        """Load from local directory."""
        data_dir = Path(self.spec.path or "data/thai_fin_legal/")
        logger.info("loading_dataset", name=self.spec.name, path=str(data_dir))

        if not data_dir.exists():
            logger.warning("dataset_dir_not_found", path=str(data_dir))
            self._samples = []
            return []

        # Look for images and optional annotations
        image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        image_files = sorted(f for f in data_dir.iterdir() if f.suffix.lower() in image_extensions)

        # Try loading annotations file
        annotations: dict[str, Any] = {}
        annotations_path = data_dir / "annotations.json"
        if annotations_path.exists():
            annotations = json.loads(annotations_path.read_text(encoding="utf-8"))

        samples: list[DocSample] = []
        for img_path in image_files:
            sample_id = img_path.stem
            ann = annotations.get(sample_id, {})

            sample = DocSample(
                sample_id=sample_id,
                image_path=img_path,
                doc_type=str(ann.get("doc_type", "thai_document")),
                language="th",
                layout_type=str(ann.get("layout_type", "single_column")),
                has_tables=bool(ann.get("has_tables", False)),
                has_formulas=bool(ann.get("has_formulas", False)),
                has_figures=bool(ann.get("has_figures", False)),
                attributes=ann,
            )
            samples.append(sample)

            md_text = ann.get("markdown", ann.get("ground_truth", ""))
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
        """Return filtered samples."""
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
