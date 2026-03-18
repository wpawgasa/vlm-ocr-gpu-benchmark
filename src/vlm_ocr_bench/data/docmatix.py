"""Docmatix training dataset loader with stratified sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from vlm_ocr_bench.data.datasets import DatasetLoader
from vlm_ocr_bench.data.omnidocbench import DocSample

logger = structlog.get_logger()


class DocmatixLoader(DatasetLoader):
    """Load stratified sample from Docmatix for training benchmarks."""

    def load(self) -> list[DocSample]:
        """Download from HuggingFace and apply stratified sampling."""
        from datasets import load_dataset

        logger.info(
            "loading_dataset",
            name=self.spec.name,
            hf_id=self.spec.hf_id,
            sample_size=self.spec.sample_size,
        )

        ds = load_dataset(
            self.spec.hf_id,
            split=self.spec.split or "train",
            trust_remote_code=True,
            streaming=True,
        )

        # Stratified sampling by doc_type if configured
        target_size = self.spec.sample_size or 50000
        samples: list[DocSample] = []
        type_counts: dict[str, int] = {}

        for idx, row in enumerate(ds):
            if len(samples) >= target_size:
                break

            doc_type = str(row.get("doc_type", "general"))
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            sample_id = str(row.get("id", f"docmatix_{idx}"))
            image = row.get("image")

            image_path = Path(f"/tmp/docmatix/{sample_id}.png")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if image is not None:
                image.save(str(image_path))

            attrs: dict[str, Any] = {}
            attrs["doc_type"] = doc_type
            if "question" in row:
                attrs["question"] = row["question"]
            if "answer" in row:
                attrs["answer"] = row["answer"]

            sample = DocSample(
                sample_id=sample_id,
                image_path=image_path,
                doc_type=doc_type,
                attributes=attrs,
            )
            samples.append(sample)

        self._samples = samples
        logger.info(
            "dataset_loaded",
            name=self.spec.name,
            num_samples=len(samples),
            type_distribution=type_counts,
        )
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
