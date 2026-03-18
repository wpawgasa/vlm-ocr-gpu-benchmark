"""Real5-OmniDocBench dataset loader (5 degradation scenarios)."""

from __future__ import annotations

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


class Real5OmniDocLoader(DatasetLoader):
    """Load Real5-OmniDocBench with 5 degradation scenarios."""

    def __init__(self, spec: DatasetSpec) -> None:
        super().__init__(spec)
        self._gt_map: dict[str, GroundTruth] = {}

    def load(self) -> list[DocSample]:
        """Download from HuggingFace and parse all scenarios."""
        from datasets import load_dataset

        logger.info("loading_dataset", name=self.spec.name, hf_id=self.spec.hf_id)

        ds = load_dataset(
            self.spec.hf_id,
            split="test",
            trust_remote_code=True,
        )

        samples: list[DocSample] = []
        for idx, row in enumerate(ds):
            sample_id = str(row.get("id", f"real5_{idx}"))
            image = row.get("image")
            scenario = str(row.get("scenario", "unknown"))

            image_path = Path(f"/tmp/real5_omnidoc/{sample_id}.png")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if image is not None:
                image.save(str(image_path))

            attrs: dict[str, Any] = dict(row) if isinstance(row, dict) else {}
            attrs["scenario"] = scenario

            sample = DocSample(
                sample_id=sample_id,
                image_path=image_path,
                doc_type=str(row.get("doc_type", "unknown")),
                language=str(row.get("language", "en")),
                attributes=attrs,
            )
            samples.append(sample)

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
        """Return filtered samples, optionally by doc type."""
        if self._samples is None:
            self.load()
        assert self._samples is not None

        result = self._samples
        if doc_types:
            result = [s for s in result if s.doc_type in doc_types]
        if max_samples is not None:
            result = result[:max_samples]
        return result

    def get_samples_by_scenario(
        self,
        scenario: str,
        max_samples: int | None = None,
    ) -> list[DocSample]:
        """Filter samples by degradation scenario."""
        if self._samples is None:
            self.load()
        assert self._samples is not None

        result = [s for s in self._samples if s.attributes.get("scenario") == scenario]
        if max_samples is not None:
            result = result[:max_samples]
        return result

    def get_ground_truth(self, sample_id: str) -> GroundTruth:
        """Return structured ground truth for a sample."""
        if sample_id not in self._gt_map:
            msg = f"No ground truth for sample: {sample_id!r}"
            raise KeyError(msg)
        return self._gt_map[sample_id]
