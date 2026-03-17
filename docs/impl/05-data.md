# 05 — Data Pipeline

## Overview

Dataset registry, loaders for evaluation benchmarks and training data, image preprocessing, and prompt construction.

## Checklist

### `data/datasets.py`

- [ ] Define `DatasetSpec` dataclass (name, hf_id, source, version, split, num_samples, doc_types, loader_cls, etc.)
- [ ] Define `DATASET_REGISTRY` with entries:
  - `omnidocbench_v1.5` — 1355 samples, 9 doc types, HuggingFace
  - `olmocr_bench` — from GitHub allenai/olmocr
  - `real5_omnidocbench` — 6775 samples (1355 × 5 scenarios), HuggingFace
  - `thai_docs_500` — 500 local Thai financial/legal docs
  - `docmatix_50k` — 50K training samples from Docmatix, stratified
- [ ] Implement `get_dataset(name: str) -> DatasetSpec`
- [ ] Implement `get_loader(name: str) -> DatasetLoader`

### `data/omnidocbench.py`

- [ ] Define `DocSample` dataclass (sample_id, image, image_path, doc_type, language, layout_type, has_tables/formulas/figures, attributes)
- [ ] Define `GroundTruth` dataclass (markdown, elements, reading_order, bboxes)
- [ ] Define `GTElement` dataclass (element_id, type, content, bbox, attributes)
- [ ] Implement `OmniDocBenchLoader`
  - `load()` — download from HuggingFace, parse annotation JSON
  - `get_samples(doc_types, max_samples)` — filter by type
  - `get_ground_truth(sample_id)` — return structured GT

### `data/olmocr_bench.py`

- [ ] Implement `OlmOCRBenchLoader` — load from GitHub source

### `data/real5_omnidoc.py`

- [ ] Implement `Real5OmniDocLoader` — 5 degradation scenarios (scanning, warping, screen_photo, illumination, skew)

### `data/thai_docs.py`

- [ ] Implement `ThaiDocsLoader` — local Thai document set

### `data/docmatix.py`

- [ ] Implement `DocmatixLoader` — stratified sampling of 50K training pairs from Docmatix

### `data/image_utils.py`

- [ ] Implement `preprocess_image(image, target_resolution, model_name)`:
  - Resize longest side to target_resolution
  - Pad to square if model requires (check `ModelConfig.requires_padding`)
  - Preserve aspect ratio
  - Convert to RGB
  - No augmentation (deterministic benchmarking)
- [ ] Implement `get_image_stats(image) -> dict` (width, height, channels, file_size, DPI)

### `data/tokenization.py`

- [ ] Define `PROMPT_TEMPLATES` dict (document_parse_md, document_parse_html, ocr_text_only)
- [ ] Implement `build_inference_input(adapter, image, output_format, resolution) -> dict`
  - Preprocess image
  - Build prompt via adapter
  - Return dict compatible with vLLM generate()

## Key Rules

- Images are loaded lazily (PIL deferred loading) to avoid memory blow-up
- No data augmentation — benchmarking must be deterministic
- Ground truth must be unicode-normalized (NFKC) before comparison
- Thai text requires `pythainlp` for segmentation in BLEU computation
