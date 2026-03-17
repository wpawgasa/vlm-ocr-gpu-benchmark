# 04 — Model Adapters

## Overview

Model registry, HuggingFace loading, quantization wrappers, and per-model adapters that normalize prompt format, output parsing, and vLLM configuration.

## Checklist

### `models/registry.py`

- [ ] Define `MODEL_REGISTRY: dict[str, ModelConfig]` mapping model names to configs
- [ ] Implement `get_model_config(name: str) -> ModelConfig`
- [ ] Implement `list_models() -> list[str]`

### `models/loader.py`

- [ ] Implement `load_model_and_tokenizer(model_config, gpu_config)` — HuggingFace AutoModel loading
- [ ] Handle `trust_remote_code=True` for all models
- [ ] Apply appropriate dtype based on precision mode
- [ ] Return model + tokenizer + processor (for vision models)

### `models/quantization.py`

- [ ] Implement FP8 quantization wrapper (for H100 + B300)
- [ ] Implement FP4 / NVFP4 quantization wrapper (B300 only)
- [ ] Auto-detect quantization support based on GPU compute capability
- [ ] Validate quantization mode against model compatibility

### `models/adapters/base.py`

- [ ] Define `OCRModelAdapter` ABC with methods:
  - `get_model_id() -> str`
  - `build_prompt(image, output_format, task) -> dict`
  - `parse_output(raw_output) -> ParsedDocument`
  - `get_vllm_kwargs(gpu_config) -> dict`
  - `get_supported_resolutions() -> list[int]`
  - `get_max_output_tokens() -> int`
  - `estimate_vision_tokens(width, height) -> int`
- [ ] Define `ParsedDocument` dataclass (raw_text, markdown, html, elements, parse_time_ms, token_count)
- [ ] Define `DocElement` dataclass (type, content, bbox, confidence)

### Model Adapters (one per model)

- [ ] `models/adapters/paddleocr_vl.py` — PaddleOCR-VL (0.9B, ERNIE-based)
- [ ] `models/adapters/dots_ocr.py` — dots.ocr-1.5 (3B)
- [ ] `models/adapters/nanonets_ocr.py` — Nanonets-OCR2 (3B)
- [ ] `models/adapters/deepseek_ocr.py` — DeepSeek-OCR (3B)
- [ ] `models/adapters/olmocr.py` — OlmOCR2 (7B, Qwen2-VL based)
- [ ] `models/adapters/chandra_ocr.py` — Chandra-OCR (9B, Qwen2-VL based)

Each adapter must implement:
- [ ] Model-specific prompt template construction
- [ ] Special token handling in output parsing
- [ ] Model-specific vLLM kwargs (max_model_len, dtype, etc.)
- [ ] Vision token estimation for that architecture

## Key Rules

- Every model has a unique prompt format — adapters must not share prompt logic
- `trust_remote_code=True` is required for all target models
- vLLM kwargs vary significantly per model — always defer to adapter
- Test each adapter's `build_prompt` and `parse_output` in unit tests with fixtures
