"""Unit tests for the models module (no GPU required)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vlm_ocr_bench.config.schema import GPUConfig, GPUType, ModelConfig, PrecisionMode
from vlm_ocr_bench.models.adapters.base import DocElement, OCRModelAdapter, ParsedDocument
from vlm_ocr_bench.models.adapters.chandra_ocr import ChandraOcrAdapter
from vlm_ocr_bench.models.adapters.deepseek_ocr import DeepSeekOcrAdapter
from vlm_ocr_bench.models.adapters.dots_ocr import DotsOcrAdapter
from vlm_ocr_bench.models.adapters.nanonets_ocr import NanonetsOcrAdapter
from vlm_ocr_bench.models.adapters.olmocr import OlmOcrAdapter
from vlm_ocr_bench.models.adapters.paddleocr_vl import PaddleOcrVlAdapter
from vlm_ocr_bench.models.registry import (
    get_adapter,
    get_model_config,
    list_models,
)

# ─── Fixtures ───


@pytest.fixture
def sample_image() -> MagicMock:
    """Mock PIL Image for testing adapters."""
    img = MagicMock()
    img.size = (1024, 768)
    img.mode = "RGB"
    return img


@pytest.fixture
def h100_gpu_config() -> GPUConfig:
    return GPUConfig(
        gpu_type=GPUType.H100_SXM,
        precision_modes=[PrecisionMode.BF16],
    )


@pytest.fixture
def b300_gpu_config() -> GPUConfig:
    return GPUConfig(
        gpu_type=GPUType.B300_SXM,
        precision_modes=[PrecisionMode.BF16, PrecisionMode.FP8, PrecisionMode.FP4],
    )


# ─── DocElement Tests ───


class TestDocElement:
    def test_basic_creation(self) -> None:
        elem = DocElement(type="text", content="hello world")
        assert elem.type == "text"
        assert elem.content == "hello world"
        assert elem.bbox is None
        assert elem.confidence is None

    def test_with_bbox_and_confidence(self) -> None:
        elem = DocElement(
            type="table",
            content="| a | b |",
            bbox=(0.1, 0.2, 0.9, 0.8),
            confidence=0.95,
        )
        assert elem.bbox == (0.1, 0.2, 0.9, 0.8)
        assert elem.confidence == 0.95


# ─── ParsedDocument Tests ───


class TestParsedDocument:
    def test_basic_creation(self) -> None:
        doc = ParsedDocument(raw_text="raw", markdown="clean")
        assert doc.raw_text == "raw"
        assert doc.markdown == "clean"
        assert doc.html is None
        assert doc.elements == []
        assert doc.parse_time_ms == 0.0
        assert doc.token_count == 0

    def test_with_elements(self) -> None:
        elems = [DocElement(type="text", content="hello")]
        doc = ParsedDocument(
            raw_text="raw",
            markdown="hello",
            elements=elems,
            parse_time_ms=12.5,
            token_count=1,
        )
        assert len(doc.elements) == 1
        assert doc.parse_time_ms == 12.5


# ─── OCRModelAdapter Defaults ───


class _DummyAdapter(OCRModelAdapter):
    """Minimal concrete adapter for testing ABC defaults."""

    def get_model_id(self) -> str:
        return "test/dummy"

    def build_prompt(
        self,
        image: Any,
        output_format: Any = "markdown",
        task: str = "full_page_parse",
    ) -> dict[str, Any]:
        return {"prompt": "test", "images": [image]}

    def parse_output(self, raw_output: str) -> ParsedDocument:
        return ParsedDocument(raw_text=raw_output, markdown=raw_output)

    def get_vllm_kwargs(self, gpu_config: Any) -> dict[str, Any]:
        return {"trust_remote_code": True}


class TestOCRModelAdapterDefaults:
    def test_supported_resolutions(self) -> None:
        adapter = _DummyAdapter()
        assert adapter.get_supported_resolutions() == [1024, 1536, 2048]

    def test_max_output_tokens(self) -> None:
        adapter = _DummyAdapter()
        assert adapter.get_max_output_tokens() == 4096

    def test_estimate_vision_tokens(self) -> None:
        adapter = _DummyAdapter()
        # 224x224 image with 14px patches = 16*16 = 256 tokens
        assert adapter.estimate_vision_tokens(224, 224) == 256

    def test_estimate_vision_tokens_large(self) -> None:
        adapter = _DummyAdapter()
        # 1024x768 => 73*54 = 3942
        assert adapter.estimate_vision_tokens(1024, 768) == 73 * 54


# ─── Model Registry Tests ───


class TestModelRegistry:
    def test_list_models(self) -> None:
        models = list_models()
        assert len(models) == 6
        assert "paddleocr_vl_0.9b" in models
        assert "dots_ocr_1.5_3b" in models
        assert "nanonets_ocr2_3b" in models
        assert "deepseek_ocr_3b" in models
        assert "olmocr2_7b" in models
        assert "chandra_ocr_9b" in models

    def test_get_model_config(self) -> None:
        cfg = get_model_config("dots_ocr_1.5_3b")
        assert cfg.hf_model_id == "rednote-hilab/dots.ocr-1.5"
        assert cfg.params_billion == 3.0
        assert cfg.adapter == "DotsOcrAdapter"

    def test_get_model_config_invalid(self) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_config("nonexistent_model")

    def test_all_registry_entries_have_adapter(self) -> None:
        for name in list_models():
            cfg = get_model_config(name)
            assert cfg.adapter, f"Model {name} has empty adapter string"

    def test_registry_adapter_names_are_valid(self) -> None:
        """Every adapter string in the registry maps to a real adapter class."""
        for name in list_models():
            adapter = get_adapter(name)
            assert isinstance(adapter, OCRModelAdapter)


# ─── GetAdapter Tests ───


class TestGetAdapter:
    def test_paddleocr_adapter_type(self) -> None:
        assert isinstance(get_adapter("paddleocr_vl_0.9b"), PaddleOcrVlAdapter)

    def test_dots_adapter_type(self) -> None:
        assert isinstance(get_adapter("dots_ocr_1.5_3b"), DotsOcrAdapter)

    def test_nanonets_adapter_type(self) -> None:
        assert isinstance(get_adapter("nanonets_ocr2_3b"), NanonetsOcrAdapter)

    def test_deepseek_adapter_type(self) -> None:
        assert isinstance(get_adapter("deepseek_ocr_3b"), DeepSeekOcrAdapter)

    def test_olmocr_adapter_type(self) -> None:
        assert isinstance(get_adapter("olmocr2_7b"), OlmOcrAdapter)

    def test_chandra_adapter_type(self) -> None:
        assert isinstance(get_adapter("chandra_ocr_9b"), ChandraOcrAdapter)

    def test_invalid_model_name(self) -> None:
        with pytest.raises(KeyError):
            get_adapter("nonexistent")


# ─── Per-Adapter Tests ───


class TestPaddleOcrVlAdapter:
    def test_model_id(self) -> None:
        adapter = PaddleOcrVlAdapter()
        assert adapter.get_model_id() == "PaddlePaddle/PaddleOCR-VL-0.9B"

    def test_build_prompt(self, sample_image: MagicMock) -> None:
        adapter = PaddleOcrVlAdapter()
        result = adapter.build_prompt(sample_image)
        assert "messages" in result
        assert "images" in result
        assert result["images"] == [sample_image]

    def test_parse_output(self) -> None:
        adapter = PaddleOcrVlAdapter()
        doc = adapter.parse_output("  hello world  ")
        assert doc.markdown == "hello world"
        assert doc.raw_text == "  hello world  "

    def test_vllm_kwargs(self, h100_gpu_config: GPUConfig) -> None:
        adapter = PaddleOcrVlAdapter()
        kwargs = adapter.get_vllm_kwargs(h100_gpu_config)
        assert kwargs["trust_remote_code"] is True
        assert kwargs["max_model_len"] == 4096

    def test_supported_resolutions(self) -> None:
        adapter = PaddleOcrVlAdapter()
        assert adapter.get_supported_resolutions() == [1024]

    def test_max_output_tokens(self) -> None:
        adapter = PaddleOcrVlAdapter()
        assert adapter.get_max_output_tokens() == 4096


class TestDotsOcrAdapter:
    def test_model_id(self) -> None:
        adapter = DotsOcrAdapter()
        assert adapter.get_model_id() == "rednote-hilab/dots.ocr-1.5"

    def test_build_prompt(self, sample_image: MagicMock) -> None:
        adapter = DotsOcrAdapter()
        result = adapter.build_prompt(sample_image)
        assert "prompt" in result
        assert "<|im_start|>" in result["prompt"]
        assert result["images"] == [sample_image]

    def test_build_prompt_layout(self, sample_image: MagicMock) -> None:
        adapter = DotsOcrAdapter()
        result = adapter.build_prompt(sample_image, task="layout_only")
        assert "layout_only" in result["prompt"]

    def test_parse_output_strips_special_tokens(self) -> None:
        adapter = DotsOcrAdapter()
        raw = "<|im_start|>assistant\n# Title\nSome text<|im_end|>"
        doc = adapter.parse_output(raw)
        assert "<|im_start|>" not in doc.markdown
        assert "<|im_end|>" not in doc.markdown
        assert "Title" in doc.markdown

    def test_parse_output_extracts_elements(self) -> None:
        adapter = DotsOcrAdapter()
        raw = "# Title\nSome text\n| a | b |"
        doc = adapter.parse_output(raw)
        types = [e.type for e in doc.elements]
        assert "title" in types
        assert "text" in types
        assert "table" in types

    def test_vllm_kwargs(self, h100_gpu_config: GPUConfig) -> None:
        adapter = DotsOcrAdapter()
        kwargs = adapter.get_vllm_kwargs(h100_gpu_config)
        assert kwargs["trust_remote_code"] is True
        assert kwargs["max_model_len"] == 16384

    def test_max_output_tokens(self) -> None:
        adapter = DotsOcrAdapter()
        assert adapter.get_max_output_tokens() == 8192


class TestNanonetsOcrAdapter:
    def test_model_id(self) -> None:
        adapter = NanonetsOcrAdapter()
        assert adapter.get_model_id() == "nanonets/Nanonets-OCR-s"

    def test_build_prompt(self, sample_image: MagicMock) -> None:
        adapter = NanonetsOcrAdapter()
        result = adapter.build_prompt(sample_image)
        assert "messages" in result
        assert result["images"] == [sample_image]

    def test_parse_output(self) -> None:
        adapter = NanonetsOcrAdapter()
        doc = adapter.parse_output("some text")
        assert doc.markdown == "some text"

    def test_vllm_kwargs(self, h100_gpu_config: GPUConfig) -> None:
        adapter = NanonetsOcrAdapter()
        kwargs = adapter.get_vllm_kwargs(h100_gpu_config)
        assert kwargs["trust_remote_code"] is True
        assert kwargs["max_model_len"] == 8192


class TestDeepSeekOcrAdapter:
    def test_model_id(self) -> None:
        adapter = DeepSeekOcrAdapter()
        assert adapter.get_model_id() == "deepseek-ai/DeepSeek-OCR-3B"

    def test_build_prompt(self, sample_image: MagicMock) -> None:
        adapter = DeepSeekOcrAdapter()
        result = adapter.build_prompt(sample_image)
        assert "messages" in result
        assert result["images"] == [sample_image]

    def test_parse_output(self) -> None:
        adapter = DeepSeekOcrAdapter()
        doc = adapter.parse_output("some text")
        assert doc.markdown == "some text"

    def test_vllm_kwargs(self, h100_gpu_config: GPUConfig) -> None:
        adapter = DeepSeekOcrAdapter()
        kwargs = adapter.get_vllm_kwargs(h100_gpu_config)
        assert kwargs["trust_remote_code"] is True


class TestOlmOcrAdapter:
    def test_model_id(self) -> None:
        adapter = OlmOcrAdapter()
        assert adapter.get_model_id() == "allenai/olmOCR2-7B-0225-preview"

    def test_build_prompt(self, sample_image: MagicMock) -> None:
        adapter = OlmOcrAdapter()
        result = adapter.build_prompt(sample_image)
        assert "messages" in result
        assert result["images"] == [sample_image]

    def test_parse_output(self) -> None:
        adapter = OlmOcrAdapter()
        doc = adapter.parse_output("some text")
        assert doc.markdown == "some text"

    def test_vllm_kwargs(self, h100_gpu_config: GPUConfig) -> None:
        adapter = OlmOcrAdapter()
        kwargs = adapter.get_vllm_kwargs(h100_gpu_config)
        assert kwargs["trust_remote_code"] is True
        assert kwargs["max_model_len"] == 16384

    def test_max_output_tokens(self) -> None:
        adapter = OlmOcrAdapter()
        assert adapter.get_max_output_tokens() == 8192

    def test_vision_tokens_qwen2vl_style(self) -> None:
        adapter = OlmOcrAdapter()
        # 1024x768 with 28px patches => 36*27 = 972
        assert adapter.estimate_vision_tokens(1024, 768) == 36 * 27


class TestChandraOcrAdapter:
    def test_model_id(self) -> None:
        adapter = ChandraOcrAdapter()
        assert adapter.get_model_id() == "amaai-lab/Chandra-OCR-9B"

    def test_build_prompt(self, sample_image: MagicMock) -> None:
        adapter = ChandraOcrAdapter()
        result = adapter.build_prompt(sample_image)
        assert "messages" in result
        assert result["images"] == [sample_image]

    def test_parse_output(self) -> None:
        adapter = ChandraOcrAdapter()
        doc = adapter.parse_output("some text")
        assert doc.markdown == "some text"

    def test_vllm_kwargs(self, h100_gpu_config: GPUConfig) -> None:
        adapter = ChandraOcrAdapter()
        kwargs = adapter.get_vllm_kwargs(h100_gpu_config)
        assert kwargs["trust_remote_code"] is True

    def test_max_output_tokens(self) -> None:
        adapter = ChandraOcrAdapter()
        assert adapter.get_max_output_tokens() == 8192

    def test_vision_tokens_qwen2vl_style(self) -> None:
        adapter = ChandraOcrAdapter()
        assert adapter.estimate_vision_tokens(1024, 768) == 36 * 27


# ─── Quantization Tests ───


class TestQuantization:
    def test_bf16_returns_none(self) -> None:
        from vlm_ocr_bench.models.quantization import get_quantization_config

        assert get_quantization_config(PrecisionMode.BF16) is None

    def test_fp16_returns_none(self) -> None:
        from vlm_ocr_bench.models.quantization import get_quantization_config

        assert get_quantization_config(PrecisionMode.FP16) is None

    def test_fp8_config(self) -> None:
        from vlm_ocr_bench.models.quantization import get_quantization_config

        config = get_quantization_config(PrecisionMode.FP8)
        assert config is not None
        assert config["load_in_8bit"] is True

    def test_fp4_config(self) -> None:
        from vlm_ocr_bench.models.quantization import get_quantization_config

        config = get_quantization_config(PrecisionMode.FP4)
        assert config is not None
        assert config["load_in_4bit"] is True
        assert config["bnb_4bit_quant_type"] == "nf4"

    def test_validate_fp8_on_unsupported_gpu(self) -> None:
        from vlm_ocr_bench.hardware.detector import GPUInfo
        from vlm_ocr_bench.models.quantization import validate_quantization

        gpu = GPUInfo(name="Old GPU", compute_capability=(7, 0))
        issues = validate_quantization(PrecisionMode.FP8, gpu)
        assert len(issues) == 1
        assert "FP8" in issues[0]

    def test_validate_fp4_on_hopper(self) -> None:
        from vlm_ocr_bench.hardware.detector import GPUInfo
        from vlm_ocr_bench.models.quantization import validate_quantization

        gpu = GPUInfo(name="H100", compute_capability=(9, 0))
        issues = validate_quantization(PrecisionMode.FP4, gpu)
        assert len(issues) == 1
        assert "fp4" in issues[0].lower() or "FP4" in issues[0]

    def test_validate_bf16_always_passes(self) -> None:
        from vlm_ocr_bench.hardware.detector import GPUInfo
        from vlm_ocr_bench.models.quantization import validate_quantization

        gpu = GPUInfo(name="Any GPU", compute_capability=(7, 0))
        issues = validate_quantization(PrecisionMode.BF16, gpu)
        assert issues == []


# ─── Loader Tests ───


class TestLoader:
    def _make_model_config(self) -> ModelConfig:
        return ModelConfig(
            name="test_model",
            hf_model_id="org/test-model",
            params_billion=3.0,
            tier="compact",
            adapter="DotsOcrAdapter",
            supported_resolutions=[1024],
        )

    def test_load_model_mocked(self, h100_gpu_config: GPUConfig) -> None:
        """Test loader with mocked transformers to verify trust_remote_code."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor

        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"

        with patch.dict(
            "sys.modules",
            {"transformers": mock_transformers, "torch": mock_torch},
        ):
            from vlm_ocr_bench.models.loader import load_model_and_tokenizer

            result = load_model_and_tokenizer(self._make_model_config(), h100_gpu_config)

            # Verify trust_remote_code=True passed to model loader
            model_call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args
            assert model_call_kwargs[1]["trust_remote_code"] is True

            # Verify trust_remote_code=True passed to tokenizer
            tokenizer_call_kwargs = mock_transformers.AutoTokenizer.from_pretrained.call_args
            assert tokenizer_call_kwargs[1]["trust_remote_code"] is True

            assert result.model is mock_model
            assert result.tokenizer is mock_tokenizer
            assert result.processor is mock_processor

    def test_load_model_no_processor(self, h100_gpu_config: GPUConfig) -> None:
        """Test that missing processor is handled gracefully."""
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.side_effect = OSError("no processor")

        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        with patch.dict(
            "sys.modules",
            {"transformers": mock_transformers, "torch": mock_torch},
        ):
            from vlm_ocr_bench.models.loader import load_model_and_tokenizer

            result = load_model_and_tokenizer(self._make_model_config(), h100_gpu_config)
            assert result.processor is None
