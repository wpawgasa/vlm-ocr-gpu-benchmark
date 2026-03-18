"""Unit tests for the data pipeline module (no GPU, no HuggingFace downloads)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from vlm_ocr_bench.config.schema import OutputFormat
from vlm_ocr_bench.data.datasets import (
    DatasetSpec,
    get_dataset,
    get_loader,
    list_datasets,
)
from vlm_ocr_bench.data.image_utils import get_image_stats, preprocess_image
from vlm_ocr_bench.data.omnidocbench import (
    DocSample,
    GroundTruth,
    GTElement,
    _normalize_text,
)
from vlm_ocr_bench.data.tokenization import PROMPT_TEMPLATES, build_inference_input

# ─── Fixtures ───


@pytest.fixture
def rgb_image() -> Image.Image:
    """Create a simple RGB test image."""
    return Image.new("RGB", (800, 600), color=(128, 128, 128))


@pytest.fixture
def rgba_image() -> Image.Image:
    """Create an RGBA test image (has alpha channel)."""
    return Image.new("RGBA", (800, 600), color=(128, 128, 128, 255))


@pytest.fixture
def wide_image() -> Image.Image:
    """Create a wide landscape image."""
    return Image.new("RGB", (2000, 1000), color=(200, 200, 200))


@pytest.fixture
def tall_image() -> Image.Image:
    """Create a tall portrait image."""
    return Image.new("RGB", (1000, 2000), color=(200, 200, 200))


@pytest.fixture
def square_image() -> Image.Image:
    """Create a square image."""
    return Image.new("RGB", (1024, 1024), color=(200, 200, 200))


@pytest.fixture
def thai_docs_dir(tmp_path: Path) -> Path:
    """Create a temp directory with Thai doc images and annotations."""
    data_dir = tmp_path / "thai_docs"
    data_dir.mkdir()

    # Create sample images
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(str(data_dir / f"doc_{i}.png"))

    # Create annotations
    annotations = {
        "doc_0": {
            "doc_type": "financial",
            "language": "th",
            "ground_truth": "ข้อมูลทางการเงิน",
            "has_tables": True,
        },
        "doc_1": {
            "doc_type": "legal",
            "language": "th",
            "ground_truth": "สัญญา",
        },
    }
    (data_dir / "annotations.json").write_text(
        json.dumps(annotations, ensure_ascii=False), encoding="utf-8"
    )
    return data_dir


# ─── DatasetSpec Tests ───


class TestDatasetSpec:
    def test_basic_creation(self) -> None:
        spec = DatasetSpec(name="test", loader_cls="TestLoader")
        assert spec.name == "test"
        assert spec.loader_cls == "TestLoader"
        assert spec.num_samples is None
        assert spec.source == "huggingface"

    def test_full_creation(self) -> None:
        spec = DatasetSpec(
            name="test",
            loader_cls="TestLoader",
            hf_id="org/dataset",
            num_samples=100,
            split="test",
            doc_types=["academic", "financial"],
        )
        assert spec.hf_id == "org/dataset"
        assert spec.num_samples == 100
        assert len(spec.doc_types) == 2

    def test_frozen(self) -> None:
        spec = DatasetSpec(name="test", loader_cls="TestLoader")
        with pytest.raises(AttributeError):
            spec.name = "changed"  # type: ignore[misc]


# ─── Dataset Registry Tests ───


class TestDatasetRegistry:
    def test_list_datasets(self) -> None:
        datasets = list_datasets()
        assert len(datasets) == 5
        assert "omnidocbench_v1.5" in datasets
        assert "olmocr_bench" in datasets
        assert "real5_omnidocbench" in datasets
        assert "thai_docs_500" in datasets
        assert "docmatix_50k" in datasets

    def test_get_dataset(self) -> None:
        spec = get_dataset("omnidocbench_v1.5")
        assert spec.hf_id == "opendatalab/OmniDocBench"
        assert spec.num_samples == 1355
        assert spec.loader_cls == "OmniDocBenchLoader"
        assert len(spec.doc_types) == 9

    def test_get_dataset_invalid(self) -> None:
        with pytest.raises(KeyError, match="Unknown dataset"):
            get_dataset("nonexistent")

    def test_omnidocbench_spec(self) -> None:
        spec = get_dataset("omnidocbench_v1.5")
        assert spec.split == "test"
        assert "academic" in spec.doc_types

    def test_real5_spec(self) -> None:
        spec = get_dataset("real5_omnidocbench")
        assert spec.num_samples == 6775
        assert len(spec.scenarios) == 5
        assert "scanning" in spec.scenarios

    def test_thai_docs_spec(self) -> None:
        spec = get_dataset("thai_docs_500")
        assert spec.source == "local"
        assert spec.path == "data/thai_fin_legal/"

    def test_docmatix_spec(self) -> None:
        spec = get_dataset("docmatix_50k")
        assert spec.sample_size == 50000
        assert spec.stratify_by == "doc_type"

    def test_all_specs_have_loader_cls(self) -> None:
        for name in list_datasets():
            spec = get_dataset(name)
            assert spec.loader_cls, f"Dataset {name} has empty loader_cls"


# ─── GetLoader Tests ───


class TestGetLoader:
    def test_get_omnidocbench_loader(self) -> None:
        from vlm_ocr_bench.data.omnidocbench import OmniDocBenchLoader

        loader = get_loader("omnidocbench_v1.5")
        assert isinstance(loader, OmniDocBenchLoader)

    def test_get_olmocr_loader(self) -> None:
        from vlm_ocr_bench.data.olmocr_bench import OlmOCRBenchLoader

        loader = get_loader("olmocr_bench")
        assert isinstance(loader, OlmOCRBenchLoader)

    def test_get_real5_loader(self) -> None:
        from vlm_ocr_bench.data.real5_omnidoc import Real5OmniDocLoader

        loader = get_loader("real5_omnidocbench")
        assert isinstance(loader, Real5OmniDocLoader)

    def test_get_thai_loader(self) -> None:
        from vlm_ocr_bench.data.thai_docs import ThaiDocsLoader

        loader = get_loader("thai_docs_500")
        assert isinstance(loader, ThaiDocsLoader)

    def test_get_docmatix_loader(self) -> None:
        from vlm_ocr_bench.data.docmatix import DocmatixLoader

        loader = get_loader("docmatix_50k")
        assert isinstance(loader, DocmatixLoader)

    def test_invalid_dataset(self) -> None:
        with pytest.raises(KeyError):
            get_loader("nonexistent")


# ─── DocSample Tests ───


class TestDocSample:
    def test_basic_creation(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (100, 100))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        sample = DocSample(
            sample_id="test_001",
            image_path=img_path,
            doc_type="academic",
        )
        assert sample.sample_id == "test_001"
        assert sample.doc_type == "academic"
        assert sample.language == "en"
        assert not sample.has_tables

    def test_lazy_image_loading(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (200, 150))
        img_path = tmp_path / "lazy.png"
        img.save(str(img_path))

        sample = DocSample(
            sample_id="lazy_001",
            image_path=img_path,
            doc_type="financial",
        )
        # Accessing .image should load the PIL image
        loaded = sample.image
        assert loaded.size == (200, 150)


# ─── GroundTruth Tests ───


class TestGroundTruth:
    def test_basic(self) -> None:
        gt = GroundTruth(markdown="# Title\nSome text")
        assert gt.markdown == "# Title\nSome text"
        assert gt.elements == []
        assert gt.reading_order == []

    def test_with_elements(self) -> None:
        elem = GTElement(
            element_id=0,
            type="title",
            content="Title",
            bbox=(0.0, 0.0, 1.0, 0.1),
        )
        gt = GroundTruth(
            markdown="# Title",
            elements=[elem],
            reading_order=[0],
        )
        assert len(gt.elements) == 1
        assert gt.elements[0].type == "title"


# ─── GTElement Tests ───


class TestGTElement:
    def test_basic(self) -> None:
        elem = GTElement(
            element_id=1,
            type="table",
            content="| a | b |",
            bbox=(0.1, 0.2, 0.9, 0.8),
        )
        assert elem.element_id == 1
        assert elem.type == "table"
        assert elem.attributes == {}

    def test_with_attributes(self) -> None:
        elem = GTElement(
            element_id=0,
            type="formula",
            content="E=mc^2",
            bbox=(0.0, 0.0, 0.5, 0.1),
            attributes={"inline": True},
        )
        assert elem.attributes["inline"] is True


# ─── Unicode Normalization Tests ───


class TestNormalization:
    def test_nfkc_normalization(self) -> None:
        # Full-width characters should be normalized
        text = "\uff21\uff22\uff23"  # ABC in full-width
        assert _normalize_text(text) == "ABC"

    def test_normal_text_unchanged(self) -> None:
        text = "Hello World"
        assert _normalize_text(text) == "Hello World"

    def test_thai_text_preserved(self) -> None:
        text = "สวัสดีครับ"
        assert _normalize_text(text) == "สวัสดีครับ"


# ─── ThaiDocsLoader Tests ───


class TestThaiDocsLoader:
    def test_load_from_directory(self, thai_docs_dir: Path) -> None:
        from vlm_ocr_bench.data.thai_docs import ThaiDocsLoader

        spec = DatasetSpec(
            name="test_thai",
            loader_cls="ThaiDocsLoader",
            source="local",
            path=str(thai_docs_dir),
        )
        loader = ThaiDocsLoader(spec)
        samples = loader.load()
        assert len(samples) == 3

    def test_annotations_loaded(self, thai_docs_dir: Path) -> None:
        from vlm_ocr_bench.data.thai_docs import ThaiDocsLoader

        spec = DatasetSpec(
            name="test_thai",
            loader_cls="ThaiDocsLoader",
            source="local",
            path=str(thai_docs_dir),
        )
        loader = ThaiDocsLoader(spec)
        loader.load()

        # doc_0 has annotations
        gt = loader.get_ground_truth("doc_0")
        assert "การเงิน" in gt.markdown

    def test_filter_by_doc_type(self, thai_docs_dir: Path) -> None:
        from vlm_ocr_bench.data.thai_docs import ThaiDocsLoader

        spec = DatasetSpec(
            name="test_thai",
            loader_cls="ThaiDocsLoader",
            source="local",
            path=str(thai_docs_dir),
        )
        loader = ThaiDocsLoader(spec)
        loader.load()
        financial = loader.get_samples(doc_types=["financial"])
        assert len(financial) == 1
        assert financial[0].doc_type == "financial"

    def test_max_samples(self, thai_docs_dir: Path) -> None:
        from vlm_ocr_bench.data.thai_docs import ThaiDocsLoader

        spec = DatasetSpec(
            name="test_thai",
            loader_cls="ThaiDocsLoader",
            source="local",
            path=str(thai_docs_dir),
        )
        loader = ThaiDocsLoader(spec)
        loader.load()
        samples = loader.get_samples(max_samples=2)
        assert len(samples) == 2

    def test_missing_directory(self, tmp_path: Path) -> None:
        from vlm_ocr_bench.data.thai_docs import ThaiDocsLoader

        spec = DatasetSpec(
            name="test_thai",
            loader_cls="ThaiDocsLoader",
            source="local",
            path=str(tmp_path / "nonexistent"),
        )
        loader = ThaiDocsLoader(spec)
        samples = loader.load()
        assert samples == []

    def test_missing_ground_truth(self, thai_docs_dir: Path) -> None:
        from vlm_ocr_bench.data.thai_docs import ThaiDocsLoader

        spec = DatasetSpec(
            name="test_thai",
            loader_cls="ThaiDocsLoader",
            source="local",
            path=str(thai_docs_dir),
        )
        loader = ThaiDocsLoader(spec)
        loader.load()
        # doc_2 has no annotations
        with pytest.raises(KeyError, match="No ground truth"):
            loader.get_ground_truth("doc_2")


# ─── Image Utils Tests ───


class TestPreprocessImage:
    def test_resize_landscape(self, wide_image: Image.Image) -> None:
        result = preprocess_image(wide_image, target_resolution=1024)
        assert result.size[0] == 1024  # width is longest
        assert result.size[1] == 512  # aspect ratio preserved

    def test_resize_portrait(self, tall_image: Image.Image) -> None:
        result = preprocess_image(tall_image, target_resolution=1024)
        assert result.size[1] == 1024  # height is longest
        assert result.size[0] == 512

    def test_already_correct_size(self, square_image: Image.Image) -> None:
        result = preprocess_image(square_image, target_resolution=1024)
        assert result.size == (1024, 1024)

    def test_rgba_to_rgb(self, rgba_image: Image.Image) -> None:
        result = preprocess_image(rgba_image, target_resolution=1024)
        assert result.mode == "RGB"

    def test_padding_to_square(self, wide_image: Image.Image) -> None:
        result = preprocess_image(wide_image, target_resolution=1024, requires_padding=True)
        assert result.size[0] == result.size[1]  # square
        assert result.size[0] == 1024

    def test_no_padding_by_default(self, wide_image: Image.Image) -> None:
        result = preprocess_image(wide_image, target_resolution=1024)
        assert result.size[0] != result.size[1]  # not square

    def test_rgb_stays_rgb(self, rgb_image: Image.Image) -> None:
        result = preprocess_image(rgb_image, target_resolution=1024)
        assert result.mode == "RGB"


class TestGetImageStats:
    def test_basic_stats(self, rgb_image: Image.Image) -> None:
        stats = get_image_stats(rgb_image)
        assert stats["width"] == 800
        assert stats["height"] == 600
        assert stats["channels"] == 3
        assert stats["mode"] == "RGB"

    def test_rgba_stats(self, rgba_image: Image.Image) -> None:
        stats = get_image_stats(rgba_image)
        assert stats["channels"] == 4
        assert stats["mode"] == "RGBA"


# ─── Prompt Templates Tests ───


class TestPromptTemplates:
    def test_markdown_template_exists(self) -> None:
        assert "document_parse_md" in PROMPT_TEMPLATES

    def test_html_template_exists(self) -> None:
        assert "document_parse_html" in PROMPT_TEMPLATES

    def test_text_only_template_exists(self) -> None:
        assert "ocr_text_only" in PROMPT_TEMPLATES

    def test_templates_are_nonempty(self) -> None:
        for key, value in PROMPT_TEMPLATES.items():
            assert value, f"Template {key} is empty"


# ─── Build Inference Input Tests ───


class TestBuildInferenceInput:
    def test_completion_style_adapter(self, rgb_image: Image.Image) -> None:
        adapter = MagicMock()
        adapter.build_prompt.return_value = {
            "prompt": "Parse this document.",
            "images": [rgb_image],
        }

        result = build_inference_input(adapter, rgb_image, OutputFormat.MARKDOWN, resolution=1024)
        assert "prompt" in result
        assert "multi_modal_data" in result
        assert result["prompt"] == "Parse this document."

    def test_chat_style_adapter(self, rgb_image: Image.Image) -> None:
        messages = [{"role": "user", "content": "Parse this."}]
        adapter = MagicMock()
        adapter.build_prompt.return_value = {
            "messages": messages,
            "images": [rgb_image],
        }

        result = build_inference_input(adapter, rgb_image, OutputFormat.MARKDOWN, resolution=1024)
        assert result["prompt"] == messages

    def test_image_is_preprocessed(self, wide_image: Image.Image) -> None:
        adapter = MagicMock()
        adapter.build_prompt.return_value = {"prompt": "test", "images": []}

        build_inference_input(adapter, wide_image, OutputFormat.MARKDOWN, resolution=512)
        # The image passed to adapter should be preprocessed
        call_args = adapter.build_prompt.call_args
        processed_image = call_args[0][0]
        assert processed_image.size[0] == 512  # resized

    def test_padding_passed_through(self, wide_image: Image.Image) -> None:
        adapter = MagicMock()
        adapter.build_prompt.return_value = {"prompt": "test", "images": []}

        build_inference_input(
            adapter,
            wide_image,
            OutputFormat.MARKDOWN,
            resolution=512,
            requires_padding=True,
        )
        call_args = adapter.build_prompt.call_args
        processed_image = call_args[0][0]
        # With padding, image should be square
        assert processed_image.size[0] == processed_image.size[1]


# ─── Module Export Tests ───


class TestModuleExports:
    def test_data_module_imports(self) -> None:
        from vlm_ocr_bench.data import (
            DATASET_REGISTRY,
            PROMPT_TEMPLATES,
        )

        assert DATASET_REGISTRY is not None
        assert PROMPT_TEMPLATES is not None
