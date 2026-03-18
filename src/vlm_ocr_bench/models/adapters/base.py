"""ABC interface and dataclasses for OCR model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from vlm_ocr_bench.config.schema import GPUConfig, OutputFormat


@dataclass
class DocElement:
    """A detected document element."""

    type: str  # "text" | "table" | "formula" | "title" | "figure" | ...
    content: str  # extracted content (Markdown for tables, LaTeX for formulas)
    bbox: tuple[float, float, float, float] | None = None  # normalized x1,y1,x2,y2
    confidence: float | None = None


@dataclass
class ParsedDocument:
    """Normalized document parse output."""

    raw_text: str
    markdown: str
    html: str | None = None
    elements: list[DocElement] = field(default_factory=list)
    parse_time_ms: float = 0.0
    token_count: int = 0


class OCRModelAdapter(ABC):
    """Adapter interface normalizing prompt/postprocess across OCR VLMs."""

    @abstractmethod
    def get_model_id(self) -> str:
        """HuggingFace model ID."""

    @abstractmethod
    def build_prompt(
        self,
        image: Image.Image,
        output_format: OutputFormat = "markdown",  # type: ignore[assignment]
        task: str = "full_page_parse",
    ) -> dict[str, Any]:
        """Build model-specific prompt.

        Returns dict with keys like 'messages', 'images', 'prompt'.
        """

    @abstractmethod
    def parse_output(self, raw_output: str) -> ParsedDocument:
        """Parse raw model output into structured ParsedDocument."""

    @abstractmethod
    def get_vllm_kwargs(self, gpu_config: GPUConfig) -> dict[str, Any]:
        """Return model-specific vLLM engine kwargs."""

    def get_supported_resolutions(self) -> list[int]:
        """Resolutions this model handles well."""
        return [1024, 1536, 2048]

    def get_max_output_tokens(self) -> int:
        """Default max tokens for full-page parsing."""
        return 4096

    def estimate_vision_tokens(self, width: int, height: int) -> int:
        """Estimate number of vision tokens for given image dimensions."""
        patch_size = 14
        return (width // patch_size) * (height // patch_size)
