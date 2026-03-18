"""Inference benchmark (Phase A): vLLM engine, runner, workload, and metrics."""

from vlm_ocr_bench.inference.engine import EngineInfo, GenerationResult, MemoryStats, VLLMEngine
from vlm_ocr_bench.inference.metrics import InferenceMetrics, compute_metrics
from vlm_ocr_bench.inference.runner import (
    InferenceBenchmarkResult,
    InferenceBenchmarkRunner,
    SingleConfigResult,
)
from vlm_ocr_bench.inference.vllm_config import build_vllm_engine_args, validate_vllm_config
from vlm_ocr_bench.inference.workload import build_workload_batch

__all__ = [
    "EngineInfo",
    "GenerationResult",
    "InferenceBenchmarkResult",
    "InferenceBenchmarkRunner",
    "InferenceMetrics",
    "MemoryStats",
    "SingleConfigResult",
    "VLLMEngine",
    "build_vllm_engine_args",
    "build_workload_batch",
    "compute_metrics",
    "validate_vllm_config",
]
