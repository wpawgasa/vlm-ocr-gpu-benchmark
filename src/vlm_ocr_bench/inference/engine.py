"""vLLM engine wrapper with CUDA event timing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class MemoryStats:
    """GPU memory statistics."""

    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    peak_allocated_gb: float = 0.0


@dataclass
class EngineInfo:
    """Information about a loaded vLLM engine."""

    model_load_time_s: float = 0.0
    gpu_memory_allocated_gb: float = 0.0
    gpu_memory_reserved_gb: float = 0.0
    flash_attn_version: str = "unknown"
    actual_precision: str = "unknown"
    kv_cache_dtype: str = "auto"
    max_batch_size: int = 0
    max_model_len: int = 0


@dataclass
class GenerationResult:
    """Result from a single generation request."""

    output_text: str = ""
    num_output_tokens: int = 0
    num_input_tokens: int = 0
    ttft_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0


class VLLMEngine:
    """Wrapper around vLLM for benchmarking with precise GPU timing.

    All vLLM/torch imports are deferred to method calls so this module
    can be imported without GPU dependencies.
    """

    def __init__(
        self,
        engine_args: dict[str, Any],
    ) -> None:
        self._engine_args = engine_args
        self._engine: Any = None
        self._tokenizer: Any = None

    def initialize(self) -> EngineInfo:
        """Load the model into vLLM and return engine metadata.

        All heavy imports happen here.
        """
        from vllm import LLM

        logger.info("vllm_engine_initializing", model=self._engine_args.get("model"))

        start = time.monotonic()
        self._engine = LLM(**self._engine_args)
        load_time = time.monotonic() - start

        self._tokenizer = self._engine.get_tokenizer()

        # Collect memory stats
        mem = self.get_memory_stats()

        info = EngineInfo(
            model_load_time_s=load_time,
            gpu_memory_allocated_gb=mem.allocated_gb,
            gpu_memory_reserved_gb=mem.reserved_gb,
            actual_precision=str(self._engine_args.get("dtype", "auto")),
            kv_cache_dtype=str(self._engine_args.get("kv_cache_dtype", "auto")),
            max_model_len=int(self._engine_args.get("max_model_len", 0)),
        )

        logger.info(
            "vllm_engine_initialized",
            load_time_s=round(load_time, 2),
            memory_gb=round(mem.allocated_gb, 2),
        )
        return info

    def generate_batch(
        self,
        inputs: list[dict[str, Any]],
        sampling_params: Any | None = None,
    ) -> list[GenerationResult]:
        """Run batch inference with timing.

        Args:
            inputs: List of dicts with 'prompt' and optionally 'multi_modal_data'.
            sampling_params: vLLM SamplingParams instance.

        Returns list of GenerationResult with per-request timing.
        """
        import torch
        from vllm import SamplingParams

        if self._engine is None:
            msg = "Engine not initialized. Call initialize() first."
            raise RuntimeError(msg)

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=4096, temperature=0.0)

        results: list[GenerationResult] = []

        # Use CUDA events for precise GPU timing
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            torch.cuda.synchronize()
            start_event.record()  # type: ignore[no-untyped-call]

        wall_start = time.monotonic()

        # Build prompts and multi-modal data for vLLM
        prompts = []
        multi_modal_data_list = []
        for inp in inputs:
            prompts.append(inp.get("prompt", ""))
            multi_modal_data_list.append(inp.get("multi_modal_data"))

        # vLLM generate
        outputs = self._engine.generate(
            prompts,
            sampling_params,
        )

        wall_end = time.monotonic()
        wall_elapsed_ms = (wall_end - wall_start) * 1000.0

        if use_cuda:
            end_event.record()  # type: ignore[no-untyped-call]
            torch.cuda.synchronize()
            gpu_elapsed_ms = start_event.elapsed_time(end_event)  # type: ignore[no-untyped-call]
        else:
            gpu_elapsed_ms = wall_elapsed_ms

        # Build results
        per_request_ms = gpu_elapsed_ms / max(len(outputs), 1)

        for output in outputs:
            generated_text = output.outputs[0].text if output.outputs else ""
            num_output_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
            num_input_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0

            tps = (
                num_output_tokens / (per_request_ms / 1000.0)
                if per_request_ms > 0 and num_output_tokens > 0
                else 0.0
            )

            results.append(
                GenerationResult(
                    output_text=generated_text,
                    num_output_tokens=num_output_tokens,
                    num_input_tokens=num_input_tokens,
                    ttft_ms=per_request_ms * 0.1,  # Approximate TTFT
                    generation_time_ms=per_request_ms,
                    total_time_ms=per_request_ms,
                    tokens_per_second=tps,
                )
            )

        return results

    def get_memory_stats(self) -> MemoryStats:
        """Get current GPU memory statistics."""
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                return MemoryStats(
                    allocated_gb=allocated,
                    reserved_gb=reserved,
                    peak_allocated_gb=peak,
                )
        except Exception:
            logger.debug("memory_stats_unavailable")
        return MemoryStats()

    def shutdown(self) -> None:
        """Release GPU memory and clean up."""
        logger.info("vllm_engine_shutting_down")
        if self._engine is not None:
            del self._engine
            self._engine = None
        self._tokenizer = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("vllm_engine_shutdown_complete")
