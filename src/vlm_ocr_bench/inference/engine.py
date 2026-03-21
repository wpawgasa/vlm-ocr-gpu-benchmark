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
    """Information about a loaded vLLM engine.

    Note: ``flash_attn_version`` and ``max_batch_size`` are not populated by
    vLLM at engine-init time and remain at their default sentinel values.
    They are kept for schema compatibility but should not be relied upon.
    """

    model_load_time_s: float = 0.0
    gpu_memory_allocated_gb: float = 0.0
    gpu_memory_reserved_gb: float = 0.0
    flash_attn_version: str = "unknown"  # not queryable from vLLM LLM API
    actual_precision: str = "unknown"
    kv_cache_dtype: str = "auto"
    max_batch_size: int = 0  # not queryable from vLLM LLM API
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

        # Build vLLM inputs, forwarding multi_modal_data per-prompt so images
        # are actually passed to the vision encoder (not silently dropped).
        vllm_inputs: list[Any] = []
        for inp in inputs:
            mm_data = inp.get("multi_modal_data")
            if "messages" in inp:
                # Apply chat template to convert messages to a prompt string
                messages = inp["messages"]
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                entry: dict[str, Any] = {"prompt": prompt}
            else:
                entry = {"prompt": inp.get("prompt", "")}
            if mm_data is not None:
                entry["multi_modal_data"] = mm_data
            vllm_inputs.append(entry)

        # vLLM generate
        outputs = self._engine.generate(
            vllm_inputs,
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

        # Build results using per-request metrics from vLLM when available.
        # vLLM's RequestOutput.metrics (RequestStateStats) provides
        # first_token_latency, arrival_time, and last_token_ts for each request.
        # Fall back to dividing total GPU time evenly when metrics are absent.
        per_request_ms_fallback = gpu_elapsed_ms / max(len(outputs), 1)

        for output in outputs:
            generated_text = output.outputs[0].text if output.outputs else ""
            num_output_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
            num_input_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0

            # Extract per-request timing from vLLM metrics.
            # RequestStateStats provides:
            #   first_token_latency: wall-clock TTFT in seconds
            #   first_token_ts / last_token_ts: monotonic timestamps
            ttft_ms = 0.0
            generation_ms = 0.0
            total_ms = per_request_ms_fallback
            metrics = getattr(output, "metrics", None)
            if metrics is not None:
                ftl = getattr(metrics, "first_token_latency", 0.0) or 0.0
                first_ts = getattr(metrics, "first_token_ts", 0.0) or 0.0
                last_ts = getattr(metrics, "last_token_ts", 0.0) or 0.0

                if ftl > 0:
                    ttft_ms = ftl * 1000.0  # seconds → ms
                if first_ts > 0 and last_ts > first_ts:
                    generation_ms = (last_ts - first_ts) * 1000.0
                if ttft_ms > 0:
                    total_ms = ttft_ms + generation_ms

            tps = (
                num_output_tokens / (total_ms / 1000.0)
                if total_ms > 0 and num_output_tokens > 0
                else 0.0
            )

            results.append(
                GenerationResult(
                    output_text=generated_text,
                    num_output_tokens=num_output_tokens,
                    num_input_tokens=num_input_tokens,
                    ttft_ms=ttft_ms,
                    generation_time_ms=generation_ms if generation_ms > 0 else total_ms,
                    total_time_ms=total_ms,
                    tokens_per_second=tps,
                )
            )

        return results

    def get_memory_stats(self) -> MemoryStats:
        """Get current GPU memory statistics.

        Uses pynvml to query device-level memory, which works even when
        vLLM runs the engine in a separate subprocess.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = info.used / (1024**3)
            total_gb = info.total / (1024**3)
            pynvml.nvmlShutdown()
            return MemoryStats(
                allocated_gb=used_gb,
                reserved_gb=total_gb,
                peak_allocated_gb=used_gb,  # nvml doesn't track peak; use current
            )
        except Exception:
            logger.debug("memory_stats_unavailable_nvml")
        # Fallback to torch (works when engine is in-process)
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
            logger.debug("memory_stats_unavailable_torch")
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
            logger.debug("vllm_engine_shutdown_cache_clear_failed", exc_info=True)

        logger.info("vllm_engine_shutdown_complete")
