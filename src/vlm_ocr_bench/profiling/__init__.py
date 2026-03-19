"""Profiling & monitoring: background GPU monitoring with fail-safe design."""

from vlm_ocr_bench.profiling.events import (
    ThrottleEvent,
    detect_throttle_event,
    is_critical_throttle,
    parse_throttle_reasons,
    summarize_throttle_events,
)
from vlm_ocr_bench.profiling.monitor import (
    GPUMonitor,
    ProfilingResult,
    ProfilingSummary,
)
from vlm_ocr_bench.profiling.nvidia_smi import (
    NvidiaSmiSample,
    is_nvidia_smi_available,
    query_nvidia_smi,
)

__all__ = [
    "GPUMonitor",
    "NvidiaSmiSample",
    "ProfilingResult",
    "ProfilingSummary",
    "ThrottleEvent",
    "detect_throttle_event",
    "is_critical_throttle",
    "is_nvidia_smi_available",
    "parse_throttle_reasons",
    "query_nvidia_smi",
    "summarize_throttle_events",
]
