"""nvidia-smi polling fallback for GPU monitoring."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class NvidiaSmiSample:
    """A single sample parsed from nvidia-smi output."""

    gpu_temp_c: float = 0.0
    power_w: float = 0.0
    gpu_util_pct: float = 0.0
    mem_util_pct: float = 0.0
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0


_QUERY_FIELDS = (
    "temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total"
)


def query_nvidia_smi(device_id: int = 0) -> NvidiaSmiSample | None:
    """Query GPU metrics via nvidia-smi CLI.

    Returns NvidiaSmiSample or None if nvidia-smi fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                f"--query-gpu={_QUERY_FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            logger.debug("nvidia_smi_query_failed", stderr=result.stderr.strip())
            return None

        values = result.stdout.strip().split(",")
        if len(values) < 6:
            return None

        return NvidiaSmiSample(
            gpu_temp_c=_parse_float(values[0]),
            power_w=_parse_float(values[1]),
            gpu_util_pct=_parse_float(values[2]),
            mem_util_pct=_parse_float(values[3]),
            mem_used_mb=_parse_float(values[4]),
            mem_total_mb=_parse_float(values[5]),
        )
    except FileNotFoundError:
        logger.debug("nvidia_smi_not_found")
        return None
    except subprocess.TimeoutExpired:
        logger.debug("nvidia_smi_timeout")
        return None
    except Exception:
        logger.debug("nvidia_smi_error", exc_info=True)
        return None


def is_nvidia_smi_available() -> bool:
    """Check if nvidia-smi is available on the system."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _parse_float(value: str) -> float:
    """Parse a float from nvidia-smi output, handling 'N/A' and whitespace."""
    v = value.strip()
    if not v or v.lower() in ("n/a", "[n/a]", "not supported"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0
