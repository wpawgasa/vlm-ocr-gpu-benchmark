"""GPU clock management: lock, reset, and query."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


def lock_gpu_clocks(gpu_id: int = 0, max_mhz: int | None = None) -> bool:
    """Lock GPU SM clocks for stable benchmarking.

    Args:
        gpu_id: GPU device index.
        max_mhz: Target clock speed. None = use max supported clock.

    Returns:
        True if clocks were locked successfully, False otherwise.
        Requires root/sudo permissions.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        if max_mhz is None:
            max_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)

        # Lock SM clocks to target frequency
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, max_mhz, max_mhz)
        logger.info("gpu_clocks_locked", gpu_id=gpu_id, clock_mhz=max_mhz)

        pynvml.nvmlShutdown()
        return True

    except PermissionError:
        logger.error(
            "gpu_clock_lock_permission_denied",
            gpu_id=gpu_id,
            hint="Clock locking requires root/sudo",
        )
        return False
    except Exception:
        logger.error("gpu_clock_lock_failed", gpu_id=gpu_id, exc_info=True)
        return False


def reset_gpu_clocks(gpu_id: int = 0) -> bool:
    """Reset GPU clocks to default (unlocked) state.

    Returns:
        True if clocks were reset successfully, False otherwise.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        logger.info("gpu_clocks_reset", gpu_id=gpu_id)
        pynvml.nvmlShutdown()
        return True

    except PermissionError:
        logger.error(
            "gpu_clock_reset_permission_denied",
            gpu_id=gpu_id,
            hint="Clock reset requires root/sudo",
        )
        return False
    except Exception:
        logger.error("gpu_clock_reset_failed", gpu_id=gpu_id, exc_info=True)
        return False


def get_clock_info(gpu_id: int = 0) -> dict[str, Any]:
    """Query current GPU clock speeds and throttle reasons.

    Returns a dict with current SM/memory clocks and throttle info.
    Returns empty dict if query fails.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        max_sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        max_mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)

        throttle_reasons: list[str] = []
        try:
            reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            reason_map = {
                pynvml.nvmlClocksThrottleReasonGpuIdle: "gpu_idle",
                pynvml.nvmlClocksThrottleReasonSwPowerCap: "sw_power_cap",
                pynvml.nvmlClocksThrottleReasonHwSlowdown: "hw_slowdown",
                pynvml.nvmlClocksThrottleReasonSwThermalSlowdown: "sw_thermal_slowdown",
                pynvml.nvmlClocksThrottleReasonHwThermalSlowdown: "hw_thermal_slowdown",
            }
            for bit, name in reason_map.items():
                if reasons & bit:
                    throttle_reasons.append(name)
        except Exception:
            pass

        info = {
            "sm_clock_mhz": sm_clock,
            "mem_clock_mhz": mem_clock,
            "max_sm_clock_mhz": max_sm_clock,
            "max_mem_clock_mhz": max_mem_clock,
            "throttle_reasons": throttle_reasons,
        }

        pynvml.nvmlShutdown()
        return info

    except Exception:
        logger.warning("get_clock_info_failed", gpu_id=gpu_id, exc_info=True)
        return {}
