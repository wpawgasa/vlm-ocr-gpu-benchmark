"""Thermal throttle detection and event logging."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

# NVML throttle reason bit flags
THROTTLE_REASONS: dict[int, str] = {
    0x0000_0001: "GpuIdle",
    0x0000_0002: "ApplicationsClocks",
    0x0000_0004: "SwPowerCap",
    0x0000_0008: "HwSlowdown",
    0x0000_0010: "SyncBoost",
    0x0000_0020: "SwThermalSlowdown",
    0x0000_0040: "HwThermalSlowdown",
    0x0000_0080: "HwPowerBrakeSlowdown",
}

# Reason codes that indicate problematic throttling
CRITICAL_THROTTLE_MASK = (
    0x0000_0004  # SwPowerCap
    | 0x0000_0008  # HwSlowdown
    | 0x0000_0020  # SwThermalSlowdown
    | 0x0000_0040  # HwThermalSlowdown
    | 0x0000_0080  # HwPowerBrakeSlowdown
)


@dataclass
class ThrottleEvent:
    """A single throttle event with timestamp and reason."""

    timestamp_s: float
    reason_code: int
    reasons: list[str] = field(default_factory=list)
    is_critical: bool = False


def parse_throttle_reasons(reason_code: int) -> list[str]:
    """Parse NVML throttle reason bitmask into human-readable strings."""
    reasons: list[str] = []
    for bit, name in THROTTLE_REASONS.items():
        if reason_code & bit:
            reasons.append(name)
    return reasons


def is_critical_throttle(reason_code: int) -> bool:
    """Check if the throttle reason indicates performance-degrading throttling."""
    return bool(reason_code & CRITICAL_THROTTLE_MASK)


def detect_throttle_event(
    reason_code: int,
    timestamp_s: float = 0.0,
) -> ThrottleEvent | None:
    """Create a ThrottleEvent if any throttle reasons are active.

    Returns None if no throttling is detected (reason_code == 0 or
    only GpuIdle).
    """
    if reason_code == 0:
        return None

    # Ignore GpuIdle-only throttling (normal when GPU is idle)
    non_idle = reason_code & ~0x0000_0001
    if non_idle == 0:
        return None

    reasons = parse_throttle_reasons(reason_code)
    critical = is_critical_throttle(reason_code)

    event = ThrottleEvent(
        timestamp_s=timestamp_s,
        reason_code=reason_code,
        reasons=reasons,
        is_critical=critical,
    )

    if critical:
        logger.warning(
            "throttle_event_critical",
            reasons=reasons,
            timestamp_s=timestamp_s,
        )
    else:
        logger.debug(
            "throttle_event",
            reasons=reasons,
            timestamp_s=timestamp_s,
        )

    return event


def summarize_throttle_events(
    events: list[ThrottleEvent],
) -> dict[str, int]:
    """Count occurrences of each throttle reason across events."""
    counts: dict[str, int] = {}
    for event in events:
        for reason in event.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts
