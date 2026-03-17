"""Shared test fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """Create a minimal config file for testing."""
    config = tmp_path / "test_config.yaml"
    config.write_text("experiment:\n  name: test\n  description: test config\n")
    return config
